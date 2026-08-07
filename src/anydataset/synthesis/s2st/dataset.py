from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from torch.utils.data import IterableDataset, get_worker_info

from ..._runtime.sharding import runtime_rank
from ...dataset import AnyDataset
from ...types import Lang, Role, Sample
from ...types._sample import select as select_sample
from .catalog import (
    FinalCatalog,
    FinalCatalogEntry,
    PairIndexRecord,
    load_catalog,
    read_pair_index,
    validate_catalog_store,
)
from .model import S2STLayout, S2STView

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotUpdate:
    previous: str | None
    current: str
    added_samples: int
    total_samples: int
    cursor: int
    wait_seconds: float


@dataclass
class _Segment:
    entry: FinalCatalogEntry
    dataset: AnyDataset
    records: tuple[PairIndexRecord, ...]


@dataclass(frozen=True)
class _Selection:
    segment: int
    local_index: int
    record: PairIndexRecord


class LiveS2STDataset(IterableDataset[Sample]):
    """Cursor-aware iterable over an atomically growing final S2ST catalog."""

    def __init__(
        self,
        root: str | Path,
        lineage_id: str,
        *,
        view: S2STView = S2STView(),
        poll_seconds: float = 5.0,
        status_seconds: float = 60.0,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.lineage_id = _non_empty("lineage_id", lineage_id)
        if not isinstance(view, S2STView):
            raise TypeError("view must be an S2STView.")
        self.view = view
        self.poll_seconds = _positive_float("poll_seconds", poll_seconds)
        self.status_seconds = _positive_float("status_seconds", status_seconds)
        if stop_requested is not None and not callable(stop_requested):
            raise TypeError("stop_requested must be callable or None.")
        self._stop_requested = stop_requested
        self._catalog = FinalCatalog(lineage_id=self.lineage_id)
        self._segments: list[_Segment] = []
        self._selected: list[_Selection] = []
        self._read_cursor = 0
        self._committed_cursor = 0

    @property
    def snapshot_id(self) -> str | None:
        return self._catalog.latest_snapshot_id

    @property
    def cursor(self) -> int:
        return self._committed_cursor

    @property
    def sample_count(self) -> int:
        return len(self._selected)

    @property
    def sealed(self) -> bool:
        return self._catalog.sealed

    def refresh(self, *, wait_seconds: float = 0.0) -> SnapshotUpdate | None:
        catalog = load_catalog(self.root, lineage_id=self.lineage_id, missing_ok=True)
        catalog.validate_successor(self._catalog)
        if len(catalog.entries) == len(self._catalog.entries):
            self._catalog = catalog
            return None
        previous = self._catalog.latest_snapshot_id
        before = len(self._selected)
        segments, selections = self._load_entries(
            catalog.entries[len(self._catalog.entries) :]
        )
        self._segments.extend(segments)
        self._selected.extend(selections)
        self._catalog = catalog
        current = cast(str, catalog.latest_snapshot_id)
        update = SnapshotUpdate(
            previous=previous,
            current=current,
            added_samples=len(self._selected) - before,
            total_samples=len(self._selected),
            cursor=self._committed_cursor,
            wait_seconds=wait_seconds,
        )
        _LOGGER.info(
            "data.snapshot.updated %s",
            json.dumps(
                {
                    "previous": update.previous,
                    "current": update.current,
                    "added_samples": update.added_samples,
                    "total_samples": update.total_samples,
                    "cursor": update.cursor,
                    "wait_seconds": update.wait_seconds,
                },
                sort_keys=True,
            ),
        )
        return update

    def wait_initial(self, *, timeout: float | None = None) -> SnapshotUpdate:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive when set.")
        started = time.monotonic()
        next_status = started
        while True:
            update = self.refresh(wait_seconds=time.monotonic() - started)
            if self.snapshot_id is not None:
                if update is None:
                    return SnapshotUpdate(
                        previous=None,
                        current=self.snapshot_id,
                        added_samples=0,
                        total_samples=len(self._selected),
                        cursor=self._committed_cursor,
                        wait_seconds=time.monotonic() - started,
                    )
                return update
            now = time.monotonic()
            if self._stopped():
                raise RuntimeError("S2ST initial snapshot wait was stopped.")
            if timeout is not None and now - started >= timeout:
                raise TimeoutError("timed out waiting for the first S2ST snapshot.")
            if now >= next_status:
                _LOGGER.info(
                    "data.snapshot.waiting %s",
                    json.dumps(
                        {"lineage_id": self.lineage_id, "wait_seconds": now - started},
                        sort_keys=True,
                    ),
                )
                next_status = now + self.status_seconds
            self._wait()

    def acknowledge(self) -> None:
        self._committed_cursor = self._read_cursor

    def state_dict(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "snapshot_id": self.snapshot_id,
            "pair_cursor": self._committed_cursor,
            "view_id": _view_id(self.view),
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("live S2ST state must be a mapping.")
        if value.get("lineage_id") != self.lineage_id:
            raise ValueError("live S2ST checkpoint lineage does not match the dataset.")
        if value.get("view_id") != _view_id(self.view):
            raise ValueError("live S2ST checkpoint view does not match the dataset.")
        snapshot = value.get("snapshot_id")
        if snapshot is not None and (not isinstance(snapshot, str) or not snapshot):
            raise TypeError(
                "live S2ST checkpoint snapshot_id must be a string or None."
            )
        cursor = value.get("pair_cursor", value.get("cursor"))
        if isinstance(cursor, bool) or not isinstance(cursor, int):
            raise TypeError("live S2ST checkpoint pair_cursor must be an integer.")
        if cursor < 0:
            raise ValueError("live S2ST checkpoint pair_cursor must be non-negative.")
        self.refresh()
        resolved_snapshot = snapshot if isinstance(snapshot, str) else None
        available = self._snapshot_sample_count(resolved_snapshot)
        if cursor > available:
            raise ValueError(
                "live S2ST checkpoint cursor is beyond its snapshot prefix: "
                f"{cursor} > {available}."
            )
        self._committed_cursor = cursor
        self._read_cursor = cursor

    def set_stop_requested(self, predicate: Callable[[], bool] | None) -> None:
        if predicate is not None and not callable(predicate):
            raise TypeError("stop predicate must be callable or None.")
        self._stop_requested = predicate

    def close(self) -> None:
        errors: list[Exception] = []
        for segment in self._segments:
            try:
                _close_dataset(segment.dataset)
            except Exception as exc:
                errors.append(exc)
        self._segments.clear()
        self._selected.clear()
        if errors:
            raise errors[0]

    def __iter__(self) -> Iterator[Sample]:
        if get_worker_info() is not None:
            raise RuntimeError("live S2ST datasets require DataLoader num_workers=0.")
        world_size, rank = runtime_rank()
        if self._committed_cursor % world_size:
            raise ValueError(
                "live S2ST checkpoint cursor is incompatible with the current world size."
            )
        self._read_cursor = self._committed_cursor
        waiting_started: float | None = None
        next_status = time.monotonic()
        while True:
            update = self.refresh(
                wait_seconds=(
                    0.0
                    if waiting_started is None
                    else time.monotonic() - waiting_started
                )
            )
            if update is not None:
                waiting_started = None
            usable = len(self._selected) // world_size * world_size
            if self._read_cursor < usable:
                selected = self._selected[self._read_cursor + rank]
                self._read_cursor += world_size
                yield self._sample(selected)
                continue
            if self._catalog.sealed:
                if len(self._selected) % world_size:
                    raise ValueError(
                        "sealed live S2ST sample count must be divisible by the "
                        f"world size: {len(self._selected)} % {world_size} != 0."
                    )
                if self._read_cursor > usable:
                    raise ValueError("live S2ST cursor is beyond the sealed catalog.")
                return
            if self._stopped():
                return
            now = time.monotonic()
            if waiting_started is None:
                waiting_started = now
            if now >= next_status:
                _LOGGER.info(
                    "data.snapshot.waiting %s",
                    json.dumps(
                        {
                            "lineage_id": self.lineage_id,
                            "snapshot_id": self.snapshot_id,
                            "cursor": self._committed_cursor,
                            "total_samples": len(self._selected),
                            "wait_seconds": now - waiting_started,
                        },
                        sort_keys=True,
                    ),
                )
                next_status = now + self.status_seconds
            self._wait()

    def _load_entries(
        self,
        entries: tuple[FinalCatalogEntry, ...],
    ) -> tuple[list[_Segment], list[_Selection]]:
        segments: list[_Segment] = []
        selections: list[_Selection] = []
        seen_sources = {
            (selected.record.source_slot, selected.record.source_row)
            for selected in self._selected
            if self.view.layout is S2STLayout.SOURCES
        }
        try:
            for entry in entries:
                records = read_pair_index(self.root, entry)
                path = self.root / entry.store_path
                validate_catalog_store(self.root, entry)
                dataset = AnyDataset.from_store(path)
                try:
                    if len(dataset) != len(records):
                        raise ValueError(
                            "final catalog store and pair index counts differ."
                        )
                except Exception:
                    _close_dataset(dataset)
                    raise
                segment_index = len(self._segments) + len(segments)
                segments.append(_Segment(entry, dataset, records))
                for local_index, record in enumerate(records):
                    if not _matches(record, self.view):
                        continue
                    source = (record.source_slot, record.source_row)
                    if self.view.layout is S2STLayout.SOURCES:
                        if not record.first_for_source or source in seen_sources:
                            continue
                        seen_sources.add(source)
                    selections.append(_Selection(segment_index, local_index, record))
        except Exception:
            for segment in segments:
                _close_dataset(segment.dataset)
            raise
        return segments, selections

    def _snapshot_sample_count(self, snapshot_id: str | None) -> int:
        if snapshot_id is None:
            if self._committed_cursor:
                raise ValueError(
                    "live S2ST checkpoint without a snapshot must have cursor zero."
                )
            return 0
        segment_limit: int | None = None
        for index, segment in enumerate(self._segments):
            if segment.entry.snapshot_id == snapshot_id:
                segment_limit = index
                break
        if segment_limit is None:
            raise ValueError(
                f"live S2ST checkpoint snapshot is not in the catalog: {snapshot_id!r}."
            )
        return sum(selected.segment <= segment_limit for selected in self._selected)

    def _sample(self, selected: _Selection) -> Sample:
        sample = self._segments[selected.segment].dataset[selected.local_index]
        return project_sample(sample, self.view)

    def _stopped(self) -> bool:
        return self._stop_requested is not None and bool(self._stop_requested())

    def _wait(self) -> None:
        deadline = time.monotonic() + self.poll_seconds
        while time.monotonic() < deadline:
            if self._stopped():
                return
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))


def project_sample(sample: Sample, view: S2STView) -> Sample:
    if view.layout is S2STLayout.SOURCES:
        sample = {
            reference: item
            for reference, item in sample.items()
            if reference[0] is Role.SOURCE
        }
    if view.schema is not None:
        sample = select_sample(sample, view.schema)
    return sample


def _matches(record: PairIndexRecord, view: S2STView) -> bool:
    if (
        view.source_languages is not None
        and Lang(record.source_language) not in view.source_languages
    ):
        return False
    if (
        view.target_languages is not None
        and Lang(record.target_language) not in view.target_languages
    ):
        return False
    if view.source_slots is not None and record.source_slot not in view.source_slots:
        return False
    return not (view.speakers is not None and record.speaker_id not in view.speakers)


def _view_id(view: S2STView) -> str:
    schema = None
    if view.schema is not None:
        schema = sorted(
            (
                role.value,
                modality.value,
                sorted(value.value for value in requirement.views),
                sorted(value.value for value in requirement.meta),
            )
            for (role, modality), requirement in view.schema.items()
        )
    payload: dict[str, Any] = {
        "layout": view.layout.value,
        "source_languages": _values(view.source_languages),
        "target_languages": _values(view.target_languages),
        "source_slots": None
        if view.source_slots is None
        else sorted(view.source_slots),
        "speakers": None if view.speakers is None else sorted(view.speakers),
        "schema": schema,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _values(value) -> list[str] | None:
    return None if value is None else sorted(item.value for item in value)


def _non_empty(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return value


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _close_dataset(dataset: AnyDataset) -> None:
    close = getattr(dataset, "close", None)
    if callable(close):
        close()
        return
    prepared = getattr(dataset, "dataset", None)
    close = getattr(prepared, "close", None)
    if callable(close):
        close()


__all__ = ["LiveS2STDataset", "SnapshotUpdate", "project_sample"]
