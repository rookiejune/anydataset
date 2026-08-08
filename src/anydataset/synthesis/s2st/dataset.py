"""Expose one fixed published S2ST prefix as a map-style dataset.

Each internal catalog entry is one immutable delta. Construction validates and
loads the prefix visible at that instant. Reopen the dataset to observe later
publications; an existing reader never changes length.
"""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from ...dataset import AnyDataset, MapStyleABC
from ...dataset._snapshot import SnapshotCatalogDataset, SnapshotSegment
from ...dataset.universe import SampleIdentity
from ...types import Lang, Role, Sample
from ...types._sample import select as select_sample
from .catalog import (
    FinalCatalog,
    PairIndexRecord,
    catalog_path,
    load_catalog,
    read_pair_index,
    validate_catalog_entry,
)
from .model import S2STLayout, S2STStage, S2STView


@dataclass(frozen=True)
class S2STStatus:
    """Published sample coverage for one independently advancing S2ST stage."""

    root: Path
    lineage_id: str
    stage: S2STStage
    expected_samples: int | None
    published_samples: int
    sealed: bool

    @property
    def missing(self) -> bool:
        return self.published_samples == 0

    @property
    def pending_samples(self) -> int | None:
        if self.expected_samples is None:
            return None
        return self.expected_samples - self.published_samples

    @property
    def caught_up(self) -> bool:
        return self.pending_samples == 0


class S2STDataset(MapStyleABC):
    """The immutable published prefix visible when this reader was opened."""

    def __init__(
        self,
        root: str | Path,
        lineage_id: str,
        *,
        stage: S2STStage = S2STStage.TTS,
        view: S2STView = S2STView(),
    ) -> None:
        self.root = Path(root)
        self.lineage_id = _non_empty("lineage_id", lineage_id)
        if not isinstance(stage, S2STStage):
            raise TypeError("stage must be an S2STStage.")
        self.stage = stage
        if not isinstance(view, S2STView):
            raise TypeError("view must be an S2STView.")
        self.view = view
        catalog, _catalog_version = _load_versioned_catalog(
            self.root,
            lineage_id=self.lineage_id,
            stage=self.stage,
            missing_ok=True,
        )
        base, indices, segments = _open_view(self.root, catalog, view)
        self._catalog = catalog
        self._base = base
        self._indices = indices
        self._snapshot_segments = segments
        self._universe_id = _view_lineage_id(self.lineage_id, self.stage, view)
        self._closed = False

    @property
    def sealed(self) -> bool:
        return self._catalog.sealed

    @property
    def snapshot_segments(self) -> tuple[SnapshotSegment, ...]:
        """Immutable catalog segments included in this fixed reader prefix."""

        self._ensure_open()
        return self._snapshot_segments

    def __len__(self) -> int:
        self._ensure_open()
        return len(self._indices)

    def __getitem__(self, index: int) -> Sample:
        position = _position(index, len(self))
        return project_sample(self._base[self._indices[position]], self.view)

    def __getitems__(self, indexes: Sequence[int]) -> list[Sample]:
        positions = tuple(_position(index, len(self)) for index in indexes)
        source_indexes = tuple(self._indices[index] for index in positions)
        return [
            project_sample(sample, self.view)
            for sample in self._base.__getitems__(source_indexes)
        ]

    def sample_id(self, index: int) -> str:
        position = _position(index, len(self))
        return self._base.sample_id(self._indices[position])

    def global_index(self, index: int) -> int:
        position = _position(index, len(self))
        return self._base.global_index(self._indices[position])

    def universe_id(self) -> str | None:
        self._ensure_open()
        return self._universe_id

    def cost_row(self, index: int) -> Any:
        position = _position(index, len(self))
        return self._base.cost_row(self._indices[position])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._base.close()

    def __enter__(self) -> S2STDataset:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("S2STDataset is closed.")


def status(
    root: str | Path,
    lineage_id: str,
    *,
    stage: S2STStage = S2STStage.TTS,
) -> S2STStatus:
    """Validate and summarize the currently published stage prefix."""

    path = Path(root)
    identity = _non_empty("lineage_id", lineage_id)
    if not isinstance(stage, S2STStage):
        raise TypeError("stage must be an S2STStage.")
    catalog = load_catalog(
        path,
        lineage_id=identity,
        stage=stage,
        missing_ok=True,
    )
    for entry in catalog.entries:
        validate_catalog_entry(path, entry)
    expected = _upstream_sample_count(path, identity, stage)
    if expected is None and catalog.sealed:
        expected = catalog.sample_count
    return S2STStatus(
        root=path,
        lineage_id=identity,
        stage=stage,
        expected_samples=expected,
        published_samples=catalog.sample_count,
        sealed=catalog.sealed,
    )


def _upstream_sample_count(
    root: Path,
    lineage_id: str,
    stage: S2STStage,
) -> int | None:
    upstream = {
        S2STStage.TRANSLATION: S2STStage.SOURCE,
        S2STStage.TTS: S2STStage.TRANSLATION,
    }.get(stage)
    if upstream is None:
        return None
    return load_catalog(
        root,
        lineage_id=lineage_id,
        stage=upstream,
        missing_ok=True,
    ).sample_count


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


class _SnapshotViewDataset(MapStyleABC):
    """Borrow one physical segment and expose its selected S2ST logical rows."""

    def __init__(
        self,
        dataset: MapStyleABC,
        indexes: Sequence[int],
        view: S2STView,
        *,
        universe_id: str,
    ) -> None:
        self._dataset = dataset
        self._indexes = tuple(indexes)
        self._view = view
        self._universe_id = universe_id

    def __len__(self) -> int:
        return len(self._indexes)

    def __getitem__(self, index: int) -> Sample:
        source_index = self._source_index(index)
        return project_sample(self._dataset[source_index], self._view)

    def __getitems__(self, indexes: Sequence[int]) -> list[Sample]:
        source_indexes = tuple(self._source_index(index) for index in indexes)
        getitems = getattr(self._dataset, "__getitems__", None)
        if callable(getitems):
            resolved = getitems(source_indexes)
            if not isinstance(resolved, Sequence):
                raise TypeError("S2ST snapshot batch read must return a sequence.")
            samples = cast(Sequence[Sample], resolved)
        else:
            samples = [self._dataset[index] for index in source_indexes]
        if len(samples) != len(source_indexes):
            raise ValueError(
                "S2ST snapshot batch read returned the wrong sample count."
            )
        return [project_sample(sample, self._view) for sample in samples]

    def sample_id(self, index: int) -> str:
        identity = cast(SampleIdentity, self._dataset)
        return identity.sample_id(self._source_index(index))

    def global_index(self, index: int) -> int:
        return _position(index, len(self))

    def universe_id(self) -> str:
        return self._universe_id

    def cost_row(self, index: int) -> Any:
        return self._dataset.cost_row(self._source_index(index))

    def _source_index(self, index: int) -> int:
        return self._indexes[_position(index, len(self._indexes))]


def _open_catalog(
    root: Path,
    catalog: FinalCatalog,
) -> tuple[SnapshotCatalogDataset, tuple[PairIndexRecord, ...]]:
    segments: list[SnapshotSegment] = []
    records: list[PairIndexRecord] = []
    try:
        for entry in catalog.entries:
            validate_catalog_entry(root, entry)
            entry_records = read_pair_index(root, entry)
            dataset = AnyDataset.from_store(root / entry.store_path)
            segments.append(
                SnapshotSegment(
                    snapshot_id=entry.snapshot_id,
                    start=entry.start,
                    sample_count=entry.sample_count,
                    dataset=dataset,
                )
            )
            records.extend(entry_records)
    except BaseException:
        for segment in reversed(segments):
            _close_dataset(segment.dataset)
        raise
    return (
        SnapshotCatalogDataset(
            segments,
            sealed=catalog.sealed,
            universe_id=_catalog_id(catalog),
        ),
        tuple(records),
    )


def _open_view(
    root: Path,
    catalog: FinalCatalog,
    view: S2STView,
) -> tuple[SnapshotCatalogDataset, tuple[int, ...], tuple[SnapshotSegment, ...]]:
    base, records = _open_catalog(root, catalog)
    try:
        indices, segments = _select_catalog(
            base,
            records,
            view,
            lineage_id=catalog.lineage_id,
            stage=catalog.stage,
            view_id=_view_id(view),
        )
    except BaseException:
        base.close()
        raise
    return base, indices, segments


def _select_catalog(
    catalog_dataset: SnapshotCatalogDataset,
    records: Sequence[PairIndexRecord],
    view: S2STView,
    *,
    lineage_id: str,
    stage: S2STStage,
    view_id: str,
) -> tuple[tuple[int, ...], tuple[SnapshotSegment, ...]]:
    if len(records) != len(catalog_dataset):
        raise ValueError("S2ST pair index does not cover the complete catalog.")
    selected: list[int] = []
    segments: list[SnapshotSegment] = []
    seen_sources: set[tuple[str, int]] = set()
    logical_start = 0
    for segment in catalog_dataset.snapshot_segments:
        local_indexes: list[int] = []
        for local_index, record in enumerate(records[segment.start : segment.stop]):
            if not _matches(record, view):
                continue
            source = (record.source_slot, record.source_row)
            if view.layout is S2STLayout.SOURCES:
                if not record.first_for_source or source in seen_sources:
                    continue
                seen_sources.add(source)
            selected.append(segment.start + local_index)
            local_indexes.append(local_index)
        if not local_indexes:
            continue
        dataset = _SnapshotViewDataset(
            segment.dataset,
            local_indexes,
            view,
            universe_id=_snapshot_view_id(
                lineage_id,
                stage,
                segment.snapshot_id,
                view_id,
            ),
        )
        segments.append(
            SnapshotSegment(
                snapshot_id=segment.snapshot_id,
                start=logical_start,
                sample_count=len(dataset),
                dataset=dataset,
            )
        )
        logical_start += len(dataset)
    return tuple(selected), tuple(segments)


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


def _catalog_id(catalog: FinalCatalog) -> str:
    payload = {
        "schema": "anydataset-s2st-catalog-v1",
        "lineage_id": catalog.lineage_id,
        "stage": catalog.stage.value,
        "entries": [
            {
                "snapshot_id": entry.snapshot_id,
                "start": entry.start,
                "sample_count": entry.sample_count,
                "store_identity": entry.store_identity,
                "index_identity": entry.index_identity,
            }
            for entry in catalog.entries
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"s2st-catalog-v1:{hashlib.sha256(encoded).hexdigest()}"


def _view_lineage_id(
    lineage_id: str,
    stage: S2STStage,
    view: S2STView,
) -> str:
    payload = {
        "schema": "anydataset-s2st-view-lineage-v1",
        "lineage_id": lineage_id,
        "stage": stage.value,
        "view": _view_id(view),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"s2st-view-lineage-v1:{hashlib.sha256(encoded).hexdigest()}"


def _snapshot_view_id(
    lineage_id: str,
    stage: S2STStage,
    snapshot_id: str,
    view_id: str,
) -> str:
    payload = {
        "schema": "anydataset-s2st-view-snapshot-v1",
        "lineage_id": lineage_id,
        "stage": stage.value,
        "snapshot_id": snapshot_id,
        "view": view_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"s2st-view-snapshot-v1:{hashlib.sha256(encoded).hexdigest()}"


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
        "source_languages": _language_values(view.source_languages),
        "target_languages": _language_values(view.target_languages),
        "source_slots": (
            None if view.source_slots is None else sorted(view.source_slots)
        ),
        "speakers": None if view.speakers is None else sorted(view.speakers),
        "schema": schema,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _language_values(value: frozenset[Lang] | None) -> list[str] | None:
    return None if value is None else sorted(item.value for item in value)


def _non_empty(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return value


def _position(index: int, length: int) -> int:
    position = operator.index(index)
    if position < 0:
        position += length
    if position < 0 or position >= length:
        raise IndexError("S2ST snapshot index out of range")
    return position


def _close_dataset(dataset: MapStyleABC) -> None:
    close = getattr(dataset, "close", None)
    if callable(close):
        close()
        return
    prepared = getattr(dataset, "dataset", None)
    close = getattr(prepared, "close", None)
    if callable(close):
        close()


_CatalogVersion = tuple[int, int, int, int]


def _catalog_version(root: Path, stage: S2STStage) -> _CatalogVersion:
    stat = catalog_path(root, stage).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _load_versioned_catalog(
    root: Path,
    *,
    lineage_id: str,
    stage: S2STStage,
    missing_ok: bool = False,
) -> tuple[FinalCatalog, _CatalogVersion]:
    path = catalog_path(root, stage)
    if not path.is_file() and missing_ok:
        return (
            load_catalog(
                root,
                lineage_id=lineage_id,
                stage=stage,
                missing_ok=True,
            ),
            (0, 0, 0, 0),
        )
    for _attempt in range(3):
        before = _catalog_version(root, stage)
        catalog = load_catalog(root, lineage_id=lineage_id, stage=stage)
        after = _catalog_version(root, stage)
        if before == after:
            return catalog, after
    raise RuntimeError("S2ST catalog changed repeatedly while being read.")


__all__ = ["S2STDataset", "S2STStatus", "project_sample", "status"]
