"""Independent S2ST stage production over fixed published prefixes."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from ...dataset import MapStyleABC
from ...dataset.universe import SampleIdentity
from ...types import Sample
from .catalog import (
    CatalogPublisher,
    FinalCatalog,
    PairIndexRecord,
    SnapshotManifest,
    load_catalog,
    read_pair_index,
    store_digest,
)
from .dataset import S2STDataset, S2STStatus, status as stage_status
from .model import S2STStage


_UPSTREAM = {
    S2STStage.TRANSLATION: S2STStage.SOURCE,
    S2STStage.TTS: S2STStage.TRANSLATION,
}


class StageInput(MapStyleABC):
    """A fixed read-only suffix from the currently published upstream stage."""

    def __init__(
        self,
        dataset: S2STDataset,
        *,
        root: Path,
        lineage_id: str,
        stage: S2STStage,
        start: int,
        stop: int,
        config_revision: str,
        records: tuple[PairIndexRecord, ...],
        catalog: FinalCatalog,
        watermark_digest: str,
    ) -> None:
        if start < 0 or stop <= start or stop > len(dataset):
            dataset.close()
            raise ValueError("stage input range must be a non-empty published suffix.")
        if len(records) != stop - start:
            dataset.close()
            raise ValueError("stage input records do not cover its sample range.")
        self.root = root
        self.lineage_id = lineage_id
        self.stage = stage
        self.start = start
        self.stop = stop
        self.config_revision = config_revision
        self._dataset = dataset
        self._records = records
        self._catalog = catalog
        self._watermark_digest = watermark_digest
        self._closed = False

    @property
    def sample_count(self) -> int:
        return self.stop - self.start

    def __len__(self) -> int:
        self._ensure_open()
        return self.sample_count

    def __getitem__(self, index: int) -> Sample:
        return self._dataset[self.start + _position(index, len(self))]

    def __getitems__(self, indexes: Sequence[int]) -> list[Sample]:
        source = tuple(self.start + _position(index, len(self)) for index in indexes)
        return self._dataset.__getitems__(source)

    def sample_id(self, index: int) -> str:
        identity = cast(SampleIdentity, self._dataset)
        return identity.sample_id(self.start + _position(index, len(self)))

    def global_index(self, index: int) -> int:
        return self.start + _position(index, len(self))

    def universe_id(self) -> str:
        self._ensure_open()
        return (
            f"s2st-stage-input-v1:{self.stage.value}:{self.start}:{self.stop}:"
            f"{self._watermark_digest}"
        )

    def cost_row(self, index: int) -> Any:
        return self._dataset.cost_row(self.start + _position(index, len(self)))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._dataset.close()

    def __enter__(self) -> StageInput:
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
            raise RuntimeError("StageInput is closed.")


class StagePublisher:
    """Advance one S2ST stage without sharing another stage's writer lifecycle."""

    def __init__(
        self,
        root: str | Path,
        lineage_id: str,
        *,
        stage: S2STStage,
        lock_timeout: float = 3600.0,
    ) -> None:
        if not isinstance(stage, S2STStage):
            raise TypeError("stage must be an S2STStage.")
        self.root = Path(root)
        self.lineage_id = _non_empty("lineage_id", lineage_id)
        self.stage = stage
        self._publisher = CatalogPublisher(
            self.root,
            self.lineage_id,
            stage=stage,
            lock_timeout=lock_timeout,
        )

    def status(self) -> S2STStatus:
        return stage_status(
            self.root,
            self.lineage_id,
            stage=self.stage,
        )

    def open_input(self, *, max_samples: int | None = None) -> StageInput | None:
        """Pin the next missing upstream suffix for translation or TTS work."""

        upstream_stage = _UPSTREAM.get(self.stage)
        if upstream_stage is None:
            raise TypeError("the source stage has no upstream stage input.")
        if max_samples is not None:
            max_samples = _positive_int("max_samples", max_samples)

        current = self._publisher.load()
        upstream = S2STDataset(
            self.root,
            self.lineage_id,
            stage=upstream_stage,
        )
        catalog = upstream._catalog
        start = current.sample_count
        if start > catalog.sample_count:
            upstream.close()
            raise ValueError(
                f"{self.stage.value} coverage exceeds its {upstream_stage.value} input."
            )
        if start == catalog.sample_count:
            upstream.close()
            return None

        entry = next(
            value
            for value in catalog.entries
            if value.start <= start < value.start + value.sample_count
        )
        stop = entry.start + entry.sample_count
        if max_samples is not None:
            stop = min(stop, start + max_samples)
        records = _catalog_records(self.root, catalog, start=start, stop=stop)
        digest = _watermark_digest(catalog, stop)
        return StageInput(
            upstream,
            root=self.root,
            lineage_id=self.lineage_id,
            stage=upstream_stage,
            start=start,
            stop=stop,
            config_revision=entry.config_revision,
            records=records,
            catalog=catalog,
            watermark_digest=digest,
        )

    def publish_source(
        self,
        store: str | Path,
        records: Sequence[PairIndexRecord],
        *,
        config_revision: str,
    ) -> S2STStatus:
        """Publish one source-stage delta planned by the S2ST growth scheduler."""

        if self.stage is not S2STStage.SOURCE:
            raise TypeError("publish_source() requires a source-stage publisher.")
        values = _records(records)
        current = self._publisher.load()
        manifest = self._manifest(
            current,
            store,
            values,
            config_revision=_non_empty("config_revision", config_revision),
            upstream=None,
        )
        self._publisher.publish(manifest, values)
        return self.status()

    def publish(self, stage_input: StageInput, store: str | Path) -> S2STStatus:
        """Publish one downstream delta computed from a pinned ``StageInput``."""

        expected_upstream = _UPSTREAM.get(self.stage)
        if expected_upstream is None:
            raise TypeError("source stages publish with publish_source().")
        if not isinstance(stage_input, StageInput):
            raise TypeError("stage_input must be a StageInput.")
        stage_input._ensure_open()
        if (
            stage_input.root != self.root
            or stage_input.lineage_id != self.lineage_id
            or stage_input.stage is not expected_upstream
        ):
            raise ValueError("stage input does not belong to this stage publisher.")

        current = self._publisher.load()
        if current.sample_count != stage_input.start:
            raise RuntimeError("stage input is stale; reopen the next upstream suffix.")
        latest_upstream = load_catalog(
            self.root,
            lineage_id=self.lineage_id,
            stage=expected_upstream,
        )
        latest_upstream.validate_successor(stage_input._catalog)
        if (
            _watermark_digest(latest_upstream, stage_input.stop)
            != stage_input._watermark_digest
        ):
            raise ValueError("stage input watermark changed before publication.")

        manifest = self._manifest(
            current,
            store,
            stage_input._records,
            config_revision=stage_input.config_revision,
            upstream=stage_input,
        )
        self._publisher.publish(manifest, stage_input._records)
        return self.status()

    def seal(self) -> S2STStatus:
        upstream_stage = _UPSTREAM.get(self.stage)
        if upstream_stage is not None:
            current = self._publisher.load()
            upstream = load_catalog(
                self.root,
                lineage_id=self.lineage_id,
                stage=upstream_stage,
                missing_ok=True,
            )
            if not upstream.sealed or current.sample_count != upstream.sample_count:
                raise ValueError(
                    f"cannot seal {self.stage.value} before its sealed "
                    f"{upstream_stage.value} prefix is fully covered."
                )
        self._publisher.seal()
        return self.status()

    def _manifest(
        self,
        current: FinalCatalog,
        store: str | Path,
        records: tuple[PairIndexRecord, ...],
        *,
        config_revision: str,
        upstream: StageInput | None,
    ) -> SnapshotManifest:
        relative = _store_path(self.root, store)
        absolute = self.root / relative
        revision = 0 if not current.entries else current.entries[-1].revision + 1
        added_sources, total_sources = _source_counts(current, records)
        stop = current.sample_count + len(records)
        digest = store_digest(absolute)
        commit_id = _commit_id(
            self.stage,
            revision=revision,
            start=current.sample_count,
            stop=stop,
            store_digest=digest,
        )
        return SnapshotManifest(
            lineage_id=self.lineage_id,
            config_revision=config_revision,
            revision=revision,
            stage=self.stage,
            snapshot_id=commit_id,
            upstream_snapshot_id=(
                None
                if upstream is None
                else f"{upstream.stage.value}-prefix-{upstream.stop}"
            ),
            upstream_digest=(
                None if upstream is None else upstream._watermark_digest
            ),
            previous_snapshot_id=current.latest_snapshot_id,
            added_sources=added_sources,
            added_pairs=len(records),
            total_sources=total_sources,
            total_pairs=stop,
            coverage=(("all", stop),),
            store_path=relative,
            store_digest=digest,
        )


def _catalog_records(
    root: Path,
    catalog: FinalCatalog,
    *,
    start: int,
    stop: int,
) -> tuple[PairIndexRecord, ...]:
    records: list[PairIndexRecord] = []
    for entry in catalog.entries:
        entry_stop = entry.start + entry.sample_count
        if entry_stop <= start:
            continue
        if entry.start >= stop:
            break
        values = read_pair_index(root, entry)
        local_start = max(start, entry.start) - entry.start
        local_stop = min(stop, entry_stop) - entry.start
        records.extend(values[local_start:local_stop])
    if len(records) != stop - start:
        raise ValueError("upstream pair index does not cover the requested stage input.")
    return tuple(records)


def _watermark_digest(catalog: FinalCatalog, stop: int) -> str:
    if stop < 0 or stop > catalog.sample_count:
        raise ValueError("stage watermark is outside the published upstream prefix.")
    entries = []
    for entry in catalog.entries:
        if entry.start >= stop:
            break
        entries.append(
            {
                "revision": entry.revision,
                "start": entry.start,
                "sample_count": min(entry.sample_count, stop - entry.start),
                "store_digest": entry.store_digest,
                "index_digest": entry.index_digest,
            }
        )
    payload = json.dumps(
        {
            "schema": "anydataset-s2st-stage-watermark-v1",
            "lineage_id": catalog.lineage_id,
            "stage": catalog.stage.value,
            "stop": stop,
            "entries": entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_counts(
    catalog: FinalCatalog,
    records: tuple[PairIndexRecord, ...],
) -> tuple[int, int]:
    current = 0 if not catalog.entries else catalog.entries[-1].total_sources
    added = len({record.source_sequence for record in records if record.source_sequence >= current})
    return added, current + added


def _records(records: Sequence[PairIndexRecord]) -> tuple[PairIndexRecord, ...]:
    values = tuple(records)
    if not values:
        raise ValueError("stage publication requires at least one sample.")
    if any(not isinstance(record, PairIndexRecord) for record in values):
        raise TypeError("records must contain PairIndexRecord values.")
    return values


def _store_path(root: Path, value: str | Path) -> str:
    path = Path(value)
    absolute = path if path.is_absolute() else root / path
    root_path = root.resolve()
    try:
        relative = absolute.resolve().relative_to(root_path)
    except ValueError as exc:
        raise ValueError("stage store must be located below the S2ST root.") from exc
    if not relative.parts:
        raise ValueError("stage store must not be the S2ST root.")
    return str(relative)


def _commit_id(
    stage: S2STStage,
    *,
    revision: int,
    start: int,
    stop: int,
    store_digest: str,
) -> str:
    payload = f"{stage.value}:{revision}:{start}:{stop}:{store_digest}".encode()
    return f"{stage.value}-{revision:08d}-{hashlib.sha256(payload).hexdigest()[:16]}"


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


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
        raise IndexError("stage input index out of range")
    return position


__all__ = ["StageInput", "StagePublisher"]
