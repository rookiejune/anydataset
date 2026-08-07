"""Append-only snapshot catalogs for online materialization.

The catalog root is the coordination address shared by one explicit producer
and any number of read-only consumers.  Every catalog entry names an immutable
canonical store containing one dense delta segment.  A consumer opens the
catalog once and therefore observes a fixed prefix for its whole lifetime.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ...dataset._snapshot import SnapshotCatalogDataset, SnapshotSegment
from ...types.item import Modality, Role, Schema, View
from .._identity import materialized_universe_id
from ..jsonio import read_json, write_json
from ..manifest.schema import normalize_provenance
from ..part.commit import commit_store_fragments
from ..paths import dataset_ready_path
from ..reader import StoreDataset, read_store_dataset

CATALOG_SCHEMA_VERSION = 1
CATALOG_FILENAME = "catalog.json"
SNAPSHOTS_DIRNAME = "snapshots"
PRODUCER_LOCK_FILENAME = ".producer.lock"


@dataclass(frozen=True)
class SnapshotEntry:
    revision: int
    start: int
    sample_count: int
    store_path: str

    @property
    def stop(self) -> int:
        return self.start + self.sample_count


@dataclass(frozen=True)
class SnapshotCatalog:
    dataset_id: str
    split: str | None
    provenance: Mapping[str, str]
    entries: tuple[SnapshotEntry, ...] = ()
    sealed: bool = False

    @property
    def sample_count(self) -> int:
        return 0 if not self.entries else self.entries[-1].stop


class SnapshotDataset(SnapshotCatalogDataset):
    """A fixed map-style concatenation of immutable materialization segments."""

    def __init__(
        self,
        *,
        root: Path,
        catalog: SnapshotCatalog,
        stores: tuple[StoreDataset, ...],
        views: tuple[tuple[Role, Modality, View], ...],
    ) -> None:
        segments = tuple(
            SnapshotSegment(
                snapshot_id=f"materialized-{entry.revision}",
                start=entry.start,
                sample_count=entry.sample_count,
                dataset=store,
            )
            for entry, store in zip(catalog.entries, stores)
        )
        super().__init__(segments, sealed=catalog.sealed)
        self.root = root
        self.catalog = catalog
        self._views = views

    def universe_id(self) -> str | None:
        self._ensure_open()
        return materialized_universe_id(
            self.catalog.dataset_id,
            self.catalog.split,
            self.catalog.provenance,
            len(self),
            self._views,
        )

class SnapshotPublisher:
    """Publish durable dense fragment prefixes while a producer lease is held."""

    def __init__(
        self,
        root: str | Path,
        fragments_dir: str | Path,
        *,
        dataset_id: str,
        split: str | None,
        provenance: Mapping[str, str],
        schema: Schema,
        snapshot_samples: int,
        max_shard_samples: int | None,
        max_shard_bytes: int | None,
        completed: Collection[int],
    ) -> None:
        if isinstance(snapshot_samples, bool) or not isinstance(snapshot_samples, int):
            raise TypeError("snapshot_samples must be an integer.")
        if snapshot_samples <= 0:
            raise ValueError("snapshot_samples must be positive.")
        self.root = Path(root).expanduser()
        self.fragments_dir = Path(fragments_dir).expanduser()
        self.dataset_id = dataset_id
        self.split = split
        self.provenance = normalize_provenance(provenance)
        self.views = _schema_views(schema)
        self.snapshot_samples = snapshot_samples
        self.max_shard_samples = max_shard_samples
        self.max_shard_bytes = max_shard_bytes
        catalog = load_catalog(
            self.root,
            dataset_id=dataset_id,
            split=split,
            provenance=self.provenance,
            missing_ok=True,
        )
        if catalog is None:
            catalog = SnapshotCatalog(
                dataset_id=dataset_id,
                split=split,
                provenance=self.provenance,
            )
            _initialize_catalog_root(self.root)
            write_catalog(self.root, catalog)
        else:
            validated = open_snapshot_dataset(
                self.root,
                dataset_id=dataset_id,
                split=split,
                provenance=self.provenance,
                schema=schema,
            )
            validated.close()
        self.catalog: SnapshotCatalog = catalog
        self._completed = set(completed)
        self._dense_stop = 0
        while self._dense_stop in self._completed:
            self._dense_stop += 1
        if self.catalog.sample_count > self._dense_stop:
            raise ValueError(
                "Materialization catalog contains samples missing from staging."
            )

    @property
    def sample_count(self) -> int:
        return self.catalog.sample_count

    def record(self, indexes: Sequence[int]) -> None:
        self._completed.update(indexes)
        while self._dense_stop in self._completed:
            self._dense_stop += 1
        self.publish(force=False)

    def publish(self, *, force: bool) -> None:
        start = self.catalog.sample_count
        stop = self._dense_stop
        if stop <= start:
            return
        if not force and stop - start < self.snapshot_samples:
            return
        if self.catalog.sealed:
            raise ValueError("Cannot append to a sealed materialization catalog.")
        revision = len(self.catalog.entries)
        relative = (
            f"{SNAPSHOTS_DIRNAME}/{revision:08d}-{start:012d}-{stop:012d}"
        )
        target = self.root / relative
        count = stop - start
        if target.exists():
            _validate_segment_store(
                target,
                dataset_id=self.dataset_id,
                split=self.split,
                provenance=self.provenance,
                sample_count=count,
                views=self.views,
            ).close()
        else:
            commit_store_fragments(
                target,
                self.fragments_dir,
                dataset_id=self.dataset_id,
                split=self.split,
                expected_sample_count=count,
                sample_start=start,
                max_shard_samples=self.max_shard_samples,
                max_shard_bytes=self.max_shard_bytes,
                provenance=self.provenance,
            )
        entry = SnapshotEntry(revision, start, count, relative)
        self.catalog = SnapshotCatalog(
            dataset_id=self.catalog.dataset_id,
            split=self.catalog.split,
            provenance=self.catalog.provenance,
            entries=self.catalog.entries + (entry,),
            sealed=False,
        )
        write_catalog(self.root, self.catalog)

    def seal(self) -> None:
        if self.catalog.sealed:
            return
        self.publish(force=True)
        if self.catalog.sample_count != self._dense_stop:
            raise ValueError("Cannot seal an incomplete materialization catalog.")
        self.catalog = SnapshotCatalog(
            dataset_id=self.catalog.dataset_id,
            split=self.catalog.split,
            provenance=self.catalog.provenance,
            entries=self.catalog.entries,
            sealed=True,
        )
        write_catalog(self.root, self.catalog)


def producer_lock_path(root: str | Path) -> Path:
    return Path(root).expanduser() / PRODUCER_LOCK_FILENAME


def open_snapshot_dataset(
    root: str | Path,
    *,
    dataset_id: str,
    split: str | None,
    provenance: Mapping[str, str],
    schema: Schema,
) -> SnapshotDataset:
    path = Path(root).expanduser()
    views = _schema_views(schema)
    catalog = load_catalog(
        path,
        dataset_id=dataset_id,
        split=split,
        provenance=provenance,
        missing_ok=True,
    )
    if catalog is None:
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"Materialization root is not a directory: {path}")
            unexpected = {
                item.name
                for item in path.iterdir()
                if item.name != PRODUCER_LOCK_FILENAME
            }
            if unexpected:
                raise ValueError(
                    f"Materialization root exists without {CATALOG_FILENAME}: {path}"
                )
        catalog = SnapshotCatalog(
            dataset_id=dataset_id,
            split=split,
            provenance=normalize_provenance(provenance),
        )
    stores: list[StoreDataset] = []
    try:
        for entry in catalog.entries:
            store = _validate_segment_store(
                _entry_path(path, entry),
                dataset_id=dataset_id,
                split=split,
                provenance=provenance,
                sample_count=entry.sample_count,
                views=views,
            )
            stores.append(store)
    except BaseException:
        for store in stores:
            store.close()
        raise
    return SnapshotDataset(
        root=path,
        catalog=catalog,
        stores=tuple(stores),
        views=views,
    )


def load_catalog(
    root: str | Path,
    *,
    dataset_id: str,
    split: str | None,
    provenance: Mapping[str, str],
    missing_ok: bool = False,
) -> SnapshotCatalog | None:
    path = Path(root).expanduser() / CATALOG_FILENAME
    if not path.is_file():
        if missing_ok:
            return None
        raise FileNotFoundError(path)
    catalog = _catalog_from_dict(read_json(path))
    if catalog.dataset_id != dataset_id:
        raise ValueError("Materialization catalog dataset_id does not match.")
    if catalog.split != split:
        raise ValueError("Materialization catalog split does not match.")
    if dict(catalog.provenance) != normalize_provenance(provenance):
        raise ValueError("Materialization catalog provenance does not match.")
    return catalog


def write_catalog(root: str | Path, catalog: SnapshotCatalog) -> Path:
    path = Path(root).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / CATALOG_FILENAME, _catalog_dict(catalog))
    return path / CATALOG_FILENAME


def _catalog_dict(catalog: SnapshotCatalog) -> dict[str, Any]:
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "dataset_id": catalog.dataset_id,
        "split": catalog.split,
        "provenance": dict(catalog.provenance),
        "sealed": catalog.sealed,
        "entries": [
            {
                "revision": entry.revision,
                "start": entry.start,
                "sample_count": entry.sample_count,
                "store_path": entry.store_path,
            }
            for entry in catalog.entries
        ],
    }


def _catalog_from_dict(value: object) -> SnapshotCatalog:
    if not isinstance(value, Mapping):
        raise TypeError("Materialization catalog must be a mapping.")
    expected_fields = {
        "schema_version",
        "dataset_id",
        "split",
        "provenance",
        "sealed",
        "entries",
    }
    if set(value) != expected_fields:
        raise ValueError("Materialization catalog fields do not match schema v1.")
    if value["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise ValueError("Unsupported materialization catalog schema_version.")
    dataset_id = value["dataset_id"]
    split = value["split"]
    sealed = value["sealed"]
    raw_entries = value["entries"]
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("Materialization catalog dataset_id must be non-empty.")
    if split is not None and not isinstance(split, str):
        raise TypeError("Materialization catalog split must be a string or None.")
    if type(sealed) is not bool:
        raise TypeError("Materialization catalog sealed must be a boolean.")
    if not isinstance(raw_entries, list):
        raise TypeError("Materialization catalog entries must be a list.")
    provenance = normalize_provenance(cast(Mapping[str, str], value["provenance"]))
    entries: list[SnapshotEntry] = []
    start = 0
    for revision, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping) or set(raw) != {
            "revision",
            "start",
            "sample_count",
            "store_path",
        }:
            raise ValueError("Materialization catalog entry fields are invalid.")
        entry = SnapshotEntry(
            revision=_integer(raw["revision"], "revision", minimum=0),
            start=_integer(raw["start"], "start", minimum=0),
            sample_count=_integer(
                raw["sample_count"], "sample_count", minimum=1
            ),
            store_path=_relative_store_path(raw["store_path"]),
        )
        if entry.revision != revision or entry.start != start:
            raise ValueError(
                "Materialization catalog entries must be a dense append-only prefix."
            )
        entries.append(entry)
        start = entry.stop
    return SnapshotCatalog(dataset_id, split, provenance, tuple(entries), sealed)


def _validate_segment_store(
    path: Path,
    *,
    dataset_id: str,
    split: str | None,
    provenance: Mapping[str, str],
    sample_count: int,
    views: tuple[tuple[Role, Modality, View], ...],
) -> StoreDataset:
    dataset = read_store_dataset(path, views=views, legacy_policy="reject")
    manifest = dataset.manifest
    try:
        if manifest.dataset_id != dataset_id or manifest.split != split:
            raise ValueError("Snapshot store identity does not match its catalog.")
        if dict(manifest.provenance) != normalize_provenance(provenance):
            raise ValueError("Snapshot store provenance does not match its catalog.")
        if len(dataset) != sample_count:
            raise ValueError("Snapshot store sample_count does not match its catalog.")
        return dataset
    except BaseException:
        dataset.close()
        raise


def _entry_path(root: Path, entry: SnapshotEntry) -> Path:
    path = (root / entry.store_path).resolve()
    snapshots = (root / SNAPSHOTS_DIRNAME).resolve()
    if path.parent != snapshots:
        raise ValueError("Materialization snapshot store_path escapes snapshots/.")
    return path


def _relative_store_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Snapshot store_path must be a non-empty string.")
    path = Path(value)
    if path.is_absolute() or path.parts[:1] != (SNAPSHOTS_DIRNAME,):
        raise ValueError("Snapshot store_path must be below snapshots/.")
    if len(path.parts) != 2 or path.parts[1] in {"", ".", ".."}:
        raise ValueError("Snapshot store_path must name one snapshots/ child.")
    return value


def _initialize_catalog_root(root: Path) -> None:
    if dataset_ready_path(root).is_file():
        raise ValueError("A sealed canonical store cannot become a live catalog.")
    root.mkdir(parents=True, exist_ok=True)
    unexpected = {
        item.name
        for item in root.iterdir()
        if item.name not in {PRODUCER_LOCK_FILENAME, SNAPSHOTS_DIRNAME}
    }
    if unexpected:
        raise ValueError(f"Materialization root is not empty: {root}")


def _schema_views(schema: Schema) -> tuple[tuple[Role, Modality, View], ...]:
    return tuple(
        sorted(
            (
                (role, modality, view)
                for (role, modality), requirement in schema.items()
                for view in requirement.views
            ),
            key=lambda item: (item[0].value, item[1].value, item[2].value),
        )
    )


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Snapshot catalog {name} must be an integer.")
    if value < minimum:
        raise ValueError(f"Snapshot catalog {name} must be at least {minimum}.")
    return value


__all__ = [
    "SnapshotCatalog",
    "SnapshotDataset",
    "SnapshotEntry",
    "SnapshotPublisher",
    "open_snapshot_dataset",
    "producer_lock_path",
]
