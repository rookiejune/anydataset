from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from .._legacy import legacy
from .._io.files import StatFingerprint as _StatFingerprint
from .._io.files import atomic_write_bytes, stat_fingerprint
from ..cache import FileLock, anydataset_home
from ..dataset.abc import MapStyleABC
from ..dataset._shuffle import shuffle_index_groups
from ..types import item
from ..types.language import remap_lang
from ._pickle_state import decode_pickle_state, validate_pickle_fields
from .payload.files import StoreFilesLease, lease_store_files, payload_path, store_id
from .manifest.index import (
    SampleManifestSequence,
    StoreView,
    StoreViews,
    ViewEntryIndex as ViewEntryIndex,
    view_path as _view_path,
)
from .jsonio import read_json
from .manifest.schema import (
    LEGACY_STORE_SCHEMA_VERSION,
    DatasetManifest,
    SampleManifestEntry,
    ViewManifestEntry,
    dataset_manifest_from_dict,
    view_from_dict,
)
from .manifest.io import (
    ManifestParquetCache,
    manifest_parquet_cache,
    read_sample_manifest_index,
    sample_manifest_layout,
    samples_manifest_exists,
)
from .payload.groups import (
    PayloadGroupCache,
    ordered_payload_groups,
    payload_groups,
)
from .paths import (
    dataset_json_path,
    dataset_ready_path,
    samples_parquet_path,
)
from .payload.archive import PayloadCache, payload_value, read_payload_bytes

_SAMPLE_INDEX_VALIDATION_VERSION = 2
_SAMPLE_ID_SET_LIMIT = 1_000_000
STORE_DATASET_PICKLE_VERSION = 1

_STORE_DATASET_PICKLE_FIELDS = frozenset(
    {
        "root",
        "manifest",
        "samples",
        "views",
        "_file_lease",
        "_payloads",
        "_unsafe_pickle_payloads",
        "_payload_group_cache",
        "_manifest_cache",
        "_resource_state",
    }
)
_STORE_DATASET_LEGACY_OPTIONAL_FIELDS = frozenset(
    {"_unsafe_pickle_payloads", "_manifest_cache", "_resource_state"}
)
_STORE_DATASET_LEGACY_REQUIRED_FIELDS = (
    _STORE_DATASET_PICKLE_FIELDS - _STORE_DATASET_LEGACY_OPTIONAL_FIELDS
)

# Legacy-v0 pickles written before payload grouping moved out of this module
# refer to this private class path. Keep it resolvable during v0 migration.
_PayloadGroupCache = PayloadGroupCache


@dataclass
class _DatasetResourceState:
    closed: bool = False


@dataclass(frozen=True)
class StoreDataset(MapStyleABC):
    root: Path
    manifest: DatasetManifest
    samples: SampleManifestSequence
    views: StoreViews
    _file_lease: StoreFilesLease | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    _payloads: PayloadCache = field(
        default_factory=PayloadCache,
        compare=False,
        repr=False,
    )
    _unsafe_pickle_payloads: bool = field(default=False, compare=False, repr=False)
    _payload_group_cache: PayloadGroupCache = field(
        default_factory=PayloadGroupCache,
        compare=False,
        repr=False,
    )
    _manifest_cache: ManifestParquetCache = field(
        default_factory=ManifestParquetCache,
        compare=False,
        repr=False,
    )
    _resource_state: _DatasetResourceState = field(
        default_factory=_DatasetResourceState,
        compare=False,
        repr=False,
    )

    def __len__(self) -> int:
        self._ensure_open()
        return len(self.samples)

    def __getitem__(self, index: int) -> item.Sample:
        self._ensure_open()
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("store dataset index out of range.")
        sample = self.samples[index]
        return _sample_for_entry(self, sample.sample_index, sample)

    def cost_row(self, index: int) -> SampleManifestEntry:
        self._ensure_open()
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("store dataset index out of range.")
        return self.samples[index]

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        self._ensure_open()
        groups = payload_groups(
            self.root,
            self.samples,
            self.views,
            self._payload_group_cache,
        )
        indexes: Iterable[Sequence[int]] = (
            (group.indexes() for bucket in groups for group in bucket)
            if shuffle
            else ordered_payload_groups(groups)
        )
        yield from shuffle_index_groups(
            indexes,
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            num_replicas=num_replicas,
            rank=rank,
        )

    @property
    def closed(self) -> bool:
        return self._resource_state.closed

    def close(self) -> None:
        if self._resource_state.closed:
            return
        self._resource_state.closed = True
        errors: list[Exception] = []
        for resource in (
            self._payloads,
            self._manifest_cache,
            self._file_lease,
        ):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> StoreDataset:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "pickle_schema_version": STORE_DATASET_PICKLE_VERSION,
            "root": self.root,
            "manifest": self.manifest,
            "samples": self.samples,
            "views": self.views,
            "_file_lease": self._file_lease,
            "_payloads": self._payloads,
            "_unsafe_pickle_payloads": self._unsafe_pickle_payloads,
            "_payload_group_cache": self._payload_group_cache,
            "_manifest_cache": self._manifest_cache,
            "_resource_state": self._resource_state,
        }

    def __setstate__(self, state: object) -> None:
        legacy, values = decode_pickle_state(
            state,
            kind="StoreDataset",
            current_version=STORE_DATASET_PICKLE_VERSION,
        )
        if legacy:
            values = _migrate_store_dataset_pickle_v0(values)
        else:
            validate_pickle_fields(
                values,
                kind="StoreDataset",
                required=_STORE_DATASET_PICKLE_FIELDS,
            )
        _validate_store_dataset_pickle_state(values)
        attributes = cast(dict[str, Any], self.__dict__)
        attributes.update(values)
        self.views.bind_manifest_cache(self._manifest_cache)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        if self._resource_state.closed:
            raise RuntimeError("StoreDataset is closed.")


def _migrate_store_dataset_pickle_v0(
    state: dict[str, Any],
) -> dict[str, Any]:
    validate_pickle_fields(
        state,
        kind="StoreDataset",
        required=_STORE_DATASET_LEGACY_REQUIRED_FIELDS,
        optional=_STORE_DATASET_LEGACY_OPTIONAL_FIELDS,
    )
    values = dict(state)
    values.setdefault("_unsafe_pickle_payloads", False)
    values.setdefault("_manifest_cache", ManifestParquetCache())
    values.setdefault("_resource_state", _DatasetResourceState())
    return values


def _validate_store_dataset_pickle_state(state: Mapping[str, Any]) -> None:
    expected = {
        "root": Path,
        "manifest": DatasetManifest,
        "samples": SampleManifestSequence,
        "views": StoreViews,
        "_payloads": PayloadCache,
        "_payload_group_cache": PayloadGroupCache,
        "_manifest_cache": ManifestParquetCache,
        "_resource_state": _DatasetResourceState,
    }
    for name, field_type in expected.items():
        if not isinstance(state[name], field_type):
            raise TypeError(
                f"StoreDataset pickle field {name!r} must be a "
                f"{field_type.__name__}."
            )
    file_lease = state["_file_lease"]
    if file_lease is not None and not isinstance(file_lease, StoreFilesLease):
        raise TypeError(
            "StoreDataset pickle field '_file_lease' must be a "
            "StoreFilesLease or None."
        )
    if type(state["_unsafe_pickle_payloads"]) is not bool:
        raise TypeError(
            "StoreDataset pickle field '_unsafe_pickle_payloads' must be a boolean."
        )
    if type(state["_resource_state"].closed) is not bool:
        raise TypeError(
            "StoreDataset pickle resource closed state must be a boolean."
        )
    root = state["root"]
    samples = state["samples"]
    views = state["views"]
    manifest = state["manifest"]
    file_lease = state["_file_lease"]
    if samples.root != root or views.root != root:
        raise ValueError("StoreDataset pickle roots must match.")
    if views.samples is not samples:
        raise ValueError("StoreDataset pickle views must reference its samples.")
    if len(samples) != manifest.sample_count:
        raise ValueError(
            "StoreDataset pickle sample count must match its manifest."
        )
    if file_lease is not None and file_lease.root != root:
        raise ValueError("StoreDataset pickle file lease root must match.")


def read_store_dataset(
    root: str | Path,
    views: Iterable[tuple[item.Role, item.Modality, item.View]] | None = None,
    *,
    preload: bool = False,
    legacy_policy: str = "reject",
    unsafe_pickle_payloads: bool = False,
) -> StoreDataset:
    if type(unsafe_pickle_payloads) is not bool:
        raise TypeError("unsafe_pickle_payloads must be a boolean.")
    root = Path(root).expanduser().resolve()
    _validate_dataset_root(root)
    manifest = read_store_manifest(root, legacy_policy=legacy_policy)
    manifest_cache = ManifestParquetCache()
    try:
        samples_path = samples_parquet_path(root)
        fingerprint = stat_fingerprint(samples_path.stat())
        actual_sample_count, row_groups = sample_manifest_layout(
            root,
            cache=manifest_cache,
        )
        if actual_sample_count != manifest.sample_count:
            raise ValueError(
                "sample manifest row count must match dataset.json sample_count."
            )
        _validate_sample_manifest_index(
            root,
            manifest.sample_count,
            fingerprint,
            cache=manifest_cache,
        )
        if stat_fingerprint(samples_path.stat()) != fingerprint:
            raise ValueError("Sample manifest changed while opening store.")
        samples = SampleManifestSequence(
            root,
            count=manifest.sample_count,
            row_groups=row_groups,
            manifest_cache=manifest_cache,
        )

        selected_views = _select_views(_discover_views(root), views)
        store_views = StoreViews(
            root,
            samples,
            selected_views,
            manifest_cache=manifest_cache,
        )
        if preload:
            store_views.preload()
        file_lease = (
            lease_store_files(root)
            if any(
                modality is item.Modality.AUDIO and key == item.AudioView.FILE
                for _role, modality, key in selected_views
            )
            else None
        )
        return StoreDataset(
            root=root,
            manifest=manifest,
            samples=samples,
            views=store_views,
            _file_lease=file_lease,
            _manifest_cache=manifest_cache,
            _unsafe_pickle_payloads=unsafe_pickle_payloads,
        )
    except Exception:
        manifest_cache.close()
        raise


def read_store_manifest(
    root: str | Path,
    *,
    legacy_policy: str = "reject",
) -> DatasetManifest:
    root = Path(root).expanduser()
    _validate_dataset_root(root)
    manifest = dataset_manifest_from_dict(read_json(dataset_json_path(root)))
    if manifest.schema_version == LEGACY_STORE_SCHEMA_VERSION:
        legacy(
            "store schema_version 2",
            legacy_policy=legacy_policy,
            warning=(
                "Reading store schema_version 2 without provenance. "
                "Compatibility is for legacy reads only; rematerialize "
                "to schema_version 3 before publishing the store or using it as "
                "the basis for cache-sensitive derived data."
            ),
            error=(
                "Store schema_version 2 is legacy and lacks provenance. "
                "Rematerialize the store to schema_version 3, or pass "
                "legacy_policy='allow' for an intentional legacy read."
            ),
            stacklevel=2,
        )
    return manifest


def read_store_views(
    root: str | Path,
) -> tuple[tuple[item.Role, item.Modality, item.View], ...]:
    root = Path(root).expanduser()
    _validate_dataset_root(root)
    return _discover_views(root)


def _validate_dataset_root(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not dataset_ready_path(root).is_file():
        raise ValueError(f"Store dataset is not ready: {root}")
    if not dataset_json_path(root).is_file():
        raise FileNotFoundError(dataset_json_path(root))
    if not samples_manifest_exists(root):
        raise FileNotFoundError(root / "samples.parquet")


def _select_views(
    available: tuple[tuple[item.Role, item.Modality, item.View], ...],
    requested: Iterable[tuple[item.Role, item.Modality, item.View]] | None,
) -> tuple[tuple[item.Role, item.Modality, item.View], ...]:
    if requested is None:
        return available
    selected = tuple(requested)
    available_set = frozenset(available)
    seen: set[tuple[item.Role, item.Modality, item.View]] = set()
    for view in selected:
        _validate_view_ref(view)
        if view in seen:
            raise ValueError(f"Duplicate requested store view: {_view_path(view)}.")
        if view not in available_set:
            raise KeyError(f"Store dataset does not contain view {_view_path(view)}.")
        seen.add(view)
    return selected


def _discover_views(
    root: Path,
) -> tuple[tuple[item.Role, item.Modality, item.View], ...]:
    views = []
    for path in _view_dirs(root):
        view = _view_from_dir(root, path)
        _validate_view_dir(path, view)
        views.append(view)
    return tuple(sorted(views, key=_view_path))


def _view_from_dir(
    root: Path,
    path: Path,
) -> tuple[item.Role, item.Modality, item.View]:
    parts = path.relative_to(root).parts
    if len(parts) != 3:
        raise ValueError(f"Store dataset view path must have three parts: {path}")
    try:
        role, modality, key = parts
        return view_from_dict({"role": role, "modality": modality, "view": key})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid store dataset view path: {path}") from exc


def _view_dirs(root: Path) -> Iterator[Path]:
    for path in root.glob("*/*/*"):
        if not path.is_dir():
            continue
        if _runtime_path(root, path):
            continue
        if _has_view_marker(path):
            yield path


def _runtime_path(root: Path, path: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _has_view_marker(path: Path) -> bool:
    return any(
        (path / name).exists() for name in ("manifest.parquet", ".ready", "shards")
    )


def _validate_view_dir(
    path: Path,
    view: tuple[item.Role, item.Modality, item.View],
) -> None:
    if not (path / ".ready").is_file():
        raise ValueError(f"Store dataset view is not ready: {_view_path(view)}.")
    if not (path / "manifest.parquet").is_file():
        raise FileNotFoundError(path / "manifest.parquet")


def _validate_sample_manifest_index(
    root: Path,
    sample_count: int,
    fingerprint: _StatFingerprint,
    *,
    cache: ManifestParquetCache | None = None,
) -> None:
    marker = _sample_index_validation_path(root, sample_count, fingerprint)
    with FileLock(
        marker.with_name(f".{marker.name}.lock"),
        wait_timeout=3600.0,
    ):
        _validate_sample_manifest_index_locked(
            root,
            sample_count,
            fingerprint,
            cache=cache,
            marker=marker,
        )


def _validate_sample_manifest_index_locked(
    root: Path,
    sample_count: int,
    fingerprint: _StatFingerprint,
    *,
    cache: ManifestParquetCache | None,
    marker: Path,
) -> None:
    path = samples_parquet_path(root)
    if marker.is_file():
        return

    sample_ids: set[str] | None = (
        set() if sample_count <= _SAMPLE_ID_SET_LIMIT else None
    )
    temporary: tempfile.TemporaryDirectory[str] | None = None
    connection: sqlite3.Connection | None = None
    if sample_ids is None:
        temporary = tempfile.TemporaryDirectory(prefix="anydataset-sample-index-")
        connection = sqlite3.connect(Path(temporary.name) / "ids.sqlite")
        connection.execute(
            "CREATE TABLE sample_ids (sample_id TEXT PRIMARY KEY)"
        )
    count = 0
    try:
        active_cache = manifest_parquet_cache() if cache is None else cache
        with active_cache.activate():
            for index, (sample_index, sample_id) in enumerate(
                read_sample_manifest_index(root)
            ):
                if sample_index != index:
                    raise ValueError(
                        f"Sample manifest row {index} has sample_index {sample_index}."
                    )
                if sample_ids is not None:
                    if sample_id in sample_ids:
                        raise ValueError(f"Duplicate sample_id {sample_id!r}.")
                    sample_ids.add(sample_id)
                else:
                    if connection is None:
                        raise RuntimeError("sample id validation database is missing.")
                    try:
                        connection.execute(
                            "INSERT INTO sample_ids(sample_id) VALUES (?)",
                            (sample_id,),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"Duplicate sample_id {sample_id!r}.") from exc
                count += 1
                if connection is not None and count % 8192 == 0:
                    connection.commit()
    finally:
        if connection is not None:
            connection.commit()
            connection.close()
        if temporary is not None:
            temporary.cleanup()
    if count != sample_count:
        raise ValueError(
            "sample manifest row count must match dataset.json sample_count."
        )
    if stat_fingerprint(path.stat()) != fingerprint:
        raise ValueError("Sample manifest changed while validating index.")
    atomic_write_bytes(marker, b"valid\n")


def _sample_index_validation_path(
    root: Path,
    sample_count: int,
    fingerprint: _StatFingerprint,
) -> Path:
    validation_id = hashlib.sha256(
        "\0".join(
            (
                str(_SAMPLE_INDEX_VALIDATION_VERSION),
                str(sample_count),
                *(str(value) for value in fingerprint),
            )
        ).encode("utf-8")
    ).hexdigest()
    return (
        anydataset_home()
        / "cache"
        / "store-validation"
        / store_id(root)
        / f"{validation_id}.ready"
    )


def _validate_view_ref(view: tuple[item.Role, item.Modality, item.View]) -> None:
    if not isinstance(view, tuple) or len(view) != 3:
        raise TypeError("store views must be (Role, Modality, View) tuples.")
    role, modality, key = view
    if not isinstance(role, item.Role):
        raise TypeError("store view role must be a Role.")
    if not isinstance(modality, item.Modality):
        raise TypeError("store view modality must be a Modality.")
    if not isinstance(key, (item.AudioView, item.ImageView, item.TextView)):
        raise TypeError("store view key must be a View.")


def _sample_for_entry(
    dataset: StoreDataset,
    index: int,
    sample: SampleManifestEntry,
) -> item.Sample:
    result: dict[tuple[item.Role, item.Modality], item.Item] = {}
    for sample_ref, item_entry in sample.items:
        views: dict[Any, Any] = {}
        for view_entry, view in dataset.views.for_ref(sample_ref):
            entry = view.entries_by_index[index]
            if entry is None:
                raise ValueError(
                    f"View {_view_path(view_entry)} is missing sample_index {index}."
                )
            views[view_entry[2]] = _view_value(dataset, view, entry)
        result[sample_ref] = _item_from_entry(sample_ref, item_entry, views)
    return result


def _item_from_entry(
    sample_ref: tuple[item.Role, item.Modality],
    meta: Mapping[str, Any] | None,
    views: Mapping[Any, Any],
) -> item.Item:
    _, modality = sample_ref
    meta = {} if meta is None else dict(meta)
    converted = _enum_keys(meta, modality.meta_type())
    if modality is item.Modality.TEXT:
        lang = converted.get(item.TextMeta.LANG)
        if lang is not None:
            converted[item.TextMeta.LANG] = remap_lang(lang)
    return modality.item(views=views, meta=converted)


def _view_value(
    dataset: StoreDataset,
    view: StoreView,
    entry: ViewManifestEntry,
) -> Any:
    if view.view[1] is item.Modality.AUDIO and view.view[2] == item.AudioView.FILE:
        return str(_cached_file_payload(dataset, entry, view))

    data = read_payload_bytes(dataset.root, view.view, entry, cache=dataset._payloads)
    return payload_value(
        view.view,
        data,
        unsafe_pickle=dataset._unsafe_pickle_payloads,
    )


def _cached_file_payload(
    dataset: StoreDataset,
    entry: ViewManifestEntry,
    view: StoreView,
) -> Path:
    lease = dataset._file_lease
    if lease is None or not lease.active:
        raise RuntimeError("Store FILE view reader does not hold a cache lease.")
    target = payload_path(
        dataset.root,
        view.view,
        entry,
        cache_path=lease.cache_path,
    )
    if not target.is_file():
        data = read_payload_bytes(
            dataset.root, view.view, entry, cache=dataset._payloads
        )
        if (
            payload_path(
                dataset.root,
                view.view,
                entry,
                cache_path=lease.cache_path,
            )
            != target
        ):
            raise ValueError("View shard changed while caching file payload.")
        # Extracted FILE payloads are disposable cache entries. Atomic rename
        # prevents torn reads; syncing every sample only adds storage latency.
        atomic_write_bytes(target, data, durable=False)
    return target


def _enum_keys(values: Mapping[str, Any], enum_type):
    converted = {}
    for key, value in values.items():
        converted[enum_type(key)] = value
    return converted
