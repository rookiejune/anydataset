from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..dataset.abc import MapStyleABC
from ..types import item
from .jsonio import read_json
from .manifest import (
    DatasetManifest,
    SampleManifestEntry,
    STORE_SCHEMA_VERSION,
    ViewManifestEntry,
    view_from_dict,
)
from .manifestio import (
    read_sample_manifest_index,
    read_samples_manifest_row_group,
    read_view_manifest_indexes,
    read_view_manifest_row_group,
    sample_manifest_row_count,
    sample_manifest_row_groups,
    samples_manifest_exists,
    view_manifest_row_count,
    view_manifest_row_groups,
)
from .paths import (
    dataset_json_path,
    dataset_ready_path,
    view_ready_path,
)
from .payload import PayloadCache, payload_value, read_payload_bytes


@dataclass(frozen=True)
class StoreDataset(MapStyleABC):
    root: Path
    manifest: DatasetManifest
    samples: SampleManifestSequence
    views: StoreViews
    _files: dict[str, Path] = field(default_factory=dict, compare=False, repr=False)
    _payloads: PayloadCache = field(
        default_factory=PayloadCache,
        compare=False,
        repr=False,
    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> item.Sample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("store dataset index out of range.")
        sample = self.samples[index]
        return _sample_for_entry(self, sample.sample_index, sample)


class SampleManifestSequence(Sequence[SampleManifestEntry]):
    def __init__(
        self,
        root: Path,
        *,
        count: int,
        row_groups: Sequence[int],
        max_cached_groups: int = 2,
    ) -> None:
        self.root = root
        self._count = count
        self._row_groups = tuple(row_groups)
        self._offsets = _offsets(self._row_groups)
        self._max_cached_groups = max_cached_groups
        self._cache: OrderedDict[int, tuple[SampleManifestEntry, ...]] = OrderedDict()

    def __len__(self) -> int:
        return self._count

    def __getitem__(
        self,
        index: int | slice,
    ) -> SampleManifestEntry | tuple[SampleManifestEntry, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return tuple(self[position] for position in range(start, stop, step))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("sample manifest index out of range.")
        row_group = bisect_right(self._offsets, index) - 1
        rows = self._row_group(row_group)
        return rows[index - self._offsets[row_group]]

    def __iter__(self) -> Iterator[SampleManifestEntry]:
        for row_group in range(len(self._row_groups)):
            yield from self._row_group(row_group)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "count": self._count,
            "row_groups": self._row_groups,
            "max_cached_groups": self._max_cached_groups,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(
            state["root"],
            count=state["count"],
            row_groups=state["row_groups"],
            max_cached_groups=state["max_cached_groups"],
        )

    def _row_group(self, row_group: int) -> tuple[SampleManifestEntry, ...]:
        cached = self._cache.get(row_group)
        if cached is not None:
            self._cache.move_to_end(row_group)
            return cached
        rows = read_samples_manifest_row_group(self.root, row_group)
        start = self._offsets[row_group]
        for offset, sample in enumerate(rows):
            _validate_sample_entry(sample, start + offset)
        self._cache[row_group] = rows
        while len(self._cache) > self._max_cached_groups:
            self._cache.popitem(last=False)
        return rows


@dataclass(frozen=True)
class StoreView:
    view: tuple[item.Role, item.Modality, item.View]
    entries_by_index: ViewEntryIndex


class ViewEntryIndex:
    def __init__(
        self,
        root: Path,
        view: tuple[item.Role, item.Modality, item.View],
        *,
        sample_count: int,
        row_groups: Sequence[int],
        sample_indexes: array[int],
        max_cached_groups: int = 2,
    ) -> None:
        self.root = root
        self.view = view
        self._sample_count = sample_count
        self._row_groups = tuple(row_groups)
        self._offsets = _offsets(self._row_groups)
        self._sample_indexes = sample_indexes
        self._max_cached_groups = max_cached_groups
        self._cache: OrderedDict[int, tuple[ViewManifestEntry, ...]] = OrderedDict()

    @classmethod
    def load(
        cls,
        root: Path,
        view: tuple[item.Role, item.Modality, item.View],
        *,
        sample_count: int,
    ) -> ViewEntryIndex:
        row_count = view_manifest_row_count(root, view)
        row_groups = view_manifest_row_groups(root, view)
        sample_indexes = array("q", read_view_manifest_indexes(root, view))
        if len(sample_indexes) != row_count:
            raise ValueError("View manifest row count changed while loading index.")
        _validate_view_indexes(view, sample_indexes, sample_count)
        return cls(
            root,
            view,
            sample_count=sample_count,
            row_groups=row_groups,
            sample_indexes=sample_indexes,
        )

    def __len__(self) -> int:
        return self._sample_count

    def __getitem__(self, sample_index: int) -> ViewManifestEntry | None:
        if sample_index < 0:
            sample_index += self._sample_count
        if sample_index < 0 or sample_index >= self._sample_count:
            raise IndexError("view entry index out of range.")
        position = bisect_left(self._sample_indexes, sample_index)
        if position >= len(self._sample_indexes):
            return None
        if self._sample_indexes[position] != sample_index:
            return None
        row_group = bisect_right(self._offsets, position) - 1
        rows = self._row_group(row_group)
        entry = rows[position - self._offsets[row_group]]
        if entry.sample_index != sample_index:
            raise ValueError(
                f"View {_view_path(self.view)} index changed while reading."
            )
        return entry

    def validate_coverage(
        self,
        expected_indexes: Iterable[int],
    ) -> None:
        actual_position = 0
        actual_count = len(self._sample_indexes)
        for expected in expected_indexes:
            if actual_position >= actual_count:
                _raise_view_coverage_error(self.view, missing=expected, extra=None)
            actual = int(self._sample_indexes[actual_position])
            if actual < expected:
                _raise_view_coverage_error(self.view, missing=None, extra=actual)
            if actual > expected:
                _raise_view_coverage_error(self.view, missing=expected, extra=None)
            actual_position += 1
        if actual_position < actual_count:
            _raise_view_coverage_error(
                self.view,
                missing=None,
                extra=int(self._sample_indexes[actual_position]),
            )

    def validate_entries(self) -> None:
        for row_group in range(len(self._row_groups)):
            self._row_group(row_group)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "view": self.view,
            "sample_count": self._sample_count,
            "row_groups": self._row_groups,
            "sample_indexes": self._sample_indexes,
            "max_cached_groups": self._max_cached_groups,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(
            state["root"],
            state["view"],
            sample_count=state["sample_count"],
            row_groups=state["row_groups"],
            sample_indexes=state["sample_indexes"],
            max_cached_groups=state["max_cached_groups"],
        )

    def _row_group(self, row_group: int) -> tuple[ViewManifestEntry, ...]:
        cached = self._cache.get(row_group)
        if cached is not None:
            self._cache.move_to_end(row_group)
            return cached
        rows = read_view_manifest_row_group(self.root, self.view, row_group)
        start = self._offsets[row_group]
        for offset, entry in enumerate(rows):
            if _entry_view(entry) != self.view:
                raise ValueError("View manifest entry ref must match its path.")
            if entry.sample_index != self._sample_indexes[start + offset]:
                raise ValueError(
                    f"View {_view_path(self.view)} index changed while reading."
                )
        self._cache[row_group] = rows
        while len(self._cache) > self._max_cached_groups:
            self._cache.popitem(last=False)
        return rows


class StoreViews(Mapping[tuple[item.Role, item.Modality, item.View], StoreView]):
    def __init__(
        self,
        root: Path,
        samples: SampleManifestSequence,
        views: Iterable[tuple[item.Role, item.Modality, item.View]],
    ) -> None:
        self.root = root
        self.samples = samples
        self._views = tuple(views)
        self._view_set = frozenset(self._views)
        views_by_ref: dict[
            tuple[item.Role, item.Modality],
            list[tuple[item.Role, item.Modality, item.View]],
        ] = {}
        for view in self._views:
            views_by_ref.setdefault(view[:2], []).append(view)
        self._views_by_ref = {
            ref: tuple(ref_views) for ref, ref_views in views_by_ref.items()
        }
        self._cache: dict[tuple[item.Role, item.Modality, item.View], StoreView] = {}
        self._validated: set[tuple[item.Role, item.Modality, item.View]] = set()

    def __getitem__(
        self,
        view: tuple[item.Role, item.Modality, item.View],
    ) -> StoreView:
        if view not in self._view_set:
            raise KeyError(view)
        return self._view(view, validate_coverage=False)

    def __iter__(self) -> Iterator[tuple[item.Role, item.Modality, item.View]]:
        yield from self._views

    def __len__(self) -> int:
        return len(self._views)

    def preload(self) -> None:
        for view in self._views:
            self._view(view, validate_coverage=True)

    def for_ref(
        self,
        ref: tuple[item.Role, item.Modality],
    ) -> Iterator[tuple[tuple[item.Role, item.Modality, item.View], StoreView]]:
        for view in self._views_by_ref.get(ref, ()):
            yield view, self[view]

    def _view(
        self,
        view: tuple[item.Role, item.Modality, item.View],
        *,
        validate_coverage: bool,
    ) -> StoreView:
        cached = self._cache.get(view)
        if cached is None:
            cached = _load_view(
                self.root,
                view,
                len(self.samples),
            )
            self._cache[view] = cached
        if validate_coverage and view not in self._validated:
            cached.entries_by_index.validate_entries()
            cached.entries_by_index.validate_coverage(
                _sample_indexes_for_ref(self.samples, view[:2])
            )
            self._validated.add(view)
        return cached


def read_store_dataset(
    root: str | Path,
    views: Iterable[tuple[item.Role, item.Modality, item.View]] | None = None,
    *,
    preload: bool = False,
) -> StoreDataset:
    root = Path(root).expanduser()
    _validate_dataset_root(root)
    manifest = read_store_manifest(root)
    actual_sample_count = sample_manifest_row_count(root)
    if actual_sample_count != manifest.sample_count:
        raise ValueError(
            "sample manifest row count must match dataset.json sample_count."
        )
    _validate_sample_manifest_index(root, manifest.sample_count)
    samples = SampleManifestSequence(
        root,
        count=manifest.sample_count,
        row_groups=sample_manifest_row_groups(root),
    )

    selected_views = _select_views(_discover_views(root), views)
    store_views = StoreViews(root, samples, selected_views)
    if preload:
        store_views.preload()
    return StoreDataset(
        root=root,
        manifest=manifest,
        samples=samples,
        views=store_views,
    )


def read_store_manifest(root: str | Path) -> DatasetManifest:
    root = Path(root).expanduser()
    _validate_dataset_root(root)
    data = read_json(dataset_json_path(root))
    version = data.get("schema_version")
    if version is None:
        data = {**data, "schema_version": 1}
    elif version != STORE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported store schema_version: "
            f"{version!r}; expected {STORE_SCHEMA_VERSION}."
        )
    return DatasetManifest(**data)


def read_store_views(root: str | Path) -> tuple[tuple[item.Role, item.Modality, item.View], ...]:
    root = Path(root).expanduser()
    _validate_dataset_root(root)
    return _discover_views(root)


def _validate_dataset_root(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not dataset_ready_path(root).exists():
        raise ValueError(f"Store dataset is not ready: {root}")
    if not dataset_json_path(root).is_file():
        raise FileNotFoundError(dataset_json_path(root))
    if not samples_manifest_exists(root):
        raise FileNotFoundError(root / "samples.parquet")


def _offsets(counts: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return tuple(offsets)


def _load_view(
    root: Path,
    view: tuple[item.Role, item.Modality, item.View],
    sample_count: int,
) -> StoreView:
    if not view_ready_path(root, view).exists():
        raise ValueError(f"Store dataset view is not ready: {_view_path(view)}.")
    return StoreView(
        view=view,
        entries_by_index=ViewEntryIndex.load(
            root,
            view,
            sample_count=sample_count,
        ),
    )


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


def _discover_views(root: Path) -> tuple[tuple[item.Role, item.Modality, item.View], ...]:
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
        (path / name).exists()
        for name in ("manifest.parquet", ".ready", "shards")
    )


def _validate_view_dir(
    path: Path,
    view: tuple[item.Role, item.Modality, item.View],
) -> None:
    if not (path / ".ready").is_file():
        raise ValueError(f"Store dataset view is not ready: {_view_path(view)}.")
    if not (path / "manifest.parquet").is_file():
        raise FileNotFoundError(path / "manifest.parquet")


def _validate_sample_manifest_index(root: Path, sample_count: int) -> None:
    sample_ids: set[str] = set()
    count = 0
    for index, (sample_index, sample_id) in enumerate(read_sample_manifest_index(root)):
        if sample_index != index:
            raise ValueError(
                f"Sample manifest row {index} has sample_index {sample_index}."
            )
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id {sample_id!r}.")
        sample_ids.add(sample_id)
        count += 1
    if count != sample_count:
        raise ValueError("sample manifest row count must match dataset.json sample_count.")


def _validate_sample_entry(sample: SampleManifestEntry, index: int) -> None:
    if sample.sample_index != index:
        raise ValueError(
            f"Sample manifest row {index} has sample_index {sample.sample_index}."
        )
    refs: set[tuple[item.Role, item.Modality]] = set()
    for ref, _ in sample.items:
        if ref in refs:
            raise ValueError(f"Duplicate sample item ref {ref!r}.")
        refs.add(ref)


def _validate_view_ref(view: tuple[item.Role, item.Modality, item.View]) -> None:
    if not isinstance(view, tuple) or len(view) != 3:
        raise TypeError("store views must be (Role, Modality, View) tuples.")
    role, modality, key = view
    if not isinstance(role, item.Role):
        raise TypeError("store view role must be a Role.")
    if not isinstance(modality, item.Modality):
        raise TypeError("store view modality must be a Modality.")
    if not isinstance(key, item.AudioView | item.ImageView | item.TextView):
        raise TypeError("store view key must be a View.")


def _validate_view_indexes(
    view: tuple[item.Role, item.Modality, item.View],
    indexes: Sequence[int],
    sample_count: int,
) -> None:
    previous: int | None = None
    for index in indexes:
        if index < 0 or index >= sample_count:
            raise ValueError(
                f"View {_view_path(view)} has sample_index outside dataset: {index}."
            )
        if previous is not None:
            if index == previous:
                raise ValueError(f"Duplicate view entry for sample_index {index}.")
            if index < previous:
                raise ValueError("View manifest entries must be ordered by sample_index.")
        previous = index


def _sample_indexes_for_ref(
    samples: Iterable[SampleManifestEntry],
    ref: tuple[item.Role, item.Modality],
) -> Iterator[int]:
    for sample in samples:
        if any(item_ref == ref for item_ref, _meta in sample.items):
            yield sample.sample_index


def _raise_view_coverage_error(
    view: tuple[item.Role, item.Modality, item.View],
    *,
    missing: int | None,
    extra: int | None,
) -> None:
    details = []
    if missing is not None:
        details.append(f"missing sample_index {missing}")
    if extra is not None:
        details.append(f"unexpected sample_index {extra}")
    detail = ", ".join(details)
    raise ValueError(f"View {_view_path(view)} sample coverage mismatch: {detail}.")


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
    if modality is item.Modality.AUDIO:
        return item.AudioItem(
            views=views,
            meta=_enum_keys(meta, item.AudioMeta),
        )
    if modality is item.Modality.IMAGE:
        return item.ImageItem(
            views=views,
            meta=_enum_keys(meta, item.ImageMeta),
        )
    if modality is item.Modality.TEXT:
        return item.TextItem(
            views=views,
            meta=_enum_keys(meta, item.TextMeta),
        )
    raise ValueError(f"Unsupported modality: {modality!r}.")


def _view_value(
    dataset: StoreDataset,
    view: StoreView,
    entry: ViewManifestEntry,
) -> Any:
    if view.view[1] is item.Modality.AUDIO and view.view[2] == item.AudioView.FILE:
        return str(_cached_file_payload(dataset, entry, view))

    data = read_payload_bytes(dataset.root, view.view, entry, cache=dataset._payloads)
    return payload_value(view.view, data)


def _cached_file_payload(
    dataset: StoreDataset,
    entry: ViewManifestEntry,
    view: StoreView,
) -> Path:
    cached = dataset._files.get(entry.key)
    if cached is not None:
        return cached

    data = read_payload_bytes(dataset.root, view.view, entry, cache=dataset._payloads)
    return _cache_file_payload(dataset, entry, data)


def _cache_file_payload(
    dataset: StoreDataset,
    entry: ViewManifestEntry,
    data: bytes,
) -> Path:
    target = _file_cache_path(dataset.root, entry)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    dataset._files[entry.key] = target
    return target


def _file_cache_path(root: Path, entry: ViewManifestEntry) -> Path:
    return root / ".cache" / "files" / entry.key


def _enum_keys(values: Mapping[str, Any], enum_type):
    converted = {}
    for key, value in values.items():
        converted[enum_type(key)] = value
    return converted


def _entry_view(entry: ViewManifestEntry) -> tuple[item.Role, item.Modality, item.View]:
    return entry.role, entry.modality, entry.view


def _view_path(view: tuple[item.Role, item.Modality, item.View]) -> tuple[str, str, str]:
    role, modality, key = view
    return role.value, modality.value, key.value


def _sample_ref_path(ref: tuple[item.Role, item.Modality]) -> tuple[str, str]:
    role, modality = ref
    return role.value, modality.value
