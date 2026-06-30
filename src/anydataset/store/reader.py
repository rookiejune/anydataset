from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
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
from .manifestio import read_samples_manifest, read_view_manifest, samples_manifest_exists
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
    samples: tuple[SampleManifestEntry, ...]
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
        return _sample_for_entry(self, index, self.samples[index])


@dataclass(frozen=True)
class StoreView:
    view: tuple[item.Role, item.Modality, item.View]
    entries_by_index: tuple[ViewManifestEntry | None, ...]


class StoreViews(Mapping[tuple[item.Role, item.Modality, item.View], StoreView]):
    def __init__(
        self,
        root: Path,
        samples: tuple[SampleManifestEntry, ...],
        views: Iterable[tuple[item.Role, item.Modality, item.View]],
    ) -> None:
        self.root = root
        self.samples = samples
        self._views = tuple(views)
        self._view_set = frozenset(self._views)
        self._views_by_ref: dict[
            tuple[item.Role, item.Modality],
            tuple[tuple[item.Role, item.Modality, item.View], ...],
        ] = {}
        for view in self._views:
            self._views_by_ref.setdefault(view[:2], ())
            self._views_by_ref[view[:2]] = (*self._views_by_ref[view[:2]], view)
        self._cache: dict[tuple[item.Role, item.Modality, item.View], StoreView] = {}
        self._expected: dict[tuple[item.Role, item.Modality], frozenset[int]] = {}

    def __getitem__(
        self,
        view: tuple[item.Role, item.Modality, item.View],
    ) -> StoreView:
        if view not in self._view_set:
            raise KeyError(view)
        cached = self._cache.get(view)
        if cached is None:
            cached = _load_view(
                self.root,
                view,
                len(self.samples),
                self._expected_indexes(view[:2]),
            )
            self._cache[view] = cached
        return cached

    def __iter__(self) -> Iterator[tuple[item.Role, item.Modality, item.View]]:
        yield from self._views

    def __len__(self) -> int:
        return len(self._views)

    def preload(self) -> None:
        for view in self._views:
            self[view]

    def for_ref(
        self,
        ref: tuple[item.Role, item.Modality],
    ) -> Iterator[tuple[tuple[item.Role, item.Modality, item.View], StoreView]]:
        for view in self._views_by_ref.get(ref, ()):
            yield view, self[view]

    def _expected_indexes(
        self,
        ref: tuple[item.Role, item.Modality],
    ) -> frozenset[int]:
        cached = self._expected.get(ref)
        if cached is None:
            cached = frozenset(
                sample.sample_index
                for sample in self.samples
                if _sample_has_item(sample, ref)
            )
            self._expected[ref] = cached
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
    samples = tuple(read_samples_manifest(root))
    if len(samples) != manifest.sample_count:
        raise ValueError(
            "sample manifest row count must match dataset.json sample_count."
        )
    _validate_samples(samples)

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
    if version != STORE_SCHEMA_VERSION:
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


def _load_view(
    root: Path,
    view: tuple[item.Role, item.Modality, item.View],
    sample_count: int,
    expected_ids: frozenset[int],
) -> StoreView:
    if not view_ready_path(root, view).exists():
        raise ValueError(f"Store dataset view is not ready: {_view_path(view)}.")
    entries_by_index: list[ViewManifestEntry | None] = [None] * sample_count
    actual_ids: set[int] = set()
    for entry in read_view_manifest(root, view):
        if _entry_view(entry) != view:
            raise ValueError("View manifest entry ref must match its path.")
        if entry.sample_index < 0 or entry.sample_index >= sample_count:
            raise ValueError(
                f"View {_view_path(view)} has sample_index outside dataset: "
                f"{entry.sample_index}."
            )
        if entries_by_index[entry.sample_index] is not None:
            raise ValueError(
                f"Duplicate view entry for sample_index {entry.sample_index}."
            )
        entries_by_index[entry.sample_index] = entry
        actual_ids.add(entry.sample_index)
    _validate_view_coverage(view, expected_ids, actual_ids)
    return StoreView(
        view=view,
        entries_by_index=tuple(entries_by_index),
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


def _validate_samples(samples: tuple[SampleManifestEntry, ...]) -> None:
    sample_ids: set[str] = set()
    for index, sample in enumerate(samples):
        if sample.sample_index != index:
            raise ValueError(
                f"Sample manifest row {index} has sample_index {sample.sample_index}."
            )
        if sample.sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id {sample.sample_id!r}.")
        sample_ids.add(sample.sample_id)
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


def _validate_view_coverage(
    view: tuple[item.Role, item.Modality, item.View],
    expected_ids: frozenset[int],
    actual_ids: set[int],
) -> None:
    if actual_ids == expected_ids:
        return
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    detail = _coverage_detail(missing, extra)
    raise ValueError(f"View {_view_path(view)} sample coverage mismatch: {detail}.")


def _sample_has_item(
    sample: SampleManifestEntry,
    ref: tuple[item.Role, item.Modality],
) -> bool:
    return any(item_ref == ref for item_ref, _meta in sample.items)


def _coverage_detail(missing: list[int], extra: list[int]) -> str:
    details = []
    if missing:
        details.append(f"missing sample_index {missing[0]}")
    if extra:
        details.append(f"unexpected sample_index {extra[0]}")
    return ", ".join(details)


def _sample_for_entry(
    dataset: StoreDataset,
    index: int,
    sample: SampleManifestEntry,
) -> item.Sample:
    views_by_ref: dict[tuple[item.Role, item.Modality], dict[Any, Any]] = {}
    item_entries = dict(sample.items)
    for sample_ref in item_entries:
        for view_entry, view in dataset.views.for_ref(sample_ref):
            entry = view.entries_by_index[index]
            if entry is None:
                raise ValueError(
                    f"View {_view_path(view_entry)} is missing sample_index {index}."
                )
            views = views_by_ref.setdefault(sample_ref, {})
            views[view_entry[2]] = _view_value(dataset, view, entry)

    result: dict[tuple[item.Role, item.Modality], item.Item] = {}
    for sample_ref, views in views_by_ref.items():
        item_entry = item_entries.get(sample_ref)
        result[sample_ref] = _item_from_entry(sample_ref, item_entry, views)

    for sample_ref, item_entry in item_entries.items():
        if sample_ref not in result:
            result[sample_ref] = _item_from_entry(sample_ref, item_entry, {})
    return result


def _item_from_entry(
    sample_ref: tuple[item.Role, item.Modality],
    meta: Mapping[str, Any] | None,
    views: Mapping[Any, Any],
) -> item.Item:
    _, modality = sample_ref
    meta = {} if meta is None else dict(meta)
    match modality:
        case item.Modality.AUDIO:
            return item.AudioItem(
                views=views,
                meta=_enum_keys(meta, item.AudioMeta),
            )
        case item.Modality.IMAGE:
            return item.ImageItem(
                views=views,
                meta=_enum_keys(meta, item.ImageMeta),
            )
        case item.Modality.TEXT:
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
