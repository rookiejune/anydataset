from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..types import item
from .jsonio import read_json
from .manifest import (
    DatasetManifest,
    SampleItemEntry,
    SampleManifestEntry,
    ViewManifestEntry,
    ViewRef,
)
from .manifestio import (
    read_samples_manifest,
    read_view_manifest,
    samples_manifest_exists,
)
from .paths import (
    dataset_json_path,
    dataset_ready_path,
    view_ready_path,
)
from .payload import matches_checksum, payload_value, read_payload_bytes


@dataclass(frozen=True)
class StoreDataset:
    root: Path
    cache_path: Path
    manifest: DatasetManifest
    samples: tuple[SampleManifestEntry, ...]
    views: Mapping[ViewRef, "StoreView"]
    require_selected_views: bool

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[item.Sample]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> item.Sample:
        return self.sample_at(index)

    def sample_at(self, index: int) -> item.Sample:
        return _sample_for_entry(self, self.samples[index])


@dataclass(frozen=True)
class StoreView:
    ref: ViewRef
    revision: str
    entries: Mapping[str, ViewManifestEntry]


def read_store_dataset(
    root: str | Path,
    *,
    split: str | None = None,
    cache_path: str | Path,
    views: Sequence[ViewRef] | None = None,
) -> StoreDataset:
    root = Path(root).expanduser()
    cache_path = Path(cache_path)
    _validate_dataset_root(root)
    manifest = DatasetManifest.from_dict(read_json(dataset_json_path(root)))
    _validate_split(split, manifest)
    samples = tuple(read_samples_manifest(root))
    if len(samples) != manifest.sample_count:
        raise ValueError(
            "sample manifest row count must match dataset.json sample_count."
        )

    requested = None if views is None else tuple(views)
    if requested is not None:
        for ref in requested:
            _validate_view_ref(ref)
    selections = _selected_views(manifest, requested)
    indexes = {
        selection.ref: _load_view(root, selection.ref, selection.revision)
        for selection in selections
    }
    return StoreDataset(
        root=root,
        cache_path=cache_path,
        manifest=manifest,
        samples=samples,
        views=indexes,
        require_selected_views=requested is not None,
    )


def _validate_dataset_root(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not dataset_ready_path(root).exists():
        raise ValueError(f"Unified dataset is not ready: {root}")
    if not dataset_json_path(root).is_file():
        raise FileNotFoundError(dataset_json_path(root))
    if not samples_manifest_exists(root):
        raise FileNotFoundError(root / "samples.jsonl")


def _validate_split(split: str | None, manifest: DatasetManifest) -> None:
    if split is not None and manifest.split is not None and split != manifest.split:
        raise ValueError(
            f"Dataset split {manifest.split!r} does not match requested split {split!r}."
        )


def _selected_views(
    manifest: DatasetManifest,
    requested: tuple[ViewRef, ...] | None,
):
    available = {selection.ref: selection for selection in manifest.views}
    if requested is not None:
        missing = [ref for ref in requested if ref not in available]
        if missing:
            raise KeyError(f"Unified dataset is missing requested views: {missing!r}.")
        return tuple(available[ref] for ref in requested)
    return manifest.views


def _load_view(root: Path, ref: ViewRef, revision: str) -> StoreView:
    _validate_view_ref(ref)
    if not view_ready_path(root, ref, revision).exists():
        raise ValueError(f"Unified dataset view is not ready: {ref.path_parts()}.")
    entries: dict[str, ViewManifestEntry] = {}
    for entry in read_view_manifest(root, ref, revision):
        if entry.ref != ref:
            raise ValueError("View manifest entry ref must match its path.")
        if entry.revision != revision:
            raise ValueError("View manifest entry revision must match its path.")
        if entry.sample_id in entries:
            raise ValueError(f"Duplicate view entry for sample_id {entry.sample_id!r}.")
        entries[entry.sample_id] = entry
    return StoreView(ref=ref, revision=revision, entries=entries)


def _sample_for_entry(
    dataset: StoreDataset,
    sample: SampleManifestEntry,
) -> item.Sample:
    views_by_ref: dict[tuple[item.Role, item.Modality], dict[Any, Any]] = {}
    for ref, view in dataset.views.items():
        entry = view.entries.get(sample.sample_id)
        if entry is None:
            if dataset.require_selected_views:
                raise KeyError(
                    f"Sample {sample.sample_id!r} is missing view {ref.path_parts()}."
                )
            continue
        sample_ref = ref.sample_ref
        views = views_by_ref.setdefault(sample_ref, {})
        views[_view_key(ref)] = _view_value(dataset, view, entry)

    result: dict[tuple[item.Role, item.Modality], item.Item] = {}
    item_entries = {entry.ref: entry for entry in sample.items}
    for sample_ref, views in views_by_ref.items():
        item_entry = item_entries.get(sample_ref)
        result[sample_ref] = _item_from_entry(sample_ref, item_entry, views)

    for sample_ref, item_entry in item_entries.items():
        if sample_ref not in result:
            result[sample_ref] = _item_from_entry(sample_ref, item_entry, {})
    return result


def _item_from_entry(
    sample_ref: tuple[item.Role, item.Modality],
    entry: SampleItemEntry | None,
    views: Mapping[Any, Any],
) -> item.Item:
    _, modality = sample_ref
    required = {} if entry is None else dict(entry.required)
    optional = {} if entry is None else dict(entry.optional)
    match modality:
        case item.Modality.AUDIO:
            return item.AudioItem(
                views=views,
                required=_enum_keys(required, item.AudioKey),
                optional=_enum_keys(optional, item.AudioOptKey),
            )
        case item.Modality.IMAGE:
            return item.ImageItem(
                views=views,
                required=_enum_keys(required, item.ImageKey),
                optional=_enum_keys(optional, item.ImageOptKey),
            )
        case item.Modality.TEXT:
            return item.TextItem(
                views=views,
                required=_enum_keys(required, item.TextKey),
                optional=_enum_keys(optional, item.TextOptKey),
            )
    raise ValueError(f"Unsupported modality: {modality!r}.")


def _view_value(
    dataset: StoreDataset,
    view: StoreView,
    entry: ViewManifestEntry,
) -> Any:
    data = read_payload_bytes(dataset.root, view.ref, view.revision, entry)
    view_key = _view_key(view.ref)
    if view.ref.modality is item.Modality.AUDIO and view_key == item.AudioView.FILE:
        return str(_cache_file_payload(dataset.cache_path, entry, data))
    return payload_value(view.ref, data)


def _cache_file_payload(
    cache_path: Path, entry: ViewManifestEntry, data: bytes
) -> Path:
    target = cache_path / "files" / entry.key
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and matches_checksum(target.read_bytes(), entry.checksum):
        return target
    target.write_bytes(data)
    return target


def _validate_view_ref(ref: ViewRef) -> None:
    if not isinstance(ref, ViewRef):
        raise TypeError("views entries must be ViewRef instances.")
    _view_key(ref)


def _view_key(ref: ViewRef):
    return ref.view_key


def _enum_keys(values: Mapping[str, Any], enum_type):
    converted = {}
    for key, value in values.items():
        converted[enum_type(key)] = value
    return converted
