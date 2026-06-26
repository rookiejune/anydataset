from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset

from ..types import item
from .jsonio import read_json
from .manifest import (
    DatasetManifest,
    SampleManifestEntry,
    ViewManifestEntry,
    view_from_dict,
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
from .payload import payload_value, read_payload_bytes


@dataclass(frozen=True)
class StoreDataset(Dataset):
    root: Path
    manifest: DatasetManifest
    samples: tuple[SampleManifestEntry, ...]
    views: Mapping[tuple[item.Role, item.Modality, item.View], "StoreView"]
    _files: dict[str, Path] = field(default_factory=dict, compare=False, repr=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[item.Sample]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> item.Sample:
        return _sample_for_entry(self, self.samples[index])


@dataclass(frozen=True)
class StoreView:
    view: tuple[item.Role, item.Modality, item.View]
    entries: Mapping[str, ViewManifestEntry]


def read_store_dataset(
    root: str | Path,
) -> StoreDataset:
    root = Path(root).expanduser()
    _validate_dataset_root(root)
    manifest = DatasetManifest(**read_json(dataset_json_path(root)))
    samples = tuple(read_samples_manifest(root))
    if len(samples) != manifest.sample_count:
        raise ValueError(
            "sample manifest row count must match dataset.json sample_count."
        )
    _validate_samples(samples)

    indexes = {
        view: _load_view(root, view)
        for view in _discover_views(root)
    }
    _validate_view_coverage(samples, indexes)
    return StoreDataset(
        root=root,
        manifest=manifest,
        samples=samples,
        views=indexes,
    )


def _validate_dataset_root(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not dataset_ready_path(root).exists():
        raise ValueError(f"Store dataset is not ready: {root}")
    if not dataset_json_path(root).is_file():
        raise FileNotFoundError(dataset_json_path(root))
    if not samples_manifest_exists(root):
        raise FileNotFoundError(root / "samples.parquet")


def _load_view(root: Path, view: tuple[item.Role, item.Modality, item.View]) -> StoreView:
    if not view_ready_path(root, view).exists():
        raise ValueError(f"Store dataset view is not ready: {_view_path(view)}.")
    entries: dict[str, ViewManifestEntry] = {}
    for entry in read_view_manifest(root, view):
        if _entry_view(entry) != view:
            raise ValueError("View manifest entry ref must match its path.")
        if entry.sample_id in entries:
            raise ValueError(f"Duplicate view entry for sample_id {entry.sample_id!r}.")
        entries[entry.sample_id] = entry
    return StoreView(view=view, entries=entries)


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
        for name in ("view.json", "manifest.parquet", ".ready", "shards")
    )


def _validate_view_dir(
    path: Path,
    view: tuple[item.Role, item.Modality, item.View],
) -> None:
    if not (path / ".ready").is_file():
        raise ValueError(f"Store dataset view is not ready: {_view_path(view)}.")
    if not (path / "view.json").is_file():
        raise FileNotFoundError(path / "view.json")
    if not (path / "manifest.parquet").is_file():
        raise FileNotFoundError(path / "manifest.parquet")
    declared = _view_from_json(path / "view.json")
    if declared != view:
        raise ValueError(
            f"View metadata {path / 'view.json'} does not match its path."
        )


def _view_from_json(path: Path) -> tuple[item.Role, item.Modality, item.View]:
    try:
        return view_from_dict(read_json(path))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid store dataset view metadata: {path}") from exc


def _validate_samples(samples: tuple[SampleManifestEntry, ...]) -> None:
    sample_ids: set[str] = set()
    for sample in samples:
        if sample.sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id {sample.sample_id!r}.")
        sample_ids.add(sample.sample_id)
        refs: set[tuple[item.Role, item.Modality]] = set()
        for ref, _ in sample.items:
            if ref in refs:
                raise ValueError(f"Duplicate sample item ref {ref!r}.")
            refs.add(ref)


def _validate_view_coverage(
    samples: tuple[SampleManifestEntry, ...],
    views: Mapping[tuple[item.Role, item.Modality, item.View], StoreView],
) -> None:
    expected = _sample_ids_by_item(samples)
    for view, store_view in views.items():
        sample_ref = view[:2]
        expected_ids = expected.get(sample_ref, set())
        actual_ids = set(store_view.entries)
        if actual_ids == expected_ids:
            continue
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        detail = _coverage_detail(missing, extra)
        raise ValueError(f"View {_view_path(view)} sample coverage mismatch: {detail}.")


def _sample_ids_by_item(
    samples: tuple[SampleManifestEntry, ...],
) -> dict[tuple[item.Role, item.Modality], set[str]]:
    sample_ids: dict[tuple[item.Role, item.Modality], set[str]] = {}
    for sample in samples:
        for ref, _ in sample.items:
            sample_ids.setdefault(ref, set()).add(sample.sample_id)
    return sample_ids


def _coverage_detail(missing: list[str], extra: list[str]) -> str:
    details = []
    if missing:
        details.append(f"missing sample_id {missing[0]!r}")
    if extra:
        details.append(f"unexpected sample_id {extra[0]!r}")
    return ", ".join(details)


def _sample_for_entry(
    dataset: StoreDataset,
    sample: SampleManifestEntry,
) -> item.Sample:
    views_by_ref: dict[tuple[item.Role, item.Modality], dict[Any, Any]] = {}
    item_entries = dict(sample.items)
    for view_entry, view in dataset.views.items():
        sample_ref = view_entry[:2]
        if sample_ref not in item_entries:
            continue
        entry = view.entries.get(sample.sample_id)
        if entry is None:
            raise ValueError(
                f"View {_view_path(view_entry)} is missing sample_id {sample.sample_id!r}."
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

    data = read_payload_bytes(dataset.root, view.view, entry)
    return payload_value(view.view, data)


def _cached_file_payload(
    dataset: StoreDataset,
    entry: ViewManifestEntry,
    view: StoreView,
) -> Path:
    cached = dataset._files.get(entry.key)
    if cached is not None:
        return cached

    data = read_payload_bytes(dataset.root, view.view, entry)
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
