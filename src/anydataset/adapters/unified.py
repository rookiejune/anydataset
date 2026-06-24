from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import tarfile
from typing import TYPE_CHECKING, Any, Iterator

import torch

from ..modalities import ModalityKey, ModalityRole, ViewRef
from ..modalities.audio import AudioKey, AudioOptKey, AudioView
from ..modalities.text import TextKey
from ..store import (
    DatasetManifest,
    SampleManifestEntry,
    ViewManifestEntry,
    dataset_json_path,
    dataset_ready_path,
    read_json,
    read_jsonl,
    samples_jsonl_path,
    view_manifest_path,
    view_ready_path,
    view_shard_path,
)
from .base import DatasetAdapter, MissingModalityError

if TYPE_CHECKING:
    from ..api.cache import CacheManifest
    from ..api.spec import DatasetSpec


class UnifiedDatasetAdapter(DatasetAdapter):
    def __init__(self, views: Sequence[ViewRef] | None = None):
        self.views = None if views is None else tuple(views)
        if self.views is not None:
            for ref in self.views:
                _validate_supported_ref(ref)

    def prepare(self, spec: "DatasetSpec", cache: "CacheManifest") -> "_UnifiedManifest":
        root = Path(spec.path).expanduser()
        _validate_dataset_root(root)
        dataset = DatasetManifest.from_dict(read_json(dataset_json_path(root)))
        _validate_split(spec.split, dataset)
        samples = tuple(
            SampleManifestEntry.from_dict(row)
            for row in read_jsonl(samples_jsonl_path(root))
        )
        if len(samples) != dataset.sample_count:
            raise ValueError("samples.jsonl row count must match dataset.json sample_count.")

        selections = _selected_views(dataset, self.views)
        views = {
            selection.ref: _load_view(root, selection.ref, selection.revision)
            for selection in selections
        }
        return _UnifiedManifest(
            root=root,
            cache_path=cache.cache_path,
            dataset=dataset,
            samples=samples,
            views=views,
            require_selected_views=self.views is not None,
        )

    def iter_samples(self, manifest: "_UnifiedManifest") -> Iterator[dict]:
        for sample in manifest.samples:
            yield _row_for_sample(manifest, sample)

    def audio(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        if role is not None or ModalityKey.AUDIO not in row:
            raise MissingModalityError(ModalityKey.AUDIO, role)
        audio = row[ModalityKey.AUDIO]
        if not isinstance(audio, Mapping):
            raise TypeError("audio modality must be a mapping.")
        return audio

    def text(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        if role is not None or ModalityKey.TEXT not in row:
            raise MissingModalityError(ModalityKey.TEXT, role)
        text = row[ModalityKey.TEXT]
        if not isinstance(text, Mapping):
            raise TypeError("text modality must be a mapping.")
        return text


@dataclass(frozen=True)
class _UnifiedManifest:
    root: Path
    cache_path: Path
    dataset: DatasetManifest
    samples: tuple[SampleManifestEntry, ...]
    views: Mapping[ViewRef, "_ViewIndex"]
    require_selected_views: bool


@dataclass(frozen=True)
class _ViewIndex:
    ref: ViewRef
    revision: str
    entries: Mapping[str, ViewManifestEntry]


def _validate_dataset_root(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not dataset_ready_path(root).exists():
        raise ValueError(f"Unified dataset is not ready: {root}")
    if not dataset_json_path(root).is_file():
        raise FileNotFoundError(dataset_json_path(root))
    if not samples_jsonl_path(root).is_file():
        raise FileNotFoundError(samples_jsonl_path(root))


def _validate_split(split: str | None, dataset: DatasetManifest) -> None:
    if split is not None and dataset.split is not None and split != dataset.split:
        raise ValueError(
            f"Dataset split {dataset.split!r} does not match requested split {split!r}."
        )


def _selected_views(
    dataset: DatasetManifest,
    requested: tuple[ViewRef, ...] | None,
):
    available = {selection.ref: selection for selection in dataset.views}
    if requested is not None:
        missing = [ref for ref in requested if ref not in available]
        if missing:
            raise KeyError(f"Unified dataset is missing requested views: {missing!r}.")
        return tuple(available[ref] for ref in requested)

    selections = tuple(
        selection
        for selection in dataset.views
        if _is_supported_ref(selection.ref)
    )
    if not selections:
        raise ValueError("Unified dataset does not contain supported audio views.")
    return selections


def _load_view(root: Path, ref: ViewRef, revision: str) -> _ViewIndex:
    _validate_supported_ref(ref)
    if not view_ready_path(root, ref, revision).exists():
        raise ValueError(f"Unified dataset view is not ready: {ref.path_parts()}.")
    entries: dict[str, ViewManifestEntry] = {}
    for row in read_jsonl(view_manifest_path(root, ref, revision)):
        entry = ViewManifestEntry.from_dict(row)
        if entry.ref != ref:
            raise ValueError("View manifest entry ref must match its path.")
        if entry.revision != revision:
            raise ValueError("View manifest entry revision must match its path.")
        if entry.sample_id in entries:
            raise ValueError(f"Duplicate view entry for sample_id {entry.sample_id!r}.")
        entries[entry.sample_id] = entry
    return _ViewIndex(ref=ref, revision=revision, entries=entries)


def _row_for_sample(
    manifest: _UnifiedManifest,
    sample: SampleManifestEntry,
) -> dict[str, Any]:
    views: dict[str, Any] = {}
    for ref, view in manifest.views.items():
        entry = view.entries.get(sample.sample_id)
        if entry is None:
            if manifest.require_selected_views:
                raise KeyError(f"Sample {sample.sample_id!r} is missing view {ref.path_parts()}.")
            continue
        views[ref.view_key] = _view_value(manifest, view, entry)
    if not views:
        raise ValueError(f"Sample {sample.sample_id!r} has no readable audio views.")

    audio: dict[str, Any] = {
        AudioKey.SAMPLE_RATE: sample.sample_rate,
        AudioKey.VIEWS: views,
    }
    if sample.duration is not None:
        audio[AudioOptKey.DURATION] = sample.duration
    if sample.label is not None:
        audio[AudioOptKey.LABEL] = sample.label
    labels = sample.metadata.get("labels")
    if labels is not None:
        audio[AudioOptKey.LABELS] = labels

    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "source": dict(sample.source),
        ModalityKey.AUDIO: audio,
    }
    if sample.text is not None:
        row[ModalityKey.TEXT] = {TextKey.CONTENT: sample.text}
    return row


def _view_value(
    manifest: _UnifiedManifest,
    view: _ViewIndex,
    entry: ViewManifestEntry,
) -> Any:
    data = _payload_bytes(manifest.root, view, entry)
    _validate_checksum(entry, data)
    if view.ref.view_key == AudioView.WAVEFORM:
        return torch.load(BytesIO(data), map_location="cpu")
    if view.ref.view_key == AudioView.FILE:
        return str(_cache_file_payload(manifest.cache_path, entry, data))
    raise ValueError(f"Unsupported view: {view.ref.view_key}")


def _payload_bytes(root: Path, view: _ViewIndex, entry: ViewManifestEntry) -> bytes:
    _validate_payload_key(entry.key)
    shard_path = view_shard_path(root, view.ref, view.revision, entry.shard)
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    with tarfile.open(shard_path, "r") as archive:
        payload = archive.extractfile(entry.key)
        if payload is None:
            raise KeyError(f"View shard {entry.shard!r} is missing payload {entry.key!r}.")
        return payload.read()


def _cache_file_payload(cache_path: Path, entry: ViewManifestEntry, data: bytes) -> Path:
    target = cache_path / "files" / entry.key
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and _matches_checksum(target.read_bytes(), entry.checksum):
        return target
    target.write_bytes(data)
    return target


def _validate_payload_key(key: str) -> None:
    if Path(key).name != key:
        raise ValueError("View payload keys cannot contain path separators.")


def _validate_checksum(entry: ViewManifestEntry, data: bytes) -> None:
    checksum = entry.checksum
    if checksum is None:
        return
    if not checksum.startswith("sha256:"):
        raise ValueError(f"Unsupported checksum: {checksum!r}.")
    if _sha256(data) != checksum:
        raise ValueError(f"Checksum mismatch for payload {entry.key!r}.")


def _matches_checksum(data: bytes, checksum: str | None) -> bool:
    if checksum is None:
        return True
    if not checksum.startswith("sha256:"):
        return False
    return _sha256(data) == checksum


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _validate_supported_ref(ref: ViewRef) -> None:
    if not isinstance(ref, ViewRef):
        raise TypeError("views entries must be ViewRef instances.")
    if ref.role is not None:
        raise ValueError("UnifiedDatasetAdapter MVP only supports default-role views.")
    if ref.modality is not ModalityKey.AUDIO:
        raise ValueError("UnifiedDatasetAdapter MVP only supports audio views.")
    if ref.view_key not in {AudioView.WAVEFORM, AudioView.FILE}:
        raise ValueError("UnifiedDatasetAdapter MVP only supports waveform and file audio views.")


def _is_supported_ref(ref: ViewRef) -> bool:
    return (
        isinstance(ref, ViewRef)
        and ref.role is None
        and ref.modality is ModalityKey.AUDIO
        and ref.view_key in {AudioView.WAVEFORM, AudioView.FILE}
    )
