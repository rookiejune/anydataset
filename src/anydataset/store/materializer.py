from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
from typing import Any
import uuid

import torch

from ..modalities import ModalityKey, ViewRef
from ..modalities.audio import AudioView
from .jsonio import read_json, read_jsonl, write_json, write_jsonl
from .paths import (
    dataset_json_path,
    dataset_ready_path,
    samples_jsonl_path,
    view_dir,
    view_json_path,
    view_manifest_path,
    view_ready_path,
    view_shard_path,
)
from .schema import (
    STORE_SCHEMA_VERSION,
    DatasetManifest,
    SampleManifestEntry,
    ViewManifestEntry,
    ViewSelection,
    view_ref_to_dict,
)

type ViewTransform = Callable[["ViewInput"], Any]


@dataclass(frozen=True)
class ViewInput:
    sample: SampleManifestEntry
    ref: ViewRef
    revision: str
    value: Any


@dataclass(init=False)
class ViewMaterializer:
    input_dir: Path
    output_dir: Path
    input_ref: ViewRef
    output_ref: ViewRef
    transform: ViewTransform
    provider_name: str
    provider_version: str
    config: Mapping[str, Any]

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        input_ref: ViewRef,
        output_ref: ViewRef,
        transform: ViewTransform,
        provider_name: str,
        provider_version: str,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input_ref = input_ref
        self.output_ref = output_ref
        self.transform = transform
        self.provider_name = provider_name
        self.provider_version = provider_version
        self.config = {} if config is None else config
        _validate_ref("input_ref", self.input_ref)
        _validate_ref("output_ref", self.output_ref)
        if not callable(self.transform):
            raise TypeError("transform must be callable.")
        _validate_segment("provider_name", self.provider_name)
        _validate_segment("provider_version", self.provider_version)
        _validate_json_mapping("config", self.config)

    def write(self) -> Path:
        _validate_target(self.output_dir)
        tmp_dir = _tmp_dir(self.output_dir)
        tmp_dir.mkdir(parents=True)
        try:
            self._write_to_tmp(tmp_dir)
            if self.output_dir.exists():
                self.output_dir.rmdir()
            os.replace(tmp_dir, self.output_dir)
            return self.output_dir
        except Exception:
            raise

    def _write_to_tmp(self, root: Path) -> None:
        source = _load_source(self.input_dir, self.input_ref)
        output_revision = _revision_for(
            provider_name=self.provider_name,
            provider_version=self.provider_version,
            config=self.config,
            input_ref=self.input_ref,
            input_revision=source.input_view.revision,
            output_ref=self.output_ref,
        )
        _copy_dataset_skeleton(self.input_dir, root, source.dataset, self.output_ref, output_revision)
        _write_materialized_view(
            root=root,
            source=source,
            output_ref=self.output_ref,
            output_revision=output_revision,
            transform=self.transform,
            provider=_provider_metadata(
                self.provider_name,
                self.provider_version,
                self.config,
                self.input_ref,
                source.input_view.revision,
            ),
        )
        dataset_ready_path(root).touch()


@dataclass(frozen=True)
class _SourceDataset:
    root: Path
    dataset: DatasetManifest
    samples: tuple[SampleManifestEntry, ...]
    input_view: "_ViewIndex"


@dataclass(frozen=True)
class _ViewIndex:
    ref: ViewRef
    revision: str
    entries: Mapping[str, ViewManifestEntry]


@dataclass(frozen=True)
class _Payload:
    key: str
    data: bytes
    shape: tuple[int, ...] | None
    dtype: str
    provenance: Mapping[str, Any]


def _load_source(root: Path, input_ref: ViewRef) -> _SourceDataset:
    _validate_dataset_root(root)
    dataset = DatasetManifest.from_dict(read_json(dataset_json_path(root)))
    samples = tuple(
        SampleManifestEntry.from_dict(row)
        for row in read_jsonl(samples_jsonl_path(root))
    )
    if len(samples) != dataset.sample_count:
        raise ValueError("samples.jsonl row count must match dataset.json sample_count.")
    revisions = {selection.ref: selection.revision for selection in dataset.views}
    if input_ref not in revisions:
        raise KeyError(f"Input dataset is missing view {input_ref.path_parts()}.")
    input_view = _load_view(root, input_ref, revisions[input_ref])
    return _SourceDataset(root=root, dataset=dataset, samples=samples, input_view=input_view)


def _validate_dataset_root(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not dataset_ready_path(root).exists():
        raise ValueError(f"Unified dataset is not ready: {root}")
    if not dataset_json_path(root).is_file():
        raise FileNotFoundError(dataset_json_path(root))
    if not samples_jsonl_path(root).is_file():
        raise FileNotFoundError(samples_jsonl_path(root))


def _load_view(root: Path, ref: ViewRef, revision: str) -> _ViewIndex:
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


def _copy_dataset_skeleton(
    input_root: Path,
    output_root: Path,
    dataset: DatasetManifest,
    output_ref: ViewRef,
    output_revision: str,
) -> None:
    selections = {selection.ref: selection for selection in dataset.views}
    for selection in selections.values():
        source = view_dir(input_root, selection.ref, selection.revision)
        target = view_dir(output_root, selection.ref, selection.revision)
        shutil.copytree(source, target)

    selections[output_ref] = ViewSelection(output_ref, output_revision)
    manifest = DatasetManifest(
        dataset_id=dataset.dataset_id,
        split=dataset.split,
        sample_count=dataset.sample_count,
        views=tuple(sorted(selections.values(), key=lambda item: item.ref.path_parts())),
        config=dataset.config,
        provenance=dataset.provenance,
    )
    write_json(dataset_json_path(output_root), manifest.to_dict())
    shutil.copyfile(samples_jsonl_path(input_root), samples_jsonl_path(output_root))


def _write_materialized_view(
    root: Path,
    source: _SourceDataset,
    output_ref: ViewRef,
    output_revision: str,
    transform: ViewTransform,
    provider: Mapping[str, Any],
) -> None:
    shard = "000000.tar"
    entries: list[ViewManifestEntry] = []
    shard_path = view_shard_path(root, output_ref, output_revision, shard)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard_path, "w") as archive:
        for sample in source.samples:
            input_entry = source.input_view.entries.get(sample.sample_id)
            if input_entry is None:
                raise KeyError(f"Sample {sample.sample_id!r} is missing input view.")
            value = _load_value(source.root, source.input_view, input_entry)
            output = transform(
                ViewInput(
                    sample=sample,
                    ref=source.input_view.ref,
                    revision=source.input_view.revision,
                    value=value,
                )
            )
            payload = _payload_for_view(output_ref, sample.sample_id, output, provider)
            info = tarfile.TarInfo(payload.key)
            info.size = len(payload.data)
            info.mtime = 0
            archive.addfile(info, BytesIO(payload.data))
            entries.append(
                ViewManifestEntry(
                    ref=output_ref,
                    revision=output_revision,
                    sample_id=sample.sample_id,
                    shard=shard,
                    key=payload.key,
                    shape=payload.shape,
                    dtype=payload.dtype,
                    checksum=f"sha256:{hashlib.sha256(payload.data).hexdigest()}",
                    provenance=payload.provenance,
                )
            )

    view_json = {
        "schema_version": STORE_SCHEMA_VERSION,
        **view_ref_to_dict(output_ref),
        "revision": output_revision,
        "provider": dict(provider),
    }
    write_json(view_json_path(root, output_ref, output_revision), view_json)
    write_jsonl(
        view_manifest_path(root, output_ref, output_revision),
        (entry.to_dict() for entry in entries),
    )
    view_ready_path(root, output_ref, output_revision).touch()


def _load_value(root: Path, view: _ViewIndex, entry: ViewManifestEntry) -> Any:
    data = _payload_bytes(root, view, entry)
    _validate_checksum(entry, data)
    if view.ref.view_key == AudioView.FILE:
        return data
    return torch.load(BytesIO(data), map_location="cpu")


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


def _payload_for_view(
    ref: ViewRef,
    sample_id: str,
    value: Any,
    provider: Mapping[str, Any],
) -> _Payload:
    if ref.view_key == AudioView.FILE:
        return _file_payload(sample_id, value, provider)
    if ref.view_key == AudioView.WAVEFORM:
        return _tensor_payload(sample_id, value, provider)
    return _torch_payload(sample_id, value, provider)


def _tensor_payload(
    sample_id: str,
    value: Any,
    provider: Mapping[str, Any],
) -> _Payload:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.contiguous()
    buffer = BytesIO()
    torch.save(tensor, buffer)
    return _Payload(
        key=f"{sample_id}.pt",
        data=buffer.getvalue(),
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        provenance=provider,
    )


def _torch_payload(
    sample_id: str,
    value: Any,
    provider: Mapping[str, Any],
) -> _Payload:
    buffer = BytesIO()
    torch.save(value, buffer)
    shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
    dtype = str(value.dtype) if isinstance(value, torch.Tensor) else type(value).__name__
    return _Payload(
        key=f"{sample_id}.pt",
        data=buffer.getvalue(),
        shape=shape,
        dtype=dtype,
        provenance=provider,
    )


def _file_payload(
    sample_id: str,
    value: Any,
    provider: Mapping[str, Any],
) -> _Payload:
    if isinstance(value, bytes):
        data = value
        suffix = ".bin"
    elif isinstance(value, str | Path):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        suffix = path.suffix if path.suffix else ".bin"
    else:
        raise TypeError("file view transforms must return bytes or a filesystem path.")
    return _Payload(
        key=f"{sample_id}{suffix}",
        data=data,
        shape=(len(data),),
        dtype="bytes",
        provenance=provider,
    )


def _revision_for(
    provider_name: str,
    provider_version: str,
    config: Mapping[str, Any],
    input_ref: ViewRef,
    input_revision: str,
    output_ref: ViewRef,
) -> str:
    payload = {
        "provider_name": provider_name,
        "provider_version": provider_version,
        "config": dict(config),
        "input": {**view_ref_to_dict(input_ref), "revision": input_revision},
        "output": view_ref_to_dict(output_ref),
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{_slug(provider_name)}-{digest}"


def _provider_metadata(
    provider_name: str,
    provider_version: str,
    config: Mapping[str, Any],
    input_ref: ViewRef,
    input_revision: str,
) -> dict[str, Any]:
    return {
        "name": provider_name,
        "version": provider_version,
        "config": dict(config),
        "input": {**view_ref_to_dict(input_ref), "revision": input_revision},
    }


def _validate_checksum(entry: ViewManifestEntry, data: bytes) -> None:
    checksum = entry.checksum
    if checksum is None:
        return
    if not checksum.startswith("sha256:"):
        raise ValueError(f"Unsupported checksum: {checksum!r}.")
    actual = f"sha256:{hashlib.sha256(data).hexdigest()}"
    if actual != checksum:
        raise ValueError(f"Checksum mismatch for payload {entry.key!r}.")


def _validate_payload_key(key: str) -> None:
    if Path(key).name != key:
        raise ValueError("View payload keys cannot contain path separators.")


def _validate_ref(name: str, ref: ViewRef) -> None:
    if not isinstance(ref, ViewRef):
        raise TypeError(f"{name} must be a ViewRef.")
    if ref.modality is not ModalityKey.AUDIO:
        raise ValueError("ViewMaterializer MVP only supports audio views.")


def _validate_target(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Target path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"Target directory must be empty: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _tmp_dir(path: Path) -> Path:
    return path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"


def _validate_segment(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value in {"", ".", ".."}:
        raise ValueError(f"{name} must be a non-empty path segment.")
    if "/" in value:
        raise ValueError(f"{name} cannot contain '/'.")


def _validate_json_mapping(name: str, value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    try:
        json.dumps(dict(value), ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON serializable.") from exc


def _slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-")
    return text or "view"
