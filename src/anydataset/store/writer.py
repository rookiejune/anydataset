from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import os
from pathlib import Path
import re
import tarfile
import uuid
from typing import Any

import torch

from ..modalities import ModalityKey, ViewRef
from ..modalities.audio import AudioKey, AudioOptKey, AudioView
from ..modalities.text import TextKey
from ..samples import Sample
from .jsonio import write_json, write_jsonl
from .paths import (
    dataset_json_path,
    dataset_ready_path,
    samples_jsonl_path,
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


DEFAULT_AUDIO_VIEWS = (
    ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM),
    ViewRef(ModalityKey.AUDIO, AudioView.FILE),
)
WRITER_PROVIDER = {"name": "DatasetWriter", "version": "mvp"}


@dataclass
class DatasetWriter:
    output_dir: str | Path
    dataset_id: str
    split: str | None = None
    revision: str = "raw"
    views: tuple[ViewRef, ...] | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.views is not None:
            self.views = tuple(self.views)
            for ref in self.views:
                _validate_supported_ref(ref)

    def write(self, samples: Iterable[Sample]) -> Path:
        output_dir = Path(self.output_dir)
        _validate_target(output_dir)
        tmp_dir = _tmp_dir(output_dir)
        tmp_dir.mkdir(parents=True)
        try:
            self._write_to_tmp(tmp_dir, samples)
            if output_dir.exists():
                output_dir.rmdir()
            os.replace(tmp_dir, output_dir)
            return output_dir
        except Exception:
            raise

    def _write_to_tmp(self, root: Path, samples: Iterable[Sample]) -> Path:
        sinks: dict[ViewRef, _ViewSink] = {}
        sample_entries: list[SampleManifestEntry] = []

        try:
            for index, sample in enumerate(samples):
                if not isinstance(sample, Sample):
                    raise TypeError("DatasetWriter.write expects Sample instances.")
                sample_id = _sample_id(sample, index)
                audio = _audio_mapping(sample)
                refs = self.views if self.views is not None else _available_supported_refs(audio)
                if not refs:
                    raise ValueError(f"Sample {sample_id} has no supported audio views.")
                sample_entries.append(_sample_manifest_entry(sample, sample_id, audio))
                for ref in refs:
                    value = _view_value(audio, ref)
                    if value is None:
                        if self.views is not None:
                            raise KeyError(f"Sample {sample_id} is missing view {ref.path_parts()}.")
                        continue
                    sink = sinks.get(ref)
                    if sink is None:
                        sink = _ViewSink(root=root, ref=ref, revision=self.revision)
                        sinks[ref] = sink
                    sink.write(sample_id, value)

            selections = tuple(
                ViewSelection(ref, self.revision)
                for ref in sorted(sinks, key=lambda item: item.path_parts())
            )
            manifest = DatasetManifest(
                dataset_id=self.dataset_id,
                split=self.split,
                sample_count=len(sample_entries),
                views=selections,
                config=self.config,
                provenance=self.provenance,
            )
            write_json(dataset_json_path(root), manifest.to_dict())
            write_jsonl(samples_jsonl_path(root), (entry.to_dict() for entry in sample_entries))
            for sink in sinks.values():
                sink.close()
            dataset_ready_path(root).touch()
            return root
        except Exception:
            for sink in sinks.values():
                sink.close_payload()
            raise


class _ViewSink:
    def __init__(self, root: Path, ref: ViewRef, revision: str):
        self.root = root
        self.ref = ref
        self.revision = revision
        self.shard = "000000.tar"
        self.entries: list[ViewManifestEntry] = []
        path = view_shard_path(root, ref, revision, self.shard)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tar = tarfile.open(path, "w")
        self.closed = False

    def write(self, sample_id: str, value: Any) -> None:
        payload = _payload_for_view(self.ref, sample_id, value)
        info = tarfile.TarInfo(payload.key)
        info.size = len(payload.data)
        info.mtime = 0
        self.tar.addfile(info, BytesIO(payload.data))
        self.entries.append(
            ViewManifestEntry(
                ref=self.ref,
                revision=self.revision,
                sample_id=sample_id,
                shard=self.shard,
                key=payload.key,
                shape=payload.shape,
                dtype=payload.dtype,
                checksum=f"sha256:{hashlib.sha256(payload.data).hexdigest()}",
                provenance=payload.provenance,
            )
        )

    def close(self) -> None:
        self.close_payload()
        view_json = {
            "schema_version": STORE_SCHEMA_VERSION,
            **view_ref_to_dict(self.ref),
            "revision": self.revision,
            "provider": WRITER_PROVIDER,
        }
        write_json(view_json_path(self.root, self.ref, self.revision), view_json)
        write_jsonl(
            view_manifest_path(self.root, self.ref, self.revision),
            (entry.to_dict() for entry in self.entries),
        )
        view_ready_path(self.root, self.ref, self.revision).touch()

    def close_payload(self) -> None:
        if not self.closed:
            self.tar.close()
            self.closed = True


@dataclass(frozen=True)
class _Payload:
    key: str
    data: bytes
    shape: tuple[int, ...] | None
    dtype: str
    provenance: Mapping[str, Any]


def _validate_target(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Target path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"Target directory must be empty: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _tmp_dir(path: Path) -> Path:
    return path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"


def _validate_supported_ref(ref: ViewRef) -> None:
    if not isinstance(ref, ViewRef):
        raise TypeError("views entries must be ViewRef instances.")
    if ref.role is not None:
        raise ValueError("DatasetWriter MVP only supports default-role views.")
    if ref.modality is not ModalityKey.AUDIO:
        raise ValueError("DatasetWriter MVP only supports audio views.")
    if ref.view_key not in {AudioView.WAVEFORM, AudioView.FILE}:
        raise ValueError("DatasetWriter MVP only supports waveform and file audio views.")


def _audio_mapping(sample: Sample) -> Mapping[str, Any]:
    if ModalityKey.AUDIO not in sample.data:
        raise KeyError(f"Sample {sample.dataset_name}:{sample.sample_index} has no audio modality.")
    audio = sample.data[ModalityKey.AUDIO]
    if not isinstance(audio, Mapping):
        raise TypeError("audio modality must be a mapping.")
    if AudioKey.VIEWS not in audio:
        raise KeyError("audio modality must include views.")
    if not isinstance(audio[AudioKey.VIEWS], Mapping):
        raise TypeError("audio views must be a mapping.")
    return audio


def _available_supported_refs(audio: Mapping[str, Any]) -> tuple[ViewRef, ...]:
    views = audio[AudioKey.VIEWS]
    return tuple(ref for ref in DEFAULT_AUDIO_VIEWS if ref.view_key in views)


def _view_value(audio: Mapping[str, Any], ref: ViewRef) -> Any:
    _validate_supported_ref(ref)
    return audio[AudioKey.VIEWS].get(ref.view_key)


def _sample_manifest_entry(
    sample: Sample,
    sample_id: str,
    audio: Mapping[str, Any],
) -> SampleManifestEntry:
    text = sample.data.get(ModalityKey.TEXT)
    if isinstance(text, Mapping):
        content = text.get(TextKey.CONTENT)
        if content is not None:
            content = str(content)
    elif text is not None:
        content = str(text)
    else:
        content = None
    return SampleManifestEntry(
        sample_id=sample_id,
        dataset_name=sample.dataset_name,
        sample_index=sample.sample_index,
        source={
            "dataset_name": sample.dataset_name,
            "sample_index": sample.sample_index,
        },
        modality=ModalityKey.AUDIO,
        duration=_optional_float(audio.get(AudioOptKey.DURATION)),
        sample_rate=_optional_int(audio.get(AudioKey.SAMPLE_RATE)),
        label=audio.get(AudioOptKey.LABEL),
        text=content,
        metadata=_sample_metadata(audio),
    )


def _sample_metadata(audio: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {}
    labels = audio.get(AudioOptKey.LABELS)
    if labels is not None:
        metadata["labels"] = labels
    return metadata


def _payload_for_view(ref: ViewRef, sample_id: str, value: Any) -> _Payload:
    if ref.view_key == AudioView.WAVEFORM:
        return _waveform_payload(sample_id, value)
    if ref.view_key == AudioView.FILE:
        return _file_payload(sample_id, value)
    raise ValueError(f"Unsupported view: {ref.view_key}")


def _waveform_payload(sample_id: str, value: Any) -> _Payload:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.contiguous()
    buffer = BytesIO()
    torch.save(tensor, buffer)
    return _Payload(
        key=f"{sample_id}.pt",
        data=buffer.getvalue(),
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        provenance=WRITER_PROVIDER,
    )


def _file_payload(sample_id: str, value: Any) -> _Payload:
    if not isinstance(value, str | Path):
        raise TypeError("file audio view must be a filesystem path.")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    suffix = path.suffix if path.suffix else ".bin"
    return _Payload(
        key=f"{sample_id}{suffix}",
        data=data,
        shape=(len(data),),
        dtype="bytes",
        provenance={**WRITER_PROVIDER, "source_path": str(path)},
    )


def _sample_id(sample: Sample, index: int) -> str:
    dataset = _slug(sample.dataset_name)
    return f"{index:012d}-{dataset}-{sample.sample_index:012d}"


def _slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-")
    return text or "sample"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
