from __future__ import annotations

import hashlib
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch

from ..types.item import AudioView, Modality, TextView
from .manifest import ViewManifestEntry, ViewRef
from .paths import view_shard_path


@dataclass(frozen=True)
class Payload:
    key: str
    data: bytes
    shape: tuple[int, ...] | None
    dtype: str
    provenance: Mapping[str, Any]


def payload_for_view(
    ref: ViewRef,
    sample_id: str,
    value: Any,
    provenance: Mapping[str, Any],
) -> Payload:
    view = ref.view_key
    if ref.modality is Modality.AUDIO and view == AudioView.FILE:
        return _file_payload(sample_id, value, provenance)
    if ref.modality is Modality.TEXT and view == TextView.TEXT:
        return _text_payload(sample_id, value, provenance)
    return _torch_payload(sample_id, value, provenance)


def payload_value(ref: ViewRef, data: bytes) -> Any:
    view = ref.view_key
    if ref.modality is Modality.AUDIO and view == AudioView.FILE:
        return data
    if ref.modality is Modality.TEXT and view == TextView.TEXT:
        return data.decode("utf-8")
    return torch.load(BytesIO(data), map_location="cpu")


def read_payload_bytes(
    root: str | Path,
    ref: ViewRef,
    revision: str,
    entry: ViewManifestEntry,
) -> bytes:
    _validate_payload_key(entry.key)
    shard_path = view_shard_path(root, ref, revision, entry.shard)
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    with tarfile.open(shard_path, "r") as archive:
        payload = archive.extractfile(entry.key)
        if payload is None:
            raise KeyError(
                f"View shard {entry.shard!r} is missing payload {entry.key!r}."
            )
        data = payload.read()
    validate_checksum(entry, data)
    return data


def add_payload(archive: tarfile.TarFile, payload: Payload) -> None:
    info = tarfile.TarInfo(payload.key)
    info.size = len(payload.data)
    info.mtime = 0
    archive.addfile(info, BytesIO(payload.data))


def checksum(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def validate_checksum(entry: ViewManifestEntry, data: bytes) -> None:
    expected = entry.checksum
    if expected is None:
        return
    if not expected.startswith("sha256:"):
        raise ValueError(f"Unsupported checksum: {expected!r}.")
    if checksum(data) != expected:
        raise ValueError(f"Checksum mismatch for payload {entry.key!r}.")


def matches_checksum(data: bytes, expected: str | None) -> bool:
    if expected is None:
        return True
    if not expected.startswith("sha256:"):
        return False
    return checksum(data) == expected


def _torch_payload(
    sample_id: str,
    value: Any,
    provenance: Mapping[str, Any],
) -> Payload:
    tensor = _maybe_tensor(value)
    payload_value = tensor if tensor is not None else value
    buffer = BytesIO()
    torch.save(payload_value, buffer)
    return Payload(
        key=f"{sample_id}.pt",
        data=buffer.getvalue(),
        shape=tuple(tensor.shape) if tensor is not None else None,
        dtype=str(tensor.dtype) if tensor is not None else type(value).__name__,
        provenance=provenance,
    )


def _file_payload(
    sample_id: str,
    value: Any,
    provenance: Mapping[str, Any],
) -> Payload:
    if isinstance(value, bytes):
        data = value
        suffix = ".bin"
        source = {}
    elif isinstance(value, str | Path):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        suffix = path.suffix if path.suffix else ".bin"
        source = {"source_path": str(path)}
    else:
        raise TypeError("file views must be bytes or a filesystem path.")
    return Payload(
        key=f"{sample_id}{suffix}",
        data=data,
        shape=(len(data),),
        dtype="bytes",
        provenance={**provenance, **source},
    )


def _text_payload(
    sample_id: str,
    value: Any,
    provenance: Mapping[str, Any],
) -> Payload:
    if not isinstance(value, str):
        raise TypeError("text views must be strings.")
    data = value.encode("utf-8")
    return Payload(
        key=f"{sample_id}.txt",
        data=data,
        shape=(len(data),),
        dtype="text",
        provenance=provenance,
    )


def _maybe_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, int | float | bool | list | tuple):
        try:
            return torch.as_tensor(value).contiguous()
        except (TypeError, ValueError):
            return None
    return None


def _validate_payload_key(key: str) -> None:
    if Path(key).name != key:
        raise ValueError("View payload keys cannot contain path separators.")
