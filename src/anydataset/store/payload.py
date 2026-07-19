from __future__ import annotations

import os
import tarfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch

from ..types.item import AudioView, Modality, Role, TextView, View
from .manifest import ViewManifestEntry
from .paths import view_shard_path

_DEFAULT_MAX_OPEN_SHARDS = 8
_StatFingerprint = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class Payload:
    key: str
    data: bytes


@dataclass
class _OpenArchive:
    archive: tarfile.TarFile
    members: dict[str, tarfile.TarInfo]
    fingerprint: _StatFingerprint

    def close(self) -> None:
        self.archive.close()


class PayloadCache:
    def __init__(self, max_open_shards: int = _DEFAULT_MAX_OPEN_SHARDS) -> None:
        if not isinstance(max_open_shards, int) or isinstance(max_open_shards, bool):
            raise TypeError("max_open_shards must be an integer.")
        if max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive.")
        self.max_open_shards = max_open_shards
        self._pid = os.getpid()
        self._archives: OrderedDict[Path, _OpenArchive] = OrderedDict()
        self._lock = threading.RLock()

    def read(
        self,
        root: str | Path,
        view: tuple[Role, Modality, View],
        entry: ViewManifestEntry,
    ) -> bytes:
        shard_path = _payload_shard_path(root, view, entry)
        self._reset_after_fork()
        with self._lock:
            opened = self._archive(shard_path)
            member = opened.members.get(entry.key)
            if member is None:
                raise KeyError(
                    f"View shard {entry.shard!r} is missing payload {entry.key!r}."
                )
            payload = opened.archive.extractfile(member)
            if payload is None:
                raise KeyError(
                    f"View shard {entry.shard!r} is missing payload {entry.key!r}."
                )
            return payload.read()

    def close(self) -> None:
        self._reset_after_fork()
        with self._lock:
            archives = tuple(self._archives.values())
            self._archives.clear()
        for archive in archives:
            archive.close()

    def __getstate__(self) -> dict[str, int]:
        return {"max_open_shards": self.max_open_shards}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(state["max_open_shards"])

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _archive(self, path: Path) -> _OpenArchive:
        fingerprint = _stat_fingerprint(path.stat())
        cached = self._archives.get(path)
        if cached is not None:
            if cached.fingerprint == fingerprint:
                self._archives.move_to_end(path)
                return cached
            del self._archives[path]
            cached.close()

        archive = tarfile.open(path, "r")
        try:
            opened_fingerprint = _stat_fingerprint(
                os.fstat(archive.fileobj.fileno())
            )
            if opened_fingerprint != _stat_fingerprint(path.stat()):
                raise ValueError(f"View shard changed while opening: {path}")
            opened = _OpenArchive(
                archive=archive,
                members={member.name: member for member in archive.getmembers()},
                fingerprint=opened_fingerprint,
            )
        except Exception:
            archive.close()
            raise
        self._archives[path] = opened
        self._evict()
        return opened

    def _evict(self) -> None:
        while len(self._archives) > self.max_open_shards:
            _path, opened = self._archives.popitem(last=False)
            opened.close()

    def _reset_after_fork(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        archives = tuple(self._archives.values())
        self._archives = OrderedDict()
        self._lock = threading.RLock()
        self._pid = pid
        for archive in archives:
            archive.close()


def _stat_fingerprint(stat: os.stat_result) -> _StatFingerprint:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def payload_for_view(
    view: tuple[Role, Modality, View],
    sample_index: int,
    value: Any,
) -> Payload:
    _, modality, key = view
    if modality is Modality.AUDIO and key == AudioView.FILE:
        return _file_payload(sample_index, value)
    if modality is Modality.AUDIO and key == AudioView.WAVEFORM:
        return _waveform_payload(sample_index, value)
    if modality is Modality.TEXT and key == TextView.TEXT:
        return _text_payload(sample_index, value)
    return _torch_payload(sample_index, value)


def payload_value(view: tuple[Role, Modality, View], data: bytes) -> Any:
    _, modality, key = view
    if modality is Modality.AUDIO and key == AudioView.FILE:
        return data
    if modality is Modality.TEXT and key == TextView.TEXT:
        return data.decode("utf-8")
    return torch.load(BytesIO(data), map_location="cpu")


def read_payload_bytes(
    root: str | Path,
    view: tuple[Role, Modality, View],
    entry: ViewManifestEntry,
    *,
    cache: PayloadCache | None = None,
) -> bytes:
    if cache is not None:
        return cache.read(root, view, entry)
    shard_path = _payload_shard_path(root, view, entry)
    with tarfile.open(shard_path, "r") as archive:
        payload = archive.extractfile(entry.key)
        if payload is None:
            raise KeyError(
                f"View shard {entry.shard!r} is missing payload {entry.key!r}."
            )
        data = payload.read()
    return data


def add_payload(archive: tarfile.TarFile, payload: Payload) -> None:
    info = tarfile.TarInfo(payload.key)
    info.size = len(payload.data)
    info.mtime = 0
    archive.addfile(info, BytesIO(payload.data))


def _torch_payload(
    sample_index: int,
    value: Any,
) -> Payload:
    tensor = _maybe_tensor(value)
    payload_value = tensor if tensor is not None else value
    buffer = BytesIO()
    torch.save(payload_value, buffer)
    return Payload(
        key=f"{_sample_key(sample_index)}.pt",
        data=buffer.getvalue(),
    )


def _waveform_payload(
    sample_index: int,
    value: Any,
) -> Payload:
    waveform, sample_rate = _waveform_value(value)
    buffer = BytesIO()
    torch.save((waveform, sample_rate), buffer)
    return Payload(
        key=f"{_sample_key(sample_index)}.pt",
        data=buffer.getvalue(),
    )


def _file_payload(
    sample_index: int,
    value: Any,
) -> Payload:
    if isinstance(value, bytes):
        data = value
        suffix = ".bin"
    elif isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        suffix = path.suffix if path.suffix else ".bin"
    else:
        raise TypeError("file views must be bytes or a filesystem path.")
    return Payload(
        key=f"{_sample_key(sample_index)}{suffix}",
        data=data,
    )


def _text_payload(
    sample_index: int,
    value: Any,
) -> Payload:
    if not isinstance(value, str):
        raise TypeError("text views must be strings.")
    data = value.encode("utf-8")
    return Payload(
        key=f"{_sample_key(sample_index)}.txt",
        data=data,
    )


def _sample_key(sample_index: int) -> str:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")
    return f"{sample_index:012d}"


def _maybe_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, (int, float, bool, list, tuple)):
        try:
            return torch.as_tensor(value).contiguous()
        except (TypeError, ValueError):
            return None
    return None


def _waveform_value(value: Any) -> tuple[torch.Tensor, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("waveform views must be (waveform, sample_rate).")
    waveform, sample_rate = value
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().cpu().contiguous()
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise TypeError("waveform sample_rate must be an integer.")
    return waveform, sample_rate


def _validate_payload_key(key: str) -> None:
    if Path(key).name != key:
        raise ValueError("View payload keys cannot contain path separators.")


def _payload_shard_path(
    root: str | Path,
    view: tuple[Role, Modality, View],
    entry: ViewManifestEntry,
) -> Path:
    _validate_payload_key(entry.key)
    shard_path = view_shard_path(root, view, entry.shard)
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    return shard_path
