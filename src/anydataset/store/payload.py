from __future__ import annotations

import os
import tarfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, cast

import torch

from .._io.files import StatFingerprint as _StatFingerprint
from .._io.files import stat_fingerprint as _stat_fingerprint
from .._validation import validate_path_segment
from ..types.item import AudioView, Modality, Role, TextView, View
from .jsonio import read_json, write_json
from .manifest import ViewManifestEntry
from .paths import view_shard_index_path, view_shard_path

_DEFAULT_MAX_OPEN_SHARDS = 8
PAYLOAD_INDEX_VERSION = 1


@dataclass(frozen=True)
class Payload:
    key: str
    data: bytes


@dataclass
class _OpenArchive:
    archive: tarfile.TarFile
    members: dict[str, tarfile.TarInfo]
    fingerprint: _StatFingerprint
    members_complete: bool = True
    verified_members: set[str] = field(default_factory=set)

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
            if member is None and not opened.members_complete:
                # A sidecar is only a read optimization.  If it is incomplete
                # or stale, fall back to the archive's authoritative member
                # table rather than returning a false missing-payload error.
                self._load_members(opened)
                member = opened.members.get(entry.key)
            elif (
                member is not None
                and not opened.members_complete
                and entry.key not in opened.verified_members
            ):
                # Verify the indexed entry against its tar header before using
                # it.  This keeps a damaged sidecar from silently returning a
                # different member while retaining O(1) lookup for valid ones.
                if not _indexed_member_matches(opened.archive, member, entry.key):
                    self._load_members(opened)
                    member = opened.members.get(entry.key)
                else:
                    opened.verified_members.add(entry.key)
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
            fileno = getattr(archive.fileobj, "fileno", None)
            if not callable(fileno):
                raise OSError(f"View shard has no file descriptor: {path}")
            opened_fingerprint = _stat_fingerprint(
                os.fstat(cast(Callable[[], int], fileno)())
            )
            if opened_fingerprint != _stat_fingerprint(path.stat()):
                raise ValueError(f"View shard changed while opening: {path}")
            opened = _OpenArchive(
                archive=archive,
                members={},
                fingerprint=opened_fingerprint,
                members_complete=False,
            )
            indexed = _read_payload_index(path, opened_fingerprint)
            if indexed is None:
                self._load_members(opened)
            else:
                opened.members = indexed
        except Exception:
            archive.close()
            raise
        self._archives[path] = opened
        self._evict()
        return opened

    @staticmethod
    def _load_members(opened: _OpenArchive) -> None:
        opened.members = _payload_members(opened.archive)
        opened.members_complete = True
        opened.verified_members.clear()

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
    # Store payloads intentionally support arbitrary Python values, so keep the
    # legacy unpickling behavior explicit and avoid PyTorch's implicit-mode warning.
    return torch.load(BytesIO(data), map_location="cpu", weights_only=False)


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
        member = _payload_members(archive).get(entry.key)
        if member is None:
            raise KeyError(
                f"View shard {entry.shard!r} is missing payload {entry.key!r}."
            )
        payload = archive.extractfile(member)
        if payload is None:
            raise KeyError(
                f"View shard {entry.shard!r} is missing payload {entry.key!r}."
            )
        data = payload.read()
    return data


def add_payload(archive: tarfile.TarFile, payload: Payload) -> None:
    _validate_payload_key(payload.key)
    info = tarfile.TarInfo(payload.key)
    info.size = len(payload.data)
    info.mtime = 0
    archive.addfile(info, BytesIO(payload.data))


def write_payload_index(
    root: str | Path,
    view: tuple[Role, Modality, View],
    shard: str,
) -> Path:
    """Write a seekable payload offset index for a completed tar shard.

    The index is deliberately treated as disposable metadata.  Its fingerprint
    is checked by readers and a missing or malformed index falls back to the
    regular tar member scan, so older stores remain fully readable.
    """

    path = view_shard_path(root, view, shard)
    fingerprint = _stat_fingerprint(path.stat())
    with tarfile.open(path, "r") as archive:
        members = {
            member.name: {
                "offset": member.offset_data,
                "size": member.size,
            }
            for member in _payload_members(archive).values()
        }
    index_path = view_shard_index_path(root, view, shard)
    write_json(
        index_path,
        {
            "version": PAYLOAD_INDEX_VERSION,
            "fingerprint": list(fingerprint),
            "members": members,
        },
    )
    return index_path


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
    validate_path_segment("View payload key", key)


def _payload_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        if member.name in members:
            raise ValueError(
                f"View shard has duplicate payload key {member.name!r}."
            )
        _validate_payload_key(member.name)
        members[member.name] = member
    return members


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


def _read_payload_index(
    shard_path: Path,
    fingerprint: _StatFingerprint,
) -> dict[str, tarfile.TarInfo] | None:
    index_path = shard_path.with_name(f"{shard_path.name}.index.json")
    try:
        data = read_json(index_path)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != PAYLOAD_INDEX_VERSION
    ):
        return None
    raw_fingerprint = data.get("fingerprint")
    if (
        not isinstance(raw_fingerprint, list)
        or len(raw_fingerprint) != len(fingerprint)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in raw_fingerprint
        )
        or tuple(raw_fingerprint) != fingerprint
    ):
        return None
    raw_members = data.get("members")
    if not isinstance(raw_members, dict):
        return None
    members: dict[str, tarfile.TarInfo] = {}
    for name, raw_member in raw_members.items():
        try:
            _validate_payload_key(name)
        except (TypeError, ValueError):
            return None
        if not isinstance(raw_member, dict):
            return None
        offset = raw_member.get("offset")
        size = raw_member.get("size")
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or offset < tarfile.BLOCKSIZE
            or offset % tarfile.BLOCKSIZE != 0
            or offset > fingerprint[2]
            or size > fingerprint[2] - offset
        ):
            return None
        member = tarfile.TarInfo(name)
        member.offset_data = offset
        member.size = size
        members[name] = member
    return members


def _indexed_member_matches(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    key: str,
) -> bool:
    offset = member.offset_data
    if offset < tarfile.BLOCKSIZE or offset % tarfile.BLOCKSIZE != 0:
        return False
    fileobj = archive.fileobj
    if fileobj is None:
        return False
    position: int | None = None
    try:
        position = fileobj.tell()
        fileobj.seek(offset - tarfile.BLOCKSIZE)
        header = fileobj.read(tarfile.BLOCKSIZE)
        if len(header) != tarfile.BLOCKSIZE:
            return False
        actual = tarfile.TarInfo.frombuf(
            header,
            archive.encoding,
            archive.errors,
        )
        return actual.isreg() and actual.name == key and actual.size == member.size
    except (EOFError, IndexError, OSError, tarfile.TarError, ValueError):
        return False
    finally:
        if position is not None:
            try:
                fileobj.seek(position)
            except OSError:
                pass
