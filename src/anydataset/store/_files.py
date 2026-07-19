from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from ..cache import anydataset_home
from ..types.item import Modality, Role, View
from .manifest import ViewManifestEntry
from .paths import view_shard_path

_StatFingerprint = tuple[int, int, int, int, int]
_LEASES: dict[tuple[int, str], int] = {}
_LEASES_LOCK = threading.Lock()


class StoreFilesInUseError(RuntimeError):
    pass


class StoreFilesLease:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.cache_path = files_dir(self.root)
        self.lock_path = lease_path(self.root)
        self._fd: int | None = None
        self._pid = os.getpid()
        self._acquire()

    @property
    def active(self) -> bool:
        return self._fd is not None

    def close(self) -> None:
        if self._fd is None:
            return
        self._reset_after_fork()
        fd = self._fd
        self._fd = None
        _unregister(self.lock_path)
        os.close(fd)

    def __enter__(self) -> StoreFilesLease:
        if self._fd is None:
            raise RuntimeError("Store files lease is closed.")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "cache_path": self.cache_path,
            "lock_path": self.lock_path,
            "active": self.active,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.root = Path(state["root"])
        self.cache_path = Path(state["cache_path"])
        self.lock_path = Path(state["lock_path"])
        self._fd = None
        self._pid = os.getpid()
        if state["active"]:
            self._acquire()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _acquire(self) -> None:
        path = self.lock_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._pid = os.getpid()
        _register(path)

    def _reset_after_fork(self) -> None:
        if self._fd is None or self._pid == os.getpid():
            return
        inherited = self._fd
        self._fd = None
        self._acquire()
        os.close(inherited)


def lease_store_files(root: str | Path) -> StoreFilesLease:
    return StoreFilesLease(root)


def cleanup_store_files(root: str | Path) -> bool:
    resolved = Path(root).expanduser().resolve()
    lock = lease_path(resolved)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if _registered(lock):
            raise StoreFilesInUseError(
                f"Store file cache is leased by an active reader: {resolved}"
            )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StoreFilesInUseError(
                f"Store file cache is leased by an active reader: {resolved}"
            ) from exc
        path = files_dir(resolved)
        if path.is_symlink():
            raise ValueError(f"Store file cache path must be a directory: {path}")
        if not path.exists():
            return False
        if not path.is_dir():
            raise ValueError(f"Store file cache path must be a directory: {path}")
        shutil.rmtree(path)
        return True
    finally:
        os.close(fd)


def payload_path(
    root: str | Path,
    view: tuple[Role, Modality, View],
    entry: ViewManifestEntry,
    *,
    cache_path: Path | None = None,
) -> Path:
    role, modality, key = view
    shard = view_shard_path(root, view, entry.shard)
    device, inode, size, modified, changed = stat_fingerprint(shard.stat())
    payload_id = hashlib.sha256(
        (
            f"{entry.shard}\0{entry.key}\0{device}\0{inode}\0{size}\0"
            f"{modified}\0{changed}"
        ).encode("utf-8")
    ).hexdigest()
    suffix = Path(entry.key).suffix or ".bin"
    return (
        (files_dir(root) if cache_path is None else cache_path)
        / role.value
        / modality.value
        / key.value
        / f"{payload_id}{suffix}"
    )


def files_dir(root: str | Path) -> Path:
    return anydataset_home() / "cache" / "store-files" / store_id(root)


def lease_path(root: str | Path) -> Path:
    return anydataset_home() / "cache" / "store-file-leases" / f"{store_id(root)}.lock"


def store_id(root: str | Path) -> str:
    resolved = Path(root).expanduser().resolve()
    return hashlib.sha256(os.fsencode(resolved)).hexdigest()


def stat_fingerprint(stat: os.stat_result) -> _StatFingerprint:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _register(path: Path) -> None:
    key = os.getpid(), str(path)
    with _LEASES_LOCK:
        _LEASES[key] = _LEASES.get(key, 0) + 1


def _unregister(path: Path) -> None:
    key = os.getpid(), str(path)
    with _LEASES_LOCK:
        count = _LEASES[key] - 1
        if count:
            _LEASES[key] = count
        else:
            del _LEASES[key]


def _registered(path: Path) -> bool:
    with _LEASES_LOCK:
        return _LEASES.get((os.getpid(), str(path)), 0) > 0
