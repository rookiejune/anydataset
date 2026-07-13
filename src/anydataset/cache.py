from __future__ import annotations

import fcntl
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import Spec


def anydataset_home() -> Path:
    if os.environ.get("ANYDATASET_HOME"):
        return Path(os.environ["ANYDATASET_HOME"]).expanduser()
    return Path("~/.cache/anydataset").expanduser()


def _cache_dir() -> Path:
    return anydataset_home() / "cache"


@dataclass(frozen=True)
class CacheManifest:
    cache_path: Path
    metadata_path: Path
    lock_path: Path
    ready_path: Path


class CacheManager:
    def __init__(self) -> None:
        self.root = _cache_dir()

    def prepare(self, spec: Spec) -> CacheManifest:
        cache_path = self._cache_path(spec)
        cache_path.mkdir(parents=True, exist_ok=True)
        metadata_path = cache_path / "metadata.json"
        lock_path = cache_path / ".prepare.lock"
        ready_path = cache_path / ".ready"
        if not metadata_path.exists():
            _write_json(metadata_path, spec.to_dict())
        return CacheManifest(
            cache_path=cache_path,
            metadata_path=metadata_path,
            lock_path=lock_path,
            ready_path=ready_path,
        )

    def is_ready(self, cache: CacheManifest) -> bool:
        return cache.ready_path.exists()

    def mark_ready(self, cache: CacheManifest) -> None:
        cache.ready_path.write_text("ready\n", encoding="utf-8")

    @contextmanager
    def prepare_lock(self, cache: CacheManifest):
        with FileLock(cache.lock_path):
            yield

    def _cache_path(self, spec: Spec) -> Path:
        return self.root / "sources" / spec.cache_relpath


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class FileLockError(RuntimeError):
    pass


class FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None

    def __enter__(self):
        lock = _thread_lock(self.path)
        if not lock.acquire(blocking=False):
            raise FileLockError(f"File lock is already held: {self.path}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(
                    self.file.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as exc:
                raise FileLockError(
                    f"File lock is already held: {self.path}"
                ) from exc
            return self
        except Exception:
            if self.file is not None:
                self.file.close()
                self.file = None
            lock.release()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.file is not None:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()
            self.file = None
        _thread_lock(self.path).release()


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        if key not in _LOCKS:
            _LOCKS[key] = threading.Lock()
        return _LOCKS[key]


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)
