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


def default_cache_root() -> Path:
    if os.environ.get("ANYDATASET_CACHE_ROOT"):
        return Path(os.environ["ANYDATASET_CACHE_ROOT"]).expanduser()
    return Path("~/.cache/anydataset").expanduser()


@dataclass(frozen=True)
class CacheManifest:
    cache_path: Path
    metadata_path: Path
    lock_path: Path
    ready_path: Path


class CacheManager:
    def __init__(self, cache_root: str | Path | None = None):
        root = cache_root if cache_root is not None else default_cache_root()
        self.root = Path(root).expanduser()

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
        lock = _lock_for(cache.lock_path)
        with lock:
            cache.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with cache.lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _cache_path(self, spec: Spec) -> Path:
        return self.root / spec.cache_relpath


_PROCESS_LOCKS: dict[str, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _PROCESS_LOCKS_GUARD:
        if key not in _PROCESS_LOCKS:
            _PROCESS_LOCKS[key] = threading.Lock()
        return _PROCESS_LOCKS[key]


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)
