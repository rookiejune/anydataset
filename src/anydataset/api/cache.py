from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from .spec import DatasetSpec


@dataclass(frozen=True)
class CacheManifest:
    spec: DatasetSpec
    cache_path: Path
    metadata_path: Path
    lock_path: Path
    ready_path: Path


class CacheManager:
    def __init__(self, cache_dir: str | Path = "~/.cache/anydataset"):
        self.root = Path(cache_dir).expanduser()

    def prepare(self, spec: DatasetSpec) -> CacheManifest:
        cache_path = self.dataset_cache_path(spec)
        cache_path.mkdir(parents=True, exist_ok=True)
        metadata_path = cache_path / "metadata.json"
        lock_path = cache_path / ".prepare.lock"
        ready_path = cache_path / ".ready"
        if not metadata_path.exists():
            _write_json(metadata_path, _spec_metadata(spec))
        return CacheManifest(
            spec=spec,
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

    def dataset_cache_path(self, spec: DatasetSpec) -> Path:
        source = _safe_segment(spec.source)
        name = _safe_segment(spec.name)
        version = _safe_segment(spec.version or _stable_hash(_spec_metadata(spec)))
        return self.root / source / name / version


def _spec_metadata(spec: DatasetSpec) -> dict[str, Any]:
    return {
        "source": spec.source,
        "path": spec.path,
        "name": spec.name,
        "split": spec.split,
        "version": spec.version,
        "load_options": dict(spec.load_options),
    }


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _safe_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return cleaned.strip("._") or "dataset"


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
