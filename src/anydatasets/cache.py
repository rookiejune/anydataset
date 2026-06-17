from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .registry import DatasetSpec


@dataclass(frozen=True)
class CacheManifest:
    spec: DatasetSpec
    cache_path: Path
    metadata_path: Path


class CacheManager:
    def __init__(self, cache_dir: str | Path = "~/.cache/anydatasets"):
        self.cache_dir = Path(cache_dir).expanduser()

    def prepare(self, spec: DatasetSpec) -> CacheManifest:
        cache_path = self.dataset_cache_path(spec)
        cache_path.mkdir(parents=True, exist_ok=True)
        metadata_path = cache_path / "metadata.json"
        if not metadata_path.exists():
            metadata_path.write_text(
                json.dumps(_spec_metadata(spec), ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        return CacheManifest(spec=spec, cache_path=cache_path, metadata_path=metadata_path)

    def dataset_cache_path(self, spec: DatasetSpec) -> Path:
        source = _safe_segment(spec.source)
        name = _safe_segment(spec.name or spec.path)
        version = _safe_segment(spec.version or _stable_hash(_spec_metadata(spec)))
        return self.cache_dir / source / name / version


def _spec_metadata(spec: DatasetSpec) -> dict[str, Any]:
    data = asdict(spec)
    data["adapter"] = type(spec.adapter).__name__ if spec.adapter is not None else None
    return data


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _safe_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return cleaned.strip("._") or "dataset"
