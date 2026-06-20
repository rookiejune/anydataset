from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from anydataset.datasets.base import DatasetAdapter

if TYPE_CHECKING:
    from anydataset.api.cache import CacheManifest
    from anydataset.api.spec import DatasetSpec


class LocalFilesDataset(DatasetAdapter):
    def prepare(self, spec: DatasetSpec, cache: CacheManifest) -> dict[str, Any]:
        return {
            "path": Path(spec.path).expanduser(),
            "split": spec.split,
            "cache_path": cache.cache_path,
        }

    def iter_samples(self, manifest: dict[str, Any]) -> Iterator[dict]:
        path = manifest["path"]
        if not path.exists():
            raise FileNotFoundError(path)

        if path.is_file():
            yield from self._iter_file(path)
            return

        for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
            yield {"path": str(file_path)}

    def _iter_file(self, path: Path) -> Iterator[dict]:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return

        yield {"path": str(path)}
