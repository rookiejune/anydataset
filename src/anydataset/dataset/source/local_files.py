from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from ... import types

if TYPE_CHECKING:
    from ...cache import CacheManifest


def prepare_local(spec: types.Spec, cache: CacheManifest) -> dict[str, Any]:
    return {
        "path": Path(spec.path).expanduser(),
        "split": spec.split,
        "cache_path": cache.cache_path,
    }


def iter_local(state: Mapping[str, Any]) -> Iterator[dict]:
    path = state["path"]
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        yield from _iter_file(path)
        return

    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        yield {"path": str(file_path)}


class LocalFilesSource:
    default_task = types.Task.AUDIO_CODEC

    def prepare(self, spec: types.Spec, cache: CacheManifest) -> dict[str, Any]:
        return prepare_local(spec, cache)

    def iter_samples(self, state: dict[str, Any]) -> Iterator[dict]:
        yield from iter_local(state)

    def _iter_file(self, path: Path) -> Iterator[dict]:
        yield from _iter_file(path)


def _iter_file(path: Path) -> Iterator[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    yield {"path": str(path)}
