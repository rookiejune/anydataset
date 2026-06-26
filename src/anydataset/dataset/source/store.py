from __future__ import annotations

from pathlib import Path

from ...store.reader import StoreDataset, read_store_dataset
from ...types import Spec


class StoreSource:
    def prepare(self, spec: Spec, _cache_path: Path) -> StoreDataset:
        return read_store_dataset(spec.path)
