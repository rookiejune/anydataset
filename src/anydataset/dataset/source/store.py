from __future__ import annotations

from pathlib import Path

from ...store.reader import StoreDataset, read_store_dataset
from ...types import Spec


class StoreSource:
    def prepare(self, spec: Spec, _cache_path: Path) -> StoreDataset:
        dataset = read_store_dataset(spec.path)
        if spec.split is not None and dataset.manifest.split != spec.split:
            raise ValueError(
                f"Store dataset split {dataset.manifest.split!r} does not match "
                f"requested split {spec.split!r}."
            )
        return dataset
