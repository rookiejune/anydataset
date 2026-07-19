from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ...store.reader import StoreDataset, read_store_dataset
from ...types import Spec
from ...types.item import Sample
from .protocol import validate_load_options


class StoreSource:
    def prepare(self, spec: Spec, _cache_path: Path) -> StoreDataset:
        validate_load_options(spec, (), source="store")
        dataset = read_store_dataset(spec.path)
        if spec.split is not None and dataset.manifest.split != spec.split:
            raise ValueError(
                f"Store dataset split {dataset.manifest.split!r} does not match "
                f"requested split {spec.split!r}."
            )
        return dataset

    def iter_indexed_shard(
        self,
        dataset: StoreDataset,
        *,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Sample]]:
        yield from dataset.iter_indexed_shard(num_shards, shard_id)
