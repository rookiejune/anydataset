from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ...store.reader import StoreDataset, read_store_dataset
from ...types import Spec
from ...types.item import Modality, Role, Sample, View
from .protocol import validate_load_options


class StoreSource:
    def __init__(
        self,
        views: tuple[tuple[Role, Modality, View], ...] | None = None,
    ) -> None:
        if views is not None and not isinstance(views, tuple):
            raise TypeError("store views must be a tuple or None.")
        self.views = views

    def prepare(self, spec: Spec, cache_path: Path) -> StoreDataset:
        _ = cache_path
        validate_load_options(spec, (), source="store")
        dataset = read_store_dataset(spec.path, views=self.views)
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
