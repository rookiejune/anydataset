from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset, IterableDataset

from .._sharding import Shard, runtime_shard, validate_shard
from ..types import Preset, Source, Spec, source_key
from ..utils import resolve_dataset

if TYPE_CHECKING:
    from ..cache import CacheManager
    from ..types.item import Sample, Schema, Transforms
    from .source import DatasetSource


class _Base(ABC):
    def __init__(
        self,
        spec: str | Preset | Spec,
        parse_fn: Callable[[Any], Sample] | None = None,
        cache_root: str | Path | None = None,
        transforms: Transforms | None = None,
    ) -> None:
        self.spec = resolve_dataset(spec)
        self._cache_manager = None
        if cache_root is not None:
            from ..cache import CacheManager

            self._cache_manager = CacheManager(cache_root)
        self._dataset = None
        self._source: DatasetSource | None = None
        self.parse_fn = parse_fn or _identity_sample
        self.transforms = None if transforms is None else dict(transforms)

    def prepare(self) -> Any:
        if self._dataset is not None:
            return self._dataset

        cache = self.cache_manager.prepare(self.spec)
        self._dataset = self.source.prepare(self.spec, cache.cache_path)
        return self._dataset

    @property
    def cache_manager(self) -> CacheManager:
        if self._cache_manager is None:
            from ..cache import CacheManager

            self._cache_manager = CacheManager()
        return self._cache_manager

    @property
    def dataset(self) -> Any:
        return self.prepare()

    @property
    def source(self) -> DatasetSource:
        if self._source is None:
            from .source import for_source

            self._source = for_source(self.spec.source)
        return self._source

    def __getstate__(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "cache_root": None if self._cache_manager is None else self._cache_manager.root,
            "parse_fn": self.parse_fn,
            "transforms": self.transforms,
            "source": self.source,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.spec = state["spec"]
        self._cache_manager = None
        cache_root = state["cache_root"]
        if cache_root is not None:
            from ..cache import CacheManager

            self._cache_manager = CacheManager(cache_root)
        self._dataset = None
        self._source = state["source"]
        self.parse_fn = state["parse_fn"]
        transforms = state["transforms"]
        self.transforms = None if transforms is None else dict(transforms)

    def __iter__(self) -> Iterator[Sample]:
        shard = runtime_shard()
        yield from self.iter_runtime_shard(shard)

    def iter_runtime_shard(self, shard: Shard) -> Iterator[Sample]:
        yield from self.iter_shard(shard.count, shard.index)

    def transform_sample(self, sample: Sample) -> Sample:
        if self.transforms is None:
            return sample
        transformed = dict(sample)
        for reference, transform in self.transforms.items():
            transformed[reference] = transform(sample[reference])
        return transformed

    @abstractmethod
    def iter_shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]:
        raise NotImplementedError

    @staticmethod
    def resolve_sample(sample: Sample, schema: Schema) -> Sample:
        return {
            reference: sample[reference].select_by(requirement)
            for reference, requirement in schema.items()
        }


class IterableAnyDataset(_Base, IterableDataset):
    def iter_rows(self) -> Iterator[Any]:
        yield from self.dataset

    def iter_shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]:
        validate_shard(num_shards, shard_id)
        for row in self.iter_shard_rows(num_shards, shard_id):
            yield self.transform_sample(self.parse_fn(row))

    def iter_shard_rows(self, num_shards: int, shard_id: int) -> Iterator[Any]:
        validate_shard(num_shards, shard_id)
        dataset = self.dataset
        shard = getattr(dataset, "shard", None)
        if shard is not None:
            yield from shard(num_shards=num_shards, index=shard_id)
            return

        yield from _iter_modulo(self.iter_rows(), num_shards, shard_id)


class SampleDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Sample:
        raise NotImplementedError


class AnyDataset(_Base, SampleDataset):
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        return self.transform_sample(self.parse_fn(self.dataset[index]))

    def merge(self, dataset: Iterable[Sample]) -> AnyDataset:
        if source_key(self.spec.source) != Source.STORE.value:
            raise TypeError("merge requires a store dataset.")
        merge = getattr(self.dataset, "merge", None)
        if not callable(merge):
            raise TypeError("merge requires a store dataset.")
        self._dataset = merge(dataset)
        return self

    def iter_shard(self, num_shards: int, shard_id: int):
        validate_shard(num_shards, shard_id)
        iter_indexed = getattr(self.dataset, "iter_indexed_shard", None)
        if callable(iter_indexed):
            for _index, row in iter_indexed(num_shards, shard_id):
                yield self.transform_sample(self.parse_fn(row))
            return

        for index in range(shard_id, len(self), num_shards):
            yield self[index]

    def iter_indexed_range(self, start: int, stop: int):
        if start < 0 or stop < start or stop > len(self):
            raise ValueError("range must satisfy 0 <= start <= stop <= len(dataset).")

        dataset = self.dataset
        iter_indexed = getattr(dataset, "iter_indexed_range", None)
        if callable(iter_indexed):
            for index, row in iter_indexed(start, stop):
                yield index, self.transform_sample(self.parse_fn(row))
            return

        for index in range(start, stop):
            yield index, self[index]

    def iter_indexed_shard(self, num_shards: int, shard_id: int):
        validate_shard(num_shards, shard_id)
        dataset = self.dataset
        iter_indexed = getattr(dataset, "iter_indexed_shard", None)
        if callable(iter_indexed):
            for index, row in iter_indexed(num_shards, shard_id):
                yield index, self.transform_sample(self.parse_fn(row))
            return

        for index in range(shard_id, len(self), num_shards):
            yield index, self[index]

    def iter_runtime_shard(self, shard: Shard):
        usable = len(self) // shard.rank_count * shard.rank_count
        if shard.flat_count > 1:
            for index, sample in self.iter_indexed_shard(shard.flat_count, shard.flat_index):
                if index < usable:
                    yield sample
            return

        for index, sample in self.iter_indexed_range(0, usable):
            yield sample


def _identity_sample(row: Any) -> Sample:
    return row


def _iter_modulo(
    rows: Iterator[Any],
    num_shards: int,
    shard_id: int,
) -> Iterator[Any]:
    for index, row in enumerate(rows):
        if index % num_shards == shard_id:
            yield row
