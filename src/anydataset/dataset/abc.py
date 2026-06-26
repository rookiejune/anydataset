from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .. import types
from ..types import Spec

if TYPE_CHECKING:
    from ..cache import CacheManager
    from .source import DatasetSource
    from ..types.item import Reference, Sample, Schema


type Ref = types.Reference
type Sample = types.Sample
type Schema = types.Schema


def _validate_shard(num_shards: int, shard_id: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards.")


class _Base(ABC):
    def __init__(
        self,
        spec: Spec,
        parse_fn: Callable[[Any], Sample] | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        self.spec = spec
        self._cache_manager = None
        if cache_root is not None:
            from ..cache import CacheManager

            self._cache_manager = CacheManager(cache_root)
        self._dataset = None
        self._source: DatasetSource | None = None
        self.parse_fn = parse_fn or _identity_sample

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

    def __iter__(self) -> Iterator[Sample]:
        yield from self.iter_shard(num_shards=1, shard_id=0)

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
    def __init__(
        self,
        spec: Spec,
        parse_fn: Callable[[Any], Sample] | None = None,
        cache_root: str | Path | None = None,
        *,
        num_shards: int = 1,
        shard_id: int = 0,
    ) -> None:
        _validate_shard(num_shards, shard_id)
        super().__init__(spec, parse_fn=parse_fn, cache_root=cache_root)
        self.num_shards = num_shards
        self.shard_id = shard_id

    def __iter__(self) -> Iterator[Sample]:
        num_shards, shard_id = _worker_shard(self.num_shards, self.shard_id)
        yield from self.iter_shard(num_shards, shard_id)

    def iter_rows(self) -> Iterator[Any]:
        yield from self.dataset

    def iter_shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]:
        _validate_shard(num_shards, shard_id)
        for row in self.iter_shard_rows(num_shards, shard_id):
            yield self.parse_fn(row)

    def iter_shard_rows(self, num_shards: int, shard_id: int) -> Iterator[Any]:
        _validate_shard(num_shards, shard_id)
        dataset = self.dataset
        shard = getattr(dataset, "shard", None)
        if shard is not None:
            yield from shard(num_shards=num_shards, index=shard_id)
            return

        yield from _iter_modulo(self.iter_rows(), num_shards, shard_id)


class AnyDataset(_Base, Dataset):
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        return self.parse_fn(self.dataset[index])

    def iter_shard(self, num_shards: int, shard_id: int):
        _validate_shard(num_shards, shard_id)
        for index in range(shard_id, len(self), num_shards):
            yield self[index]


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


def _worker_shard(num_shards: int, shard_id: int) -> tuple[int, int]:
    worker = get_worker_info()
    if worker is None:
        return num_shards, shard_id
    return (
        num_shards * worker.num_workers,
        shard_id * worker.num_workers + worker.id,
    )
