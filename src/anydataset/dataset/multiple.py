from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

from torch.utils.data import IterableDataset

from ..types.item import Sample
from .abc import _validate_shard


class IterationStrategy(Protocol):
    def iter(self, datasets: Sequence[Iterable[Sample]]) -> Iterator[Sample]: ...


@dataclass(frozen=True)
class SequentialStrategy:
    def iter(self, datasets: Sequence[Iterable[Sample]]) -> Iterator[Sample]:
        for dataset in datasets:
            yield from dataset


@dataclass(frozen=True)
class RoundRobinStrategy:
    def iter(self, datasets: Sequence[Iterable[Sample]]) -> Iterator[Sample]:
        active = [iter(dataset) for dataset in datasets]
        while active:
            remaining = []
            for iterator in active:
                try:
                    yield next(iterator)
                except StopIteration:
                    continue
                remaining.append(iterator)
            active = remaining


@dataclass(frozen=True)
class WeightedRandomStrategy:
    weights: Sequence[float] | None = None
    seed: int | None = None

    def iter(self, datasets: Sequence[Iterable[Sample]]) -> Iterator[Sample]:
        weights = self._weights(len(datasets))
        active = [
            (iter(dataset), weight)
            for dataset, weight in zip(datasets, weights, strict=True)
            if weight > 0
        ]
        rng = random.Random(self.seed)

        while active:
            index = rng.choices(
                range(len(active)),
                weights=[weight for _, weight in active],
                k=1,
            )[0]
            iterator, _ = active[index]
            try:
                yield next(iterator)
            except StopIteration:
                del active[index]

    def _weights(self, count: int) -> tuple[float, ...]:
        if self.weights is None:
            if count == 0:
                return ()
            return tuple(1.0 for _ in range(count))

        weights = tuple(float(weight) for weight in self.weights)
        if len(weights) != count:
            raise ValueError("weights length must match datasets length.")
        if any(weight < 0 for weight in weights):
            raise ValueError("weights must be non-negative.")
        if not any(weight > 0 for weight in weights):
            raise ValueError("At least one dataset weight must be positive.")
        return weights


@dataclass(frozen=True)
class MultipleAnyDataset(IterableDataset):
    datasets: Sequence[Iterable[Sample]]
    strategy: IterationStrategy = field(default_factory=SequentialStrategy)

    def __post_init__(self) -> None:
        datasets = tuple(self.datasets)
        if not datasets:
            raise ValueError("MultipleAnyDataset requires at least one dataset.")
        object.__setattr__(self, "datasets", datasets)

    def __iter__(self) -> Iterator[Sample]:
        yield from self.iter_shard(num_shards=1, shard_id=0)

    def iter_shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]:
        _validate_shard(num_shards, shard_id)
        datasets = tuple(
            _iter_shard(dataset, num_shards, shard_id) for dataset in self.datasets
        )
        yield from self.strategy.iter(datasets)

    def shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]:
        yield from self.iter_shard(num_shards, shard_id)


@runtime_checkable
class _Shardable(Protocol):
    def iter_shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]: ...


def _iter_shard(
    dataset: Iterable[Sample],
    num_shards: int,
    shard_id: int,
) -> Iterator[Sample]:
    if isinstance(dataset, _Shardable):
        yield from dataset.iter_shard(num_shards, shard_id)
        return

    for index, sample in enumerate(dataset):
        if index % num_shards == shard_id:
            yield sample
