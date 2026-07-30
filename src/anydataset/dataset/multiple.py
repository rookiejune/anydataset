from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import isfinite, nextafter
from typing import Iterator, Protocol

from torch.utils.data import IterableDataset

from .._compat import strict_zip
from .._sharding import runtime_shard, validate_shard
from ..types.item import Sample
from .abc import IterableAnyDataset, MapStyleABC


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
        active: list[Iterator[Sample]] = []
        active_weights: list[float] = []
        for dataset, weight in strict_zip(datasets, weights):
            if weight > 0:
                active.append(iter(dataset))
                active_weights.append(weight)
        rng = random.Random(self.seed)

        if len(active) <= 64:
            yield from self._iter_choices(active, active_weights, rng)
            return

        sampler = _WeightTree(active_weights)
        remaining = len(active)
        while remaining:
            index = sampler.choose(rng)
            iterator = active[index]
            try:
                yield next(iterator)
            except StopIteration:
                sampler.remove(index)
                remaining -= 1

    def _iter_choices(
        self,
        active: list[Iterator[Sample]],
        active_weights: list[float],
        rng: random.Random,
    ) -> Iterator[Sample]:
        cumulative_weights = _cumulative_weights(active_weights)
        while active:
            index = rng.choices(
                range(len(active)),
                cum_weights=cumulative_weights,
                k=1,
            )[0]
            iterator = active[index]
            try:
                yield next(iterator)
            except StopIteration:
                del active[index]
                del active_weights[index]
                cumulative_weights = _cumulative_weights(active_weights)

    def _weights(self, count: int) -> tuple[float, ...]:
        if self.weights is None:
            if count == 0:
                return ()
            return tuple(1.0 for _ in range(count))

        weights = tuple(float(weight) for weight in self.weights)
        if len(weights) != count:
            raise ValueError("weights length must match datasets length.")
        if any(not isfinite(weight) for weight in weights):
            raise ValueError("weights must be finite.")
        if any(weight < 0 for weight in weights):
            raise ValueError("weights must be non-negative.")
        if not any(weight > 0 for weight in weights):
            raise ValueError("At least one dataset weight must be positive.")
        return weights


def _cumulative_weights(weights: Sequence[float]) -> list[float]:
    if not weights:
        return []
    scale = max(weights)
    total = 0.0
    output = []
    for weight in weights:
        total += weight / scale
        output.append(total)
    return output


class _WeightTree:
    def __init__(self, weights: Sequence[float]) -> None:
        self._weights = list(weights)
        self._active = sum(weight > 0.0 for weight in self._weights)
        self._size = 1 << (len(self._weights) - 1).bit_length()
        self._tree = [0.0] * (self._size * 2)
        self._rebuild()

    def choose(self, rng: random.Random) -> int:
        total = self._tree[1]
        if total <= 0.0:
            raise RuntimeError("weighted sampler has no active weight.")
        target = min(rng.random() * total, nextafter(total, 0.0))
        node = 1
        while node < self._size:
            left = node * 2
            left_weight = self._tree[left]
            if target < left_weight:
                node = left
            else:
                target = min(
                    target - left_weight,
                    nextafter(self._tree[left + 1], 0.0),
                )
                node = left + 1
        return node - self._size

    def remove(self, index: int) -> None:
        weight = self._weights[index]
        if weight == 0.0:
            return
        self._weights[index] = 0.0
        self._active -= 1
        node = self._size + index
        self._tree[node] = 0.0
        while node > 1:
            node //= 2
            self._tree[node] = self._tree[node * 2] + self._tree[node * 2 + 1]
        if self._active > 0 and self._tree[1] == 0.0:
            self._rebuild()

    def _rebuild(self) -> None:
        scale = max(self._weights)
        for index in range(self._size):
            weight = self._weights[index] if index < len(self._weights) else 0.0
            self._tree[self._size + index] = weight / scale if scale > 0.0 else 0.0
        for node in range(self._size - 1, 0, -1):
            self._tree[node] = self._tree[node * 2] + self._tree[node * 2 + 1]


@dataclass
class MultipleAnyDataset(IterableDataset):
    datasets: Sequence[MapStyleABC | IterableAnyDataset]
    strategy: IterationStrategy = field(default_factory=SequentialStrategy)

    def __post_init__(self) -> None:
        datasets = tuple(self.datasets)
        if not datasets:
            raise ValueError("MultipleAnyDataset requires at least one dataset.")
        self.datasets = datasets

    def __iter__(self) -> Iterator[Sample]:
        shard = runtime_shard()
        datasets = tuple(dataset.iter_runtime_shard(shard) for dataset in self.datasets)
        yield from self.strategy.iter(datasets)

    def iter_shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]:
        validate_shard(num_shards, shard_id)
        datasets = tuple(
            dataset.iter_shard(num_shards, shard_id) for dataset in self.datasets
        )
        yield from self.strategy.iter(datasets)

    def shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]:
        yield from self.iter_shard(num_shards, shard_id)
