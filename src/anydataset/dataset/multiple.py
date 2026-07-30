from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import isfinite
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

        sampler = _FenwickWeights(active_weights)
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


class _FenwickWeights:
    def __init__(self, weights: Sequence[float]) -> None:
        scale = max(weights)
        self._weights = [weight / scale for weight in weights]
        self._tree = [0.0] * (len(self._weights) + 1)
        for index, weight in enumerate(self._weights, start=1):
            self._add(index, weight)
        self._total = sum(self._weights)

    def choose(self, rng: random.Random) -> int:
        target = rng.random() * self._total
        index = 0
        step = 1 << (len(self._weights).bit_length() - 1)
        while step:
            candidate = index + step
            if candidate < len(self._tree) and self._tree[candidate] <= target:
                index = candidate
                target -= self._tree[candidate]
            step >>= 1
        return index

    def remove(self, index: int) -> None:
        weight = self._weights[index]
        if weight == 0.0:
            return
        self._weights[index] = 0.0
        self._add(index + 1, -weight)
        self._total -= weight

    def _add(self, index: int, value: float) -> None:
        while index < len(self._tree):
            self._tree[index] += value
            index += index & -index


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
