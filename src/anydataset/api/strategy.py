from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Iterator, Mapping, Protocol, Sequence

from anydataset.samples import Sample


class SampleSource(Protocol):
    @property
    def name(self) -> str:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Sample]:
        raise NotImplementedError


class IterationStrategy(ABC):
    @abstractmethod
    def iter(self, datasets: Sequence[SampleSource]) -> Iterator[Sample]:
        raise NotImplementedError


@dataclass(frozen=True)
class SequentialStrategy(IterationStrategy):
    def iter(self, datasets: Sequence[SampleSource]) -> Iterator[Sample]:
        for dataset in datasets:
            yield from dataset


@dataclass(frozen=True)
class RoundRobinStrategy(IterationStrategy):
    def iter(self, datasets: Sequence[SampleSource]) -> Iterator[Sample]:
        active = [iter(dataset) for dataset in datasets]
        while active:
            remaining = []
            for iterator in active:
                try:
                    yield next(iterator)
                    remaining.append(iterator)
                except StopIteration:
                    continue
            active = remaining


@dataclass(frozen=True)
class WeightedRandomStrategy(IterationStrategy):
    weights: Mapping[str, float] | None = None
    seed: int | None = None

    def iter(self, datasets: Sequence[SampleSource]) -> Iterator[Sample]:
        rng = random.Random(self.seed)
        active = [
            _WeightedIterator(
                name=dataset.name,
                iterator=iter(dataset),
                weight=self._weight_for(dataset),
            )
            for dataset in datasets
        ]
        active = [item for item in active if item.weight > 0]
        if not active:
            raise ValueError("At least one dataset must have a positive weight.")

        while active:
            weights = [item.weight for item in active]
            selected = rng.choices(range(len(active)), weights=weights, k=1)[0]
            item = active[selected]
            try:
                yield next(item.iterator)
            except StopIteration:
                active.pop(selected)

    def _weight_for(self, dataset: SampleSource) -> float:
        weight = 1.0 if self.weights is None else self.weights.get(dataset.name, 1.0)
        if weight < 0:
            raise ValueError(f"Dataset {dataset.name!r} has negative weight.")
        return weight


@dataclass(frozen=True)
class _WeightedIterator:
    name: str
    iterator: Iterator[Sample]
    weight: float
