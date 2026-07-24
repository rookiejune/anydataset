"""Cost-aware batching for map-style AnyDataset inputs."""

from __future__ import annotations

import math
import operator
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any

import torch.distributed as dist
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

from .abc import AnyDataset


@dataclass(frozen=True)
class BatchCost:
    """Memory budget, compute estimate, and avoidable work for one batch."""

    memory: int
    compute: float
    waste: float

    def __post_init__(self) -> None:
        memory = operator.index(self.memory)
        if memory <= 0:
            raise ValueError("BatchCost.memory must be positive.")
        if not math.isfinite(self.compute) or self.compute <= 0:
            raise ValueError("BatchCost.compute must be finite and positive.")
        if not math.isfinite(self.waste) or self.waste < 0:
            raise ValueError("BatchCost.waste must be finite and non-negative.")


@dataclass(frozen=True)
class _Record:
    index: int
    cost: Any


@dataclass(frozen=True)
class _Plan:
    records: tuple[_Record, ...]
    cost: BatchCost


class CostBatchSampler(Sampler[list[int]]):
    """Plan memory-bounded batches and balance their compute across ranks."""

    def __init__(
        self,
        dataset: AnyDataset,
        *,
        batch_cost_fn: Callable[[Sequence[Any]], BatchCost],
        max_batch_memory: int,
        sampler: Sampler[int],
        planning_window: int = 256,
        max_batch_samples: int | None = None,
        drop_distributed_tail: bool = True,
    ) -> None:
        if not isinstance(dataset, AnyDataset):
            raise TypeError("dataset must be an AnyDataset.")
        if dataset.cost_fn is None:
            raise TypeError("dataset must define cost_fn.")
        if not callable(batch_cost_fn):
            raise TypeError("batch_cost_fn must be callable.")
        self.dataset = dataset
        self.batch_cost_fn = batch_cost_fn
        self.max_batch_memory = _positive_int("max_batch_memory", max_batch_memory)
        self.sampler = sampler
        self.planning_window = _positive_int("planning_window", planning_window)
        self.max_batch_samples = (
            None
            if max_batch_samples is None
            else _positive_int("max_batch_samples", max_batch_samples)
        )
        if not isinstance(drop_distributed_tail, bool):
            raise TypeError("drop_distributed_tail must be a bool.")
        self.drop_distributed_tail = drop_distributed_tail

    def __iter__(self) -> Iterator[list[int]]:
        local = [
            _Record(index=operator.index(index), cost=self.dataset.cost(index))
            for index in self.sampler
        ]
        records = _global_records(local)
        plans = _plans(
            records,
            batch_cost_fn=self.batch_cost_fn,
            max_batch_memory=self.max_batch_memory,
            planning_window=self.planning_window,
            max_batch_samples=self.max_batch_samples,
        )
        assigned = _assign(plans, drop_tail=self.drop_distributed_tail)
        yield from ([record.index for record in plan.records] for plan in assigned)

    def __len__(self) -> int:
        raise TypeError("CostBatchSampler batch count is unavailable before planning.")

    def set_epoch(self, epoch: int) -> None:
        set_epoch = getattr(self.sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)


class CostDataLoader(DataLoader[Any]):
    """Materialize batches planned from lightweight AnyDataset costs."""

    def __init__(
        self,
        dataset: AnyDataset,
        *,
        batch_cost_fn: Callable[[Sequence[Any]], BatchCost],
        max_batch_memory: int,
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
        planning_window: int = 256,
        max_batch_samples: int | None = None,
        drop_distributed_tail: bool = True,
        **loader_kwargs: Any,
    ) -> None:
        conflicts = {"batch_sampler", "batch_size", "drop_last"} & loader_kwargs.keys()
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"CostDataLoader owns loader kwargs: {names}.")
        if sampler is not None and shuffle:
            raise ValueError("sampler and shuffle define overlapping sample order.")
        source = _sampler(dataset, shuffle=shuffle) if sampler is None else sampler
        batch_sampler = CostBatchSampler(
            dataset,
            batch_cost_fn=batch_cost_fn,
            max_batch_memory=max_batch_memory,
            sampler=source,
            planning_window=planning_window,
            max_batch_samples=max_batch_samples,
            drop_distributed_tail=drop_distributed_tail,
        )
        super().__init__(dataset, batch_sampler=batch_sampler, **loader_kwargs)

    def __len__(self) -> int:
        raise TypeError("CostDataLoader batch count is unavailable before planning.")

    def set_epoch(self, epoch: int) -> None:
        """Advance the underlying distributed sampler before the next iteration."""
        self.batch_sampler.set_epoch(epoch)


def _sampler(dataset: AnyDataset, *, shuffle: bool) -> Sampler[int]:
    if dist.is_available() and dist.is_initialized():
        # Global cost planning must not receive DistributedSampler padding
        # duplicates; final batch tails are handled explicitly by _assign().
        return DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
    if shuffle:
        return RandomSampler(dataset)
    return SequentialSampler(dataset)


def _global_records(local: list[_Record]) -> list[_Record]:
    if not dist.is_available() or not dist.is_initialized():
        return local
    gathered: list[list[_Record]] = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    return [
        record
        for records in zip_longest(*gathered)
        for record in records
        if record is not None
    ]


def _plans(
    records: list[_Record],
    *,
    batch_cost_fn: Callable[[Sequence[Any]], BatchCost],
    max_batch_memory: int,
    planning_window: int,
    max_batch_samples: int | None,
) -> list[_Plan]:
    pending = list(records)
    plans: list[_Plan] = []
    while pending:
        selected = [pending.pop(0)]
        cost = _batch_cost(batch_cost_fn, selected)
        if cost.memory > max_batch_memory:
            raise ValueError(
                "A sample exceeds max_batch_memory: "
                f"index={selected[0].index} memory={cost.memory} "
                f"budget={max_batch_memory}."
            )
        while max_batch_samples is None or len(selected) < max_batch_samples:
            candidate = _candidate(
                pending[:planning_window],
                selected,
                batch_cost_fn=batch_cost_fn,
                max_batch_memory=max_batch_memory,
            )
            if candidate is None:
                break
            offset, cost = candidate
            selected.append(pending.pop(offset))
        plans.append(_Plan(records=tuple(selected), cost=cost))
    return plans


def _candidate(
    candidates: Sequence[_Record],
    selected: list[_Record],
    *,
    batch_cost_fn: Callable[[Sequence[Any]], BatchCost],
    max_batch_memory: int,
) -> tuple[int, BatchCost] | None:
    best: tuple[tuple[float, int], int, BatchCost] | None = None
    for offset, record in enumerate(candidates):
        cost = _batch_cost(batch_cost_fn, [*selected, record])
        if cost.memory > max_batch_memory:
            continue
        key = (cost.waste, -cost.memory)
        if best is None or key < best[0]:
            best = key, offset, cost
    if best is None:
        return None
    return best[1], best[2]


def _batch_cost(
    batch_cost_fn: Callable[[Sequence[Any]], BatchCost],
    records: Sequence[_Record],
) -> BatchCost:
    cost = batch_cost_fn([record.cost for record in records])
    if not isinstance(cost, BatchCost):
        raise TypeError("batch_cost_fn must return BatchCost.")
    return cost


def _assign(plans: list[_Plan], *, drop_tail: bool) -> list[_Plan]:
    if not dist.is_available() or not dist.is_initialized():
        return plans
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    tail = len(plans) % world_size
    if tail:
        if not drop_tail:
            raise RuntimeError(
                "CostDataLoader cannot assign the final batch tail equally across ranks."
            )
        warnings.warn(
            f"CostDataLoader dropped {tail} final batches for equal DDP steps.",
            RuntimeWarning,
            stacklevel=2,
        )
        plans = plans[:-tail]
    ordered = sorted(plans, key=lambda plan: plan.cost.compute, reverse=True)
    assigned: list[_Plan] = []
    for step in range(0, len(ordered), world_size):
        group = ordered[step : step + world_size]
        target = (rank + step // world_size) % world_size
        assigned.append(group[target])
    return assigned


def _positive_int(name: str, value: int) -> int:
    resolved = operator.index(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


__all__ = ["BatchCost", "CostBatchSampler", "CostDataLoader"]
