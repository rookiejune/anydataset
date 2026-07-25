"""Cost-aware batching for map-style dataset inputs."""

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data import DataLoader as TorchDataLoader

from .abc import MapStyleABC


@dataclass(frozen=True)
class _Record:
    index: int
    cost: int


@dataclass(frozen=True)
class _Plan:
    records: tuple[_Record, ...]
    cost: int


class _BatchSampler(Sampler[list[int]]):
    """Plan memory-bounded batches and balance their compute across ranks."""

    def __init__(
        self,
        dataset: MapStyleABC,
        *,
        cost_fn: Callable[[int], int],
        max_batch_memory: int,
        sampler: Sampler[int] | None,
        shuffle: bool,
        seed: int,
        epoch: int,
        planning_window: int = 256,
        max_batch_samples: int | None = None,
        drop_distributed_tail: bool = True,
    ) -> None:
        if not isinstance(dataset, MapStyleABC):
            raise TypeError("dataset must be a MapStyleABC.")
        if not callable(cost_fn):
            raise TypeError("cost_fn must be callable.")
        self.dataset = dataset
        self.cost_fn = cost_fn
        self.max_batch_memory = _positive_int("max_batch_memory", max_batch_memory)
        self.sampler = sampler
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool.")
        self.shuffle = shuffle
        self.seed = _int("seed", seed)
        self.epoch = _non_negative_int("epoch", epoch)
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
        plans: list[_Plan] = []
        for indexes in self._index_groups():
            records: list[_Record] = []
            for index in indexes:
                resolved = operator.index(index)
                records.append(
                    _Record(
                        index=resolved,
                        cost=_sample_cost(self.cost_fn, resolved),
                    )
                )
            if records:
                plans.extend(
                    _plans(
                        records,
                        max_batch_memory=self.max_batch_memory,
                        planning_window=self.planning_window,
                        max_batch_samples=self.max_batch_samples,
                    )
                )
        plans = _drop_distributed_tail(plans, drop_tail=self.drop_distributed_tail)
        yield from ([record.index for record in plan.records] for plan in plans)

    def __len__(self) -> int:
        raise TypeError("dataloader batch count is unavailable before planning.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _non_negative_int("epoch", epoch)
        set_epoch = getattr(self.sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def _index_groups(self) -> Iterator[Sequence[int]]:
        if self.sampler is not None:
            yield tuple(operator.index(index) for index in self.sampler)
            return
        num_replicas, rank = _rank()
        yield from self.dataset._shuffle(
            shuffle=self.shuffle,
            seed=self.seed,
            epoch=self.epoch,
            num_replicas=num_replicas,
            rank=rank,
        )


class _DataLoader(TorchDataLoader):
    """Materialize batches planned from index-level sample costs."""

    def __init__(
        self,
        dataset: MapStyleABC,
        *,
        cost_fn: Callable[[int], int],
        max_batch_memory: int,
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
        seed: int = 0,
        epoch: int = 0,
        planning_window: int = 256,
        max_batch_samples: int | None = None,
        drop_distributed_tail: bool = True,
        **loader_kwargs: Any,
    ) -> None:
        conflicts = {"batch_sampler", "batch_size", "drop_last"} & loader_kwargs.keys()
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"dataloader owns loader kwargs: {names}.")
        if sampler is not None and shuffle:
            raise ValueError("sampler and shuffle define overlapping sample order.")
        batch_sampler = _BatchSampler(
            dataset,
            cost_fn=cost_fn,
            max_batch_memory=max_batch_memory,
            sampler=sampler,
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            planning_window=planning_window,
            max_batch_samples=max_batch_samples,
            drop_distributed_tail=drop_distributed_tail,
        )
        super().__init__(dataset, batch_sampler=batch_sampler, **loader_kwargs)

    def __len__(self) -> int:
        raise TypeError("dataloader batch count is unavailable before planning.")

    def set_epoch(self, epoch: int) -> None:
        """Advance the underlying distributed sampler before the next iteration."""
        self.batch_sampler.set_epoch(epoch)


def _plans(
    records: list[_Record],
    *,
    max_batch_memory: int,
    planning_window: int,
    max_batch_samples: int | None,
) -> list[_Plan]:
    pending = list(records)
    plans: list[_Plan] = []
    while pending:
        selected = [pending.pop(0)]
        cost = _batch_cost(selected)
        if cost > max_batch_memory:
            raise ValueError(
                "A sample exceeds max_batch_memory: "
                f"index={selected[0].index} memory={cost} "
                f"budget={max_batch_memory}."
            )
        while max_batch_samples is None or len(selected) < max_batch_samples:
            candidate = _candidate(
                pending[:planning_window],
                selected,
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
    max_batch_memory: int,
) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for offset, record in enumerate(candidates):
        cost = _batch_cost([*selected, record])
        if cost > max_batch_memory:
            continue
        if best is None or cost > best[1]:
            best = offset, cost
    if best is None:
        return None
    return best


def _batch_cost(records: Sequence[_Record]) -> int:
    return sum(record.cost for record in records)


def _drop_distributed_tail(plans: list[_Plan], *, drop_tail: bool) -> list[_Plan]:
    if not dist.is_available() or not dist.is_initialized():
        return plans
    world_size = dist.get_world_size()
    counts: list[int] = [0 for _ in range(world_size)]
    dist.all_gather_object(counts, len(plans))
    kept = min(counts)
    if any(count != kept for count in counts):
        if not drop_tail:
            raise RuntimeError(
                "dataloader cannot keep equal rank-local batch counts."
            )
        dropped = len(plans) - kept
        if dropped:
            warnings.warn(
                f"dataloader dropped {dropped} rank-local final batches "
                "for equal DDP steps.",
                RuntimeWarning,
                stacklevel=2,
            )
    return plans[:kept]


def _rank() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        return 1, 0
    return dist.get_world_size(), dist.get_rank()


def _int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    return operator.index(value)


def _non_negative_int(name: str, value: int) -> int:
    resolved = _int(name, value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative.")
    return resolved


def _positive_int(name: str, value: int) -> int:
    resolved = _int(name, value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _sample_cost(cost_fn: Callable[[int], int], index: int) -> int:
    cost = operator.index(cost_fn(index))
    if cost <= 0:
        raise ValueError(f"cost_fn must return a positive integer: index={index}.")
    return cost
