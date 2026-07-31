"""Cost-aware batching for map-style dataset inputs."""

from __future__ import annotations

import operator
import warnings
from bisect import bisect_left, bisect_right, insort
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data import DataLoader as TorchDataLoader

from .abc import MapStyleABC


_DISTRIBUTED_PLAN_WINDOW = 128
_CostFn = Callable[[Any], int]
_Costs = None | Sequence[int]


class _CallableCosts(Sequence[int]):
    def __init__(self, dataset: MapStyleABC, cost_fn: _CostFn) -> None:
        self.dataset = dataset
        self.cost_fn = cost_fn
        self._cache: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> int:
        resolved = operator.index(index)
        if resolved < 0:
            resolved += len(self)
        if resolved < 0 or resolved >= len(self):
            raise IndexError("cost index out of range.")
        cached = self._cache.get(resolved)
        if cached is None:
            row = self.dataset.cost_row(resolved)
            cached = operator.index(self.cost_fn(row))
            self._cache[resolved] = cached
        return cached


@dataclass(frozen=True)
class _Record:
    index: int
    cost: int


@dataclass(frozen=True)
class _Plan:
    records: tuple[_Record, ...]
    cost: int


class _Pending:
    def __init__(self, records: Iterable[_Record]) -> None:
        self._source = iter(records)
        self._order: OrderedDict[int, _Record] = OrderedDict()
        self._costs: list[tuple[int, int, int]] = []
        self._arrival = 0

    def __bool__(self) -> bool:
        return bool(self._order)

    def fill(self, limit: int) -> None:
        while len(self._order) < limit:
            try:
                record = next(self._source)
            except StopIteration:
                return
            arrival = self._arrival
            self._arrival += 1
            self._order[arrival] = record
            insort(self._costs, (record.cost, -arrival, arrival))

    def pop_first(self) -> _Record:
        arrival, record = self._order.popitem(last=False)
        key = (record.cost, -arrival, arrival)
        offset = bisect_left(self._costs, key)
        if offset >= len(self._costs) or self._costs[offset] != key:
            raise RuntimeError("batch planner pending cost index is inconsistent.")
        self._costs.pop(offset)
        return record

    def pop_fitting(self, budget: int) -> _Record | None:
        offset = bisect_right(self._costs, (budget, self._arrival, self._arrival)) - 1
        if offset < 0:
            return None
        _cost, _reverse_arrival, arrival = self._costs.pop(offset)
        return self._order.pop(arrival)


class _DatasetSampler(Sampler[int]):
    """Expose dataset-owned rank-local ordering through PyTorch's sampler contract."""

    def __init__(self, batch_sampler: _BatchSampler) -> None:
        self.batch_sampler = batch_sampler

    def __iter__(self) -> Iterator[int]:
        for indexes in self.batch_sampler._dataset_index_groups():
            yield from indexes

    def set_epoch(self, epoch: int) -> None:
        self.batch_sampler.epoch = _non_negative_int("epoch", epoch)


class _BatchSampler(Sampler[list[int]]):
    """Plan memory-bounded batches and balance their compute across ranks."""

    def __init__(
        self,
        dataset: MapStyleABC,
        *,
        costs: None | Iterable[int] | _CostFn,
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
        self.dataset = dataset
        self.costs = _costs(costs, dataset=dataset, sample_count=len(dataset))
        self.max_batch_memory = _positive_int("max_batch_memory", max_batch_memory)
        self._source_sampler = sampler
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
        self.sampler: Sampler[int] = (
            sampler if sampler is not None else _DatasetSampler(self)
        )

    def __iter__(self) -> Iterator[list[int]]:
        plans = self._iter_plans()
        for plan in _synchronized_plans(
            plans,
            drop_tail=self.drop_distributed_tail,
        ):
            yield [record.index for record in plan.records]

    def _iter_plans(self) -> Iterator[_Plan]:
        for indexes in self._index_groups():
            records = (_record(self.costs, index) for index in indexes)
            yield from _plans(
                records,
                max_batch_memory=self.max_batch_memory,
                planning_window=self.planning_window,
                max_batch_samples=self.max_batch_samples,
            )

    def __len__(self) -> int:
        raise TypeError("dataloader batch count is unavailable before planning.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _non_negative_int("epoch", epoch)
        set_epoch = getattr(self._source_sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def _index_groups(self) -> Iterator[Iterable[int]]:
        if self._source_sampler is not None:
            yield self._source_sampler
            return
        yield from self._dataset_index_groups()

    def _dataset_index_groups(self) -> Iterator[Sequence[int]]:
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
        costs: None | Iterable[int] | _CostFn,
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
            costs=costs,
            max_batch_memory=max_batch_memory,
            sampler=sampler,
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            planning_window=planning_window,
            max_batch_samples=max_batch_samples,
            drop_distributed_tail=drop_distributed_tail,
        )
        self._batch_sampler = batch_sampler
        super().__init__(dataset, batch_sampler=batch_sampler, **loader_kwargs)

    def __len__(self) -> int:
        raise TypeError("dataloader batch count is unavailable before planning.")

    def set_epoch(self, epoch: int) -> None:
        """Advance the underlying distributed sampler before the next iteration."""
        self._batch_sampler.set_epoch(epoch)


def _plans(
    records: Iterable[_Record],
    *,
    max_batch_memory: int,
    planning_window: int,
    max_batch_samples: int | None,
) -> Iterator[_Plan]:
    pending = _Pending(records)
    while True:
        pending.fill(1)
        if not pending:
            return
        selected = [pending.pop_first()]
        cost = selected[0].cost
        if cost > max_batch_memory:
            raise ValueError(
                "A sample exceeds max_batch_memory: "
                f"index={selected[0].index} memory={cost} "
                f"budget={max_batch_memory}."
            )
        while max_batch_samples is None or len(selected) < max_batch_samples:
            pending.fill(planning_window)
            candidate = pending.pop_fitting(max_batch_memory - cost)
            if candidate is None:
                break
            selected.append(candidate)
            cost += candidate.cost
        yield _Plan(records=tuple(selected), cost=cost)


def _synchronized_plans(
    plans: Iterable[_Plan],
    *,
    drop_tail: bool,
) -> Iterator[_Plan]:
    if not dist.is_available() or not dist.is_initialized():
        yield from plans
        return

    source = iter(plans)
    world_size = dist.get_world_size()
    while True:
        local = list(islice(source, _DISTRIBUTED_PLAN_WINDOW))
        counts = _plan_counts(len(local), world_size)
        kept = min(counts)
        if any(count != kept for count in counts):
            if not drop_tail:
                raise RuntimeError(
                    "dataloader cannot keep equal rank-local batch counts."
                )
            if len(local) > kept:
                warnings.warn(
                    "dataloader dropped rank-local final batches for equal DDP steps.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        yield from local[:kept]
        if kept < _DISTRIBUTED_PLAN_WINDOW:
            return


def _plan_counts(local_count: int, world_size: int) -> tuple[int, ...]:
    backend = str(dist.get_backend()).lower()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    local = torch.tensor([local_count], dtype=torch.int64, device=device)
    gathered = torch.empty(world_size, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local)
    return tuple(int(value) for value in gathered.cpu().tolist())


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


def _costs(
    costs: None | Iterable[int] | _CostFn,
    *,
    dataset: MapStyleABC,
    sample_count: int,
) -> _Costs:
    if isinstance(costs, bool) or isinstance(costs, int):
        raise TypeError(
            "costs must be None, an iterable of integers, or a callable."
        )
    if costs is None:
        return None
    if callable(costs):
        return _CallableCosts(dataset, costs)
    if isinstance(costs, (str, bytes, bytearray)) or not isinstance(costs, Iterable):
        raise TypeError(
            "costs must be None, an iterable of integers, or a callable."
        )
    if not isinstance(costs, Sequence):
        costs = tuple(costs)
    if len(costs) != sample_count:
        raise ValueError("costs and dataset must have equal length.")
    return costs


def _record(costs: _Costs, index: int) -> _Record:
    resolved = operator.index(index)
    return _Record(index=resolved, cost=_sample_cost(costs, resolved))


def _sample_cost(costs: _Costs, index: int) -> int:
    cost = 1 if costs is None else operator.index(costs[index])
    if cost <= 0:
        raise ValueError(f"sample cost must be a positive integer: index={index}.")
    return cost
