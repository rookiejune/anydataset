"""Cost-aware batching for map-style dataset inputs."""

from __future__ import annotations

import operator
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Optional, overload

from torch.utils.data import Sampler
from torch.utils.data import DataLoader as TorchDataLoader

from ._ddp import debug_plans_enabled, log_debug_plan, rank, synchronized_plans
from .abc import MapStyleABC


_CostFn = Callable[[Any], int]
_Costs = Optional[Sequence[int]]
_MAX_CALLABLE_COST_CACHE = 4096


class _CallableCosts(Sequence[int]):
    def __init__(self, dataset: MapStyleABC, cost_fn: _CostFn) -> None:
        self.dataset = dataset
        self.cost_fn = cost_fn
        self._cache: OrderedDict[int, int] = OrderedDict()

    def __len__(self) -> int:
        return len(self.dataset)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[int]: ...

    def __getitem__(self, index: int | slice) -> int | Sequence[int]:
        if isinstance(index, slice):
            return tuple(self[offset] for offset in range(*index.indices(len(self))))
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
            if len(self._cache) > _MAX_CALLABLE_COST_CACHE:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(resolved)
        return cached


@dataclass(frozen=True)
class _Record:
    index: int
    cost: int


@dataclass(frozen=True)
class _PendingRecord:
    arrival: int
    record: _Record


@dataclass(frozen=True)
class _Plan:
    records: tuple[_Record, ...]
    cost: int


@dataclass(frozen=True)
class _Candidate:
    records: tuple[_PendingRecord, ...]
    cost: int
    padding_ratio: float
    arrival: int


@dataclass(frozen=True)
class _CandidateRange:
    start: int
    stop: int
    cost: int
    padding_ratio: float
    arrival: int


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
        distributed_plan_window: int = 32,
        max_batch_samples: int | None = None,
        max_padding_ratio: float = 0.2,
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
        self.distributed_plan_window = _positive_int(
            "distributed_plan_window", distributed_plan_window
        )
        self.max_batch_samples = (
            None
            if max_batch_samples is None
            else _positive_int("max_batch_samples", max_batch_samples)
        )
        self.max_padding_ratio = _non_negative_float(
            "max_padding_ratio", max_padding_ratio
        )
        if not isinstance(drop_distributed_tail, bool):
            raise TypeError("drop_distributed_tail must be a bool.")
        self.drop_distributed_tail = drop_distributed_tail
        self.sampler: Sampler[int] = (
            sampler if sampler is not None else _DatasetSampler(self)
        )

    def __iter__(self) -> Iterator[list[int]]:
        plans = self._iter_plans()
        for plan in synchronized_plans(
            plans,
            drop_tail=self.drop_distributed_tail,
            plan_window=self.distributed_plan_window,
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
                max_padding_ratio=self.max_padding_ratio,
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
        num_replicas, rank_id = rank()
        groups = self.dataset._shuffle(
            shuffle=self.shuffle,
            seed=self.seed,
            epoch=self.epoch,
            num_replicas=num_replicas,
            rank=rank_id,
        )
        if debug_plans_enabled():
            yield from self._debug_dataset_index_groups(
                groups,
                num_replicas=num_replicas,
                rank_id=rank_id,
            )
            return
        yield from groups

    def _debug_dataset_index_groups(
        self,
        groups: Iterable[Sequence[int]],
        *,
        num_replicas: int,
        rank_id: int,
    ) -> Iterator[Sequence[int]]:
        log_debug_plan(
            "rank-local dataset index groups "
            f"rank={rank_id} world_size={num_replicas} "
            f"dataset_length={len(self.dataset)} shuffle={self.shuffle} "
            f"seed={self.seed} epoch={self.epoch}"
        )
        count = 0
        for indexes in groups:
            if count < 3:
                log_debug_plan(
                    "rank-local dataset index group "
                    f"rank={rank_id} group={count} length={len(indexes)} "
                    f"head={_index_group_head(indexes)}"
                )
            count += 1
            yield indexes
        if count == 0:
            log_debug_plan(
                "rank-local dataset index groups empty "
                f"rank={rank_id} world_size={num_replicas} "
                f"dataset_length={len(self.dataset)}"
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
        distributed_plan_window: int = 32,
        max_batch_samples: int | None = None,
        max_padding_ratio: float = 0.2,
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
            distributed_plan_window=distributed_plan_window,
            max_batch_samples=max_batch_samples,
            max_padding_ratio=max_padding_ratio,
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
    max_padding_ratio: float = 0.2,
) -> Iterator[_Plan]:
    source = iter(records)
    pending: list[_PendingRecord] = []
    next_arrival = 0
    source_exhausted = False
    window = 1 if max_batch_samples == 1 else planning_window
    while True:
        while len(pending) < window and not source_exhausted:
            try:
                record = next(source)
            except StopIteration:
                source_exhausted = True
                break
            pending.append(_PendingRecord(arrival=next_arrival, record=record))
            next_arrival += 1
        if not pending:
            return
        oldest = min(pending, key=lambda item: item.arrival)
        if oldest.record.cost > max_batch_memory:
            raise ValueError(
                "A sample exceeds max_batch_memory: "
                f"index={oldest.record.index} memory={oldest.record.cost} "
                f"budget={max_batch_memory}."
            )
        candidate = _select_candidate(
            pending,
            max_batch_memory=max_batch_memory,
            max_batch_samples=max_batch_samples,
            max_padding_ratio=max_padding_ratio,
        )
        selected_arrivals = {item.arrival for item in candidate.records}
        pending = [item for item in pending if item.arrival not in selected_arrivals]
        selected = tuple(
            item.record for item in sorted(candidate.records, key=lambda item: item.arrival)
        )
        yield _Plan(records=selected, cost=candidate.cost)


def _select_candidate(
    pending: Sequence[_PendingRecord],
    *,
    max_batch_memory: int,
    max_batch_samples: int | None,
    max_padding_ratio: float,
) -> _Candidate:
    sorted_pending = sorted(
        pending,
        key=lambda item: (item.record.cost, item.arrival),
    )
    prefix_costs = [0]
    for item in sorted_pending:
        prefix_costs.append(prefix_costs[-1] + item.record.cost)

    tree_size = 1 << (len(sorted_pending) - 1).bit_length()
    arrival_sentinel = max(item.arrival for item in sorted_pending) + 1
    arrival_tree = [arrival_sentinel] * (tree_size * 2)
    for offset, item in enumerate(sorted_pending):
        arrival_tree[tree_size + offset] = item.arrival
    for offset in range(tree_size - 1, 0, -1):
        arrival_tree[offset] = min(
            arrival_tree[offset * 2], arrival_tree[offset * 2 + 1]
        )

    def minimum_arrival(start: int, stop: int) -> int:
        left = tree_size + start
        right = tree_size + stop + 1
        result = arrival_sentinel
        while left < right:
            if left & 1:
                result = min(result, arrival_tree[left])
                left += 1
            if right & 1:
                right -= 1
                result = min(result, arrival_tree[right])
            left //= 2
            right //= 2
        return result

    def first_start_with_padding_at_most(
        start: int,
        stop: int,
        maximum: float,
    ) -> int:
        left = start
        right = stop - 1
        while left < right:
            middle = (left + right) // 2
            count = stop - middle + 1
            cost = prefix_costs[stop + 1] - prefix_costs[middle]
            if (
                _padding_ratio(
                    cost,
                    sorted_pending[stop].record.cost,
                    count,
                )
                <= maximum
            ):
                right = middle
            else:
                left = middle + 1
        return left

    def candidate_range(start: int, stop: int) -> _CandidateRange:
        cost = prefix_costs[stop + 1] - prefix_costs[start]
        return _CandidateRange(
            start=start,
            stop=stop,
            cost=cost,
            padding_ratio=_padding_ratio(
                cost,
                sorted_pending[stop].record.cost,
                stop - start + 1,
            ),
            arrival=minimum_arrival(start, stop),
        )

    best_threshold: _CandidateRange | None = None
    feasible_ranges: list[tuple[int, int, float]] = []
    if max_batch_samples is None or max_batch_samples >= 2:
        for stop in range(1, len(sorted_pending)):
            # For a fixed maximum cost, both budget feasibility and padding
            # improve monotonically as the range start moves right.
            start = bisect_left(
                prefix_costs,
                prefix_costs[stop + 1] - max_batch_memory,
                0,
                stop,
            )
            if max_batch_samples is not None:
                start = max(start, stop - max_batch_samples + 1)
            if start >= stop:
                continue

            pair_start = stop - 1
            pair_cost = prefix_costs[stop + 1] - prefix_costs[pair_start]
            pair_padding = _padding_ratio(
                pair_cost,
                sorted_pending[stop].record.cost,
                2,
            )
            feasible_ranges.append((start, stop, pair_padding))
            if pair_padding > max_padding_ratio:
                continue
            threshold = candidate_range(
                first_start_with_padding_at_most(
                    start,
                    stop,
                    max_padding_ratio,
                ),
                stop,
            )
            if best_threshold is None or (
                threshold.cost,
                -threshold.padding_ratio,
                -threshold.arrival,
                -threshold.start,
                -threshold.stop,
            ) > (
                best_threshold.cost,
                -best_threshold.padding_ratio,
                -best_threshold.arrival,
                -best_threshold.start,
                -best_threshold.stop,
            ):
                best_threshold = threshold

    best_fallback: _CandidateRange | None = None
    if best_threshold is None:
        for start, stop, pair_padding in feasible_ranges:
            # Preserve the largest cost on a rounded minimum-padding plateau.
            fallback = candidate_range(
                first_start_with_padding_at_most(start, stop, pair_padding),
                stop,
            )
            if best_fallback is None or (
                -fallback.padding_ratio,
                fallback.cost,
                -fallback.arrival,
                -fallback.start,
                -fallback.stop,
            ) > (
                -best_fallback.padding_ratio,
                best_fallback.cost,
                -best_fallback.arrival,
                -best_fallback.start,
                -best_fallback.stop,
            ):
                best_fallback = fallback

    selected = best_threshold if best_threshold is not None else best_fallback
    if selected is not None:
        return _Candidate(
            records=tuple(sorted_pending[selected.start : selected.stop + 1]),
            cost=selected.cost,
            padding_ratio=selected.padding_ratio,
            arrival=selected.arrival,
        )

    single = max(
        (
            item
            for item in sorted_pending
            if item.record.cost <= max_batch_memory
        ),
        key=lambda item: (item.record.cost, -item.arrival),
        default=None,
    )
    if single is not None:
        return _Candidate(
            records=(single,),
            cost=single.record.cost,
            padding_ratio=0.0,
            arrival=single.arrival,
        )
    oldest = min(pending, key=lambda item: item.arrival)
    raise ValueError(
        "A sample exceeds max_batch_memory: "
        f"index={oldest.record.index} memory={oldest.record.cost} "
        f"budget={max_batch_memory}."
    )


def _padding_ratio(cost: int, max_cost: int, count: int) -> float:
    padded = max_cost * count
    return 0.0 if padded == 0 else (padded - cost) / padded


def _int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    return operator.index(value)


def _non_negative_int(name: str, value: int) -> int:
    resolved = _int(name, value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative.")
    return resolved


def _non_negative_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number.")
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
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


def _index_group_head(indexes: Sequence[int], limit: int = 8) -> tuple[int, ...]:
    return tuple(indexes[:limit])


def _record(costs: _Costs, index: int) -> _Record:
    resolved = operator.index(index)
    return _Record(index=resolved, cost=_sample_cost(costs, resolved))


def _sample_cost(costs: _Costs, index: int) -> int:
    cost = 1 if costs is None else operator.index(costs[index])
    if cost <= 0:
        raise ValueError(f"sample cost must be a positive integer: index={index}.")
    return cost
