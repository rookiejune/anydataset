from __future__ import annotations

from collections.abc import Sequence

import pytest

from anydataset import AnyDataset, BatchCost, CostDataLoader, Source, Spec


def _dataset(rows, *, parse_fn=lambda row: row, cost_fn=lambda row: row):
    dataset = AnyDataset(
        Spec(source=Source.STORE, path="unused"),
        parse_fn=parse_fn,
        cost_fn=cost_fn,
    )
    dataset._dataset = rows
    return dataset


def _padded_cost(costs: Sequence[int]) -> BatchCost:
    padded = max(costs) * len(costs)
    return BatchCost(
        memory=padded,
        compute=float(sum(cost * cost for cost in costs)),
        waste=float(padded - sum(costs)),
    )


def test_cost_does_not_parse_sample() -> None:
    parsed: list[int] = []
    measured: list[int] = []

    def parse(row: int) -> int:
        parsed.append(row)
        return row * 10

    def cost(row: int) -> int:
        measured.append(row)
        return row

    dataset = _dataset([4, 1, 3, 2], parse_fn=parse, cost_fn=cost)
    loader = CostDataLoader(
        dataset,
        batch_cost_fn=_padded_cost,
        max_batch_memory=8,
        planning_window=4,
        collate_fn=list,
    )

    iterator = iter(loader)
    assert parsed == []
    batches = list(iterator)

    assert measured == [4, 1, 3, 2]
    assert parsed == [4, 3, 1, 2]
    assert batches == [[40, 30], [10, 20]]


def test_requires_dataset_cost_fn() -> None:
    dataset = _dataset([1], cost_fn=None)

    with pytest.raises(TypeError, match="must define cost_fn"):
        CostDataLoader(
            dataset,
            batch_cost_fn=_padded_cost,
            max_batch_memory=1,
        )


def test_rejects_oversized_sample() -> None:
    loader = CostDataLoader(
        _dataset([9]),
        batch_cost_fn=_padded_cost,
        max_batch_memory=8,
    )

    with pytest.raises(ValueError, match="index=0 memory=9 budget=8"):
        list(loader)


def test_batch_cost_validates_fields() -> None:
    with pytest.raises(ValueError, match="memory must be positive"):
        BatchCost(memory=0, compute=1, waste=0)
    with pytest.raises(ValueError, match="compute must be finite and positive"):
        BatchCost(memory=1, compute=float("nan"), waste=0)
    with pytest.raises(ValueError, match="waste must be finite and non-negative"):
        BatchCost(memory=1, compute=1, waste=-1)


def test_batch_count_is_explicitly_unavailable() -> None:
    loader = CostDataLoader(
        _dataset([1]),
        batch_cost_fn=_padded_cost,
        max_batch_memory=1,
    )

    with pytest.raises(TypeError, match="unavailable before planning"):
        len(loader)


def test_set_epoch_forwards_to_custom_sampler() -> None:
    class EpochSampler:
        epoch = None

        def __iter__(self):
            return iter([0])

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

    sampler = EpochSampler()
    loader = CostDataLoader(
        _dataset([1]),
        batch_cost_fn=_padded_cost,
        max_batch_memory=1,
        sampler=sampler,
    )

    loader.set_epoch(4)

    assert sampler.epoch == 4
