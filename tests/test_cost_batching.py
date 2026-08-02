from __future__ import annotations

from collections.abc import Sequence
from unittest import mock

import anydataset
import anydataset.dataset
import pytest
import torch

from anydataset import AnyDataset, Source, Spec
from anydataset.dataset import MapStyleABC
from anydataset.dataset.batching import (
    _Plan,
    _Record,
    _plans,
)
from anydataset.dataset._ddp import plan_counts, synchronized_plans


def _dataset(rows, *, parse_fn=lambda row: row):
    dataset = AnyDataset(
        Spec(source=Source.STORE, path="unused"),
        parse_fn=parse_fn,
    )
    dataset._dataset = rows
    return dataset


def test_cost_does_not_parse_sample() -> None:
    rows = [4, 1, 3, 2]
    parsed: list[int] = []
    measured: list[int] = []

    def parse(row: int) -> int:
        parsed.append(row)
        return row * 10

    dataset = _dataset(rows, parse_fn=parse)

    loader = dataset.dataloader(
        costs=lambda row: measured.append(row) or row,
        max_batch_memory=8,
        planning_window=4,
        collate_fn=list,
    )

    iterator = iter(loader)
    assert parsed == []
    batches = list(iterator)

    assert measured == [4, 1, 3, 2]
    assert parsed == [4, 3, 1, 2]
    assert batches == [[40, 30, 10], [20]]


def test_accepts_plain_iterable_costs() -> None:
    loader = _dataset([4, 1, 3, 2]).dataloader(
        costs=(cost for cost in [4, 1, 3, 2]),
        max_batch_memory=8,
        planning_window=4,
        collate_fn=list,
    )

    assert list(loader) == [[4, 3, 1], [2]]


def test_requires_supported_costs() -> None:
    dataset = _dataset([1])

    with pytest.raises(TypeError, match="costs must be None"):
        dataset.dataloader(
            costs=object(),
            max_batch_memory=1,
        )


def test_requires_costs_to_match_dataset_length() -> None:
    with pytest.raises(ValueError, match="costs and dataset must have equal length"):
        _dataset([1]).dataloader(
            costs=[1, 1],
            max_batch_memory=1,
        )


def test_none_costs_use_unit_cost() -> None:
    loader = _dataset([1, 2, 3]).dataloader(
        costs=None,
        max_batch_memory=2,
        collate_fn=list,
    )

    assert list(loader) == [[1, 2], [3]]


def test_rejects_integer_costs() -> None:
    with pytest.raises(TypeError, match="costs must be None"):
        _dataset([1]).dataloader(
            costs=99,  # type: ignore[arg-type]
            max_batch_memory=1,
        )


def test_rejects_oversized_sample() -> None:
    loader = _dataset([9]).dataloader(
        costs=[9],
        max_batch_memory=8,
    )

    with pytest.raises(ValueError, match="index=0 memory=9 budget=8"):
        list(loader)


def test_rejects_non_positive_sample_cost() -> None:
    loader = _dataset([0]).dataloader(
        costs=[0],
        max_batch_memory=1,
    )

    with pytest.raises(ValueError, match="sample cost must be a positive integer"):
        list(loader)


def test_batch_count_is_explicitly_unavailable() -> None:
    loader = _dataset([1]).dataloader(
        costs=[1],
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
    loader = _dataset([1]).dataloader(
        costs=[1],
        max_batch_memory=1,
        sampler=sampler,
    )

    loader.set_epoch(4)

    assert sampler.epoch == 4


def test_dataloader_uses_dataset_shuffle_groups() -> None:
    dataset = _GroupedDataset()
    loader = dataset.dataloader(
        costs=None,
        max_batch_memory=2,
        max_batch_samples=2,
        shuffle=True,
        seed=7,
        epoch=2,
        collate_fn=list,
    )

    assert list(loader) == [[1, 0], [3, 2]]
    assert dataset.calls == [(True, 7, 2, 1, 0)]

    loader.set_epoch(5)

    assert list(loader) == [[1, 0], [3, 2]]
    assert dataset.calls[-1] == (True, 7, 5, 1, 0)


def test_pytorch_sampler_epoch_contract_advances_dataset_shuffle() -> None:
    dataset = _GroupedDataset()
    loader = dataset.dataloader(
        costs=None,
        max_batch_memory=2,
        max_batch_samples=2,
        shuffle=True,
        seed=7,
        epoch=2,
        collate_fn=list,
    )

    loader.batch_sampler.sampler.set_epoch(5)

    assert list(loader) == [[1, 0], [3, 2]]
    assert dataset.calls == [(True, 7, 5, 1, 0)]


def test_map_style_abc_can_use_dataloader() -> None:
    dataset = _IndexDataset([4, 1, 3, 2])

    loader = dataset.dataloader(
        costs=dataset.rows,
        max_batch_memory=8,
        planning_window=4,
        collate_fn=list,
    )

    assert list(loader) == [[40, 30, 10], [20]]


def test_callable_costs_use_map_style_cost_row() -> None:
    dataset = _CostRowDataset([4, 1, 3, 2])

    loader = dataset.dataloader(
        costs=lambda row: row["cost"],
        max_batch_memory=8,
        planning_window=4,
        collate_fn=list,
    )

    assert list(loader) == [[40, 30, 10], [20]]
    assert dataset.cost_rows == [0, 1, 2, 3]


def test_map_style_shuffle_strides_flattened_groups_across_ranks() -> None:
    dataset = _IndexDataset(list(range(10)))
    shuffled = [
        index
        for group in dataset._shuffle(
            shuffle=True,
            seed=7,
            epoch=3,
            num_replicas=1,
            rank=0,
        )
        for index in group
    ]

    rank_indexes = [
        [
            index
            for group in dataset._shuffle(
                shuffle=True,
                seed=7,
                epoch=3,
                num_replicas=3,
                rank=rank,
            )
            for index in group
        ]
        for rank in range(3)
    ]

    assert rank_indexes == [shuffled[rank::3] for rank in range(3)]


def test_cost_dataloader_uses_rank_environment() -> None:
    from anydataset.dataset.batching import _BatchSampler

    dataset = _GroupedDataset()
    sampler = _BatchSampler(
        dataset,
        costs=None,
        max_batch_memory=1,
        sampler=None,
        shuffle=False,
        seed=0,
        epoch=0,
    )
    with mock.patch.dict("os.environ", {"WORLD_SIZE": "2", "RANK": "1"}):
        list(sampler._dataset_index_groups())

    assert dataset.calls == [(False, 0, 0, 2, 1)]


def test_cost_planning_is_lazy() -> None:
    measured: list[int] = []
    costs = _MeasuredCosts([1] * 100, measured)
    loader = _dataset([1] * 100).dataloader(
        costs=costs,
        max_batch_memory=1,
        max_batch_samples=1,
        planning_window=4,
        collate_fn=list,
    )

    first = next(iter(loader))

    assert first == [1]
    assert measured == [0]


def test_distributed_planning_sync_is_bounded() -> None:
    consumed: list[int] = []

    def plans():
        for index in range(1_000):
            consumed.append(index)
            yield _Plan(records=(_Record(index=index, cost=1),), cost=1)

    with (
        mock.patch("anydataset.dataset._ddp.dist.is_available", return_value=True),
        mock.patch("anydataset.dataset._ddp.dist.is_initialized", return_value=True),
        mock.patch("anydataset.dataset._ddp.dist.get_world_size", return_value=2),
        mock.patch(
            "anydataset.dataset._ddp.plan_counts",
            side_effect=lambda local, _world_size: (local, 7),
        ),
        pytest.warns(RuntimeWarning, match="dropped rank-local final batches"),
    ):
        synchronized = list(
            synchronized_plans(plans(), drop_tail=True, plan_window=8)
        )

    assert len(synchronized) == 7
    assert consumed == list(range(8))


def test_distributed_planning_fails_fast_on_empty_rank() -> None:
    consumed: list[int] = []

    def plans():
        for index in range(8):
            consumed.append(index)
            yield _Plan(records=(_Record(index=index, cost=1),), cost=1)

    with (
        mock.patch("anydataset.dataset._ddp.dist.is_available", return_value=True),
        mock.patch("anydataset.dataset._ddp.dist.is_initialized", return_value=True),
        mock.patch("anydataset.dataset._ddp.dist.get_world_size", return_value=2),
        mock.patch("anydataset.dataset._ddp.dist.get_rank", return_value=0),
        mock.patch(
            "anydataset.dataset._ddp.plan_counts",
            return_value=(8, 0),
        ),
        pytest.raises(
            RuntimeError,
            match="rank-local planning produced zero batches",
        ),
    ):
        list(synchronized_plans(plans(), drop_tail=True, plan_window=8))

    assert consumed == list(range(8))


def test_distributed_planning_fails_fast_on_invalid_counts() -> None:
    consumed: list[int] = []

    def plans():
        for index in range(8):
            consumed.append(index)
            yield _Plan(records=(_Record(index=index, cost=1),), cost=1)

    with (
        mock.patch("anydataset.dataset._ddp.dist.is_available", return_value=True),
        mock.patch("anydataset.dataset._ddp.dist.is_initialized", return_value=True),
        mock.patch("anydataset.dataset._ddp.dist.get_world_size", return_value=2),
        mock.patch("anydataset.dataset._ddp.dist.get_rank", return_value=0),
        mock.patch(
            "anydataset.dataset._ddp.plan_counts",
            return_value=(8, -1),
        ),
        pytest.raises(
            RuntimeError,
            match="invalid rank-local plan counts",
        ),
    ):
        list(synchronized_plans(plans(), drop_tail=True, plan_window=8))

    assert consumed == list(range(8))


def test_distributed_planning_requires_equal_counts_without_tail_drop() -> None:
    def plans():
        for index in range(8):
            yield _Plan(records=(_Record(index=index, cost=1),), cost=1)

    with (
        mock.patch("anydataset.dataset._ddp.dist.is_available", return_value=True),
        mock.patch("anydataset.dataset._ddp.dist.is_initialized", return_value=True),
        mock.patch("anydataset.dataset._ddp.dist.get_world_size", return_value=2),
        mock.patch(
            "anydataset.dataset._ddp.plan_counts",
            return_value=(8, 7),
        ),
        pytest.raises(RuntimeError, match="equal rank-local batch counts"),
    ):
        list(synchronized_plans(plans(), drop_tail=False, plan_window=8))


def test_distributed_plan_window_is_loader_configurable() -> None:
    loader = _dataset([1]).dataloader(
        costs=None,
        max_batch_memory=1,
        distributed_plan_window=4,
    )

    assert loader.batch_sampler.distributed_plan_window == 4


def test_debug_rank_local_index_groups() -> None:
    from anydataset.dataset.batching import _BatchSampler

    dataset = _GroupedDataset()
    sampler = _BatchSampler(
        dataset,
        costs=None,
        max_batch_memory=1,
        sampler=None,
        shuffle=True,
        seed=7,
        epoch=2,
    )

    with (
        mock.patch.dict("os.environ", {"ANYDATASET_DEBUG_DDP_PLANS": "1"}),
        mock.patch("anydataset.dataset.batching.rank", return_value=(2, 1)),
        mock.patch("anydataset.dataset.batching.log_debug_plan") as debug,
    ):
        groups = list(sampler._dataset_index_groups())

    assert groups == [[1, 0], [3, 2]]
    messages = [call.args[0] for call in debug.call_args_list]
    assert any("rank=1 world_size=2 dataset_length=4" in message for message in messages)
    assert any("shuffle=True seed=7 epoch=2" in message for message in messages)
    assert any("group=0 length=2 head=(1, 0)" in message for message in messages)


def test_requires_positive_distributed_plan_window() -> None:
    with pytest.raises(ValueError, match="distributed_plan_window"):
        _dataset([1]).dataloader(
            costs=None,
            max_batch_memory=1,
            distributed_plan_window=0,
        )


def test_distributed_plan_counts_use_tensor_collective() -> None:
    def gather(output: torch.Tensor, local: torch.Tensor) -> None:
        output.copy_(torch.tensor([int(local.item()), 127], dtype=torch.int64))

    with (
        mock.patch("anydataset.dataset._ddp.dist.get_backend", return_value="gloo"),
        mock.patch(
            "anydataset.dataset._ddp.dist.all_gather_into_tensor",
            side_effect=gather,
        ) as collective,
    ):
        counts = plan_counts(128, 2)

    assert counts == (128, 127)
    collective.assert_called_once()


def test_distributed_plan_counts_use_cuda_device_for_nccl() -> None:
    class FakeTensor:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def cpu(self) -> FakeTensor:
            return self

        def tolist(self) -> list[int]:
            return self.values

    devices: list[object] = []

    def fake_tensor(
        values: list[int],
        *,
        dtype: object,
        device: object,
    ) -> FakeTensor:
        assert dtype is torch.int64
        devices.append(device)
        return FakeTensor(values)

    def fake_full(
        size: tuple[int, ...],
        fill_value: int,
        *,
        dtype: object,
        device: object,
    ) -> FakeTensor:
        assert size == (2,)
        assert fill_value == -1
        assert dtype is torch.int64
        devices.append(device)
        return FakeTensor([fill_value] * size[0])

    def gather(output: FakeTensor, local: FakeTensor) -> None:
        output.values[:] = [local.values[0], 127]

    with (
        mock.patch("anydataset.dataset._ddp.dist.get_backend", return_value="nccl"),
        mock.patch("anydataset.dataset._ddp.torch.cuda.current_device", return_value=3),
        mock.patch("anydataset.dataset._ddp.torch.device") as device,
        mock.patch("anydataset.dataset._ddp.torch.tensor", side_effect=fake_tensor),
        mock.patch("anydataset.dataset._ddp.torch.full", side_effect=fake_full),
        mock.patch(
            "anydataset.dataset._ddp.dist.all_gather_into_tensor",
            side_effect=gather,
        ),
    ):
        device.side_effect = lambda kind, index=None: (kind, index)

        counts = plan_counts(128, 2)

    assert counts == (128, 127)
    assert devices == [("cuda", 3), ("cuda", 3)]


@pytest.mark.parametrize(
    ("budget", "window", "max_samples"),
    [(97, 1, None), (128, 7, 4), (256, 32, 8), (512, 256, None)],
)
def test_streaming_planner_matches_reference_greedy_order(
    budget: int,
    window: int,
    max_samples: int | None,
) -> None:
    costs = [(index * 37) % 97 + 1 for index in range(300)]
    actual = [
        [record.index for record in plan.records]
        for plan in _plans(
            (_Record(index, cost) for index, cost in enumerate(costs)),
            max_batch_memory=budget,
            planning_window=window,
            max_batch_samples=max_samples,
        )
    ]

    assert actual == _reference_plans(
        costs,
        budget=budget,
        window=window,
        max_samples=max_samples,
    )


def test_loader_class_is_not_public_api() -> None:
    assert callable(MapStyleABC.dataloader)
    assert "AnyDataset" in anydataset.__all__
    assert "has_source" not in anydataset.__all__
    assert "MapStyleABC" in anydataset.dataset.__all__
    assert "collate_fn" in anydataset.dataset.__all__
    assert "FieldGroup" in anydataset.dataset.__all__
    assert "FieldRef" in anydataset.dataset.__all__
    assert all(not name.endswith(("Loader", "Sampler")) for name in anydataset.__all__)
    assert all(
        not name.endswith(("Loader", "Sampler"))
        for name in anydataset.dataset.__all__
    )


class _IndexDataset(MapStyleABC):
    def __init__(self, rows: list[int]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> int:
        return self.rows[index] * 10


class _CostRowDataset(_IndexDataset):
    def __init__(self, rows: list[int]) -> None:
        super().__init__(rows)
        self.cost_rows: list[int] = []

    def cost_row(self, index: int):
        self.cost_rows.append(index)
        return {"cost": self.rows[index]}


class _MeasuredCosts(Sequence[int]):
    def __init__(self, values: Sequence[int], measured: list[int]) -> None:
        self.values = tuple(values)
        self.measured = measured

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return self.values[index]
        self.measured.append(index)
        return self.values[index]


def _reference_plans(
    costs: Sequence[int],
    *,
    budget: int,
    window: int,
    max_samples: int | None,
) -> list[list[int]]:
    pending = [_Record(index, cost) for index, cost in enumerate(costs)]
    plans = []
    while pending:
        selected = [pending.pop(0)]
        selected_cost = selected[0].cost
        while max_samples is None or len(selected) < max_samples:
            best: tuple[int, int] | None = None
            for offset, record in enumerate(pending[:window]):
                candidate_cost = selected_cost + record.cost
                if candidate_cost > budget:
                    continue
                if best is None or candidate_cost > best[1]:
                    best = offset, candidate_cost
            if best is None:
                break
            offset, selected_cost = best
            selected.append(pending.pop(offset))
        plans.append([record.index for record in selected])
    return plans


class _GroupedDataset(MapStyleABC):
    def __init__(self) -> None:
        self.calls: list[tuple[bool, int, int, int, int]] = []

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        return index

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ):
        self.calls.append((shuffle, seed, epoch, num_replicas, rank))
        yield [1, 0]
        yield [3, 2]
