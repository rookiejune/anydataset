from __future__ import annotations

import math
import random

import torch

from anydataset.dataset.collate import FieldGroup, FieldRef, _batch_tensors
from anydataset.dataset.multiple import WeightedRandomStrategy
from anydataset.types.item import Modality, Role, Sample, TextItem, TextView

_TEXT_REF = (Role.DEFAULT, Modality.TEXT)


def test_variable_length_batch_uses_one_padded_allocation() -> None:
    batch, mask = _batch_tensors(
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0])),
        field=FieldRef(
            ref=(Role.DEFAULT, Modality.TEXT),
            group=FieldGroup.VIEWS,
            key=TextView.TEXT,
        ),
    )

    assert batch.tolist() == [[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]]
    assert mask.tolist() == [[True, True, True], [True, False, False]]


def test_variable_length_batch_preserves_dtype_promotion() -> None:
    batch, _mask = _batch_tensors(
        (
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([3.5], dtype=torch.float32),
        ),
        field=FieldRef(
            ref=(Role.DEFAULT, Modality.TEXT),
            group=FieldGroup.VIEWS,
            key=TextView.TEXT,
        ),
    )

    assert batch.dtype == torch.float32
    assert batch.tolist() == [[1.0, 2.0], [3.5, 0.0]]


def test_large_weighted_strategy_removes_exhausted_datasets_without_rebuilding_all_weights() -> None:
    datasets = _weighted_datasets(65)

    rows = list(WeightedRandomStrategy(seed=7).iter(datasets))

    assert len(rows) == len(datasets)
    assert _weighted_values(rows) == set(range(65))


def test_large_weighted_strategy_preserves_tiny_active_weights() -> None:
    datasets = _weighted_datasets(65)
    weights = (1.0, *([1e-20] * 64))

    rows = list(WeightedRandomStrategy(weights=weights, seed=7).iter(datasets))

    assert len(rows) == len(datasets)
    assert _weighted_values(rows) == set(range(65))


def test_large_weighted_strategy_handles_subnormal_remaining_totals() -> None:
    weight_rng = random.Random(3)
    for case in range(18):
        count = weight_rng.randint(65, 300)
        weights = tuple(
            math.ldexp(1.0, weight_rng.randint(-1074, 1023))
            for _ in range(count)
        )
        datasets = _weighted_datasets(count)

        rows = list(WeightedRandomStrategy(weights=weights, seed=case).iter(datasets))

        assert len(rows) == count
        assert _weighted_values(rows) == set(range(count))


def _weighted_datasets(count: int) -> tuple[tuple[Sample], ...]:
    return tuple(
        (
            {
                _TEXT_REF: TextItem(
                    views={TextView.TEXT: str(index)},
                )
            },
        )
        for index in range(count)
    )


def _weighted_values(rows: list[Sample]) -> set[int]:
    output: set[int] = set()
    for row in rows:
        item = row[_TEXT_REF]
        assert isinstance(item, TextItem)
        output.add(int(item.views[TextView.TEXT]))
    return output
