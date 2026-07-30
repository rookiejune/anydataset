from __future__ import annotations

import torch

from anydataset.dataset.collate import FieldGroup, FieldRef, _batch_tensors
from anydataset.dataset.multiple import WeightedRandomStrategy
from anydataset.types.item import Modality, Role, TextView


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
    datasets = tuple(({"value": index},) for index in range(65))

    rows = list(WeightedRandomStrategy(seed=7).iter(datasets))

    assert len(rows) == len(datasets)
    assert {row["value"] for row in rows} == set(range(65))
