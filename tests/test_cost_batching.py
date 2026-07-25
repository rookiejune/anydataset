from __future__ import annotations

import anydataset
import anydataset.dataset
import pytest

from anydataset import AnyDataset, Source, Spec
from anydataset.dataset import MapStyleABC


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

    def cost(index: int) -> int:
        measured.append(index)
        return rows[index]

    loader = dataset.dataloader(
        cost_fn=cost,
        max_batch_memory=8,
        planning_window=4,
        collate_fn=list,
    )

    iterator = iter(loader)
    assert parsed == []
    batches = list(iterator)

    assert measured == [0, 1, 2, 3]
    assert parsed == [4, 3, 1, 2]
    assert batches == [[40, 30, 10], [20]]


def test_requires_cost_fn() -> None:
    dataset = _dataset([1])

    with pytest.raises(TypeError, match="cost_fn must be callable"):
        dataset.dataloader(
            cost_fn=None,
            max_batch_memory=1,
        )


def test_rejects_oversized_sample() -> None:
    loader = _dataset([9]).dataloader(
        cost_fn=lambda index: [9][index],
        max_batch_memory=8,
    )

    with pytest.raises(ValueError, match="index=0 memory=9 budget=8"):
        list(loader)


def test_rejects_non_positive_sample_cost() -> None:
    loader = _dataset([0]).dataloader(
        cost_fn=lambda index: [0][index],
        max_batch_memory=1,
    )

    with pytest.raises(ValueError, match="cost_fn must return a positive integer"):
        list(loader)


def test_batch_count_is_explicitly_unavailable() -> None:
    loader = _dataset([1]).dataloader(
        cost_fn=lambda index: [1][index],
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
        cost_fn=lambda index: [1][index],
        max_batch_memory=1,
        sampler=sampler,
    )

    loader.set_epoch(4)

    assert sampler.epoch == 4


def test_dataloader_uses_dataset_shuffle_groups() -> None:
    dataset = _GroupedDataset()
    loader = dataset.dataloader(
        cost_fn=lambda _index: 1,
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


def test_map_style_abc_can_use_dataloader() -> None:
    dataset = _IndexDataset([4, 1, 3, 2])

    loader = dataset.dataloader(
        cost_fn=lambda index: dataset.rows[index],
        max_batch_memory=8,
        planning_window=4,
        collate_fn=list,
    )

    assert list(loader) == [[40, 30, 10], [20]]


def test_loader_class_is_not_public_api() -> None:
    assert callable(MapStyleABC.dataloader)
    assert "AnyDataset" in anydataset.__all__
    assert "MapStyleABC" in anydataset.dataset.__all__
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
