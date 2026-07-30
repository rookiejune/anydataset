from __future__ import annotations

import pickle
from collections.abc import Iterator, Sequence

import pytest

from anydataset.dataset import IndexSelection, MapStyleABC
from anydataset.types import AudioItem, AudioView, Modality, Role, Sample


class _Dataset(MapStyleABC):
    def __init__(self, values: Sequence[int]) -> None:
        self.values = tuple(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Sample:
        return {
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: self.values[index]}
            )
        }

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        if num_replicas != 1 or rank != 0:
            raise AssertionError("selection must request the complete base ordering")
        groups = ((2, 3), (0, 1), (4,)) if shuffle else ((0, 1), (2, 3), (4,))
        yield from groups


def _values(dataset: IndexSelection) -> list[int]:
    return [
        sample[Role.DEFAULT, Modality.AUDIO].views[AudioView.WAVEFORM]
        for sample in dataset
    ]


def _shuffled(dataset: IndexSelection, *, rank: int) -> tuple[int, ...]:
    return tuple(
        index
        for group in dataset._shuffle(
            shuffle=True,
            seed=0,
            epoch=0,
            num_replicas=2,
            rank=rank,
        )
        for index in group
    )


def test_selects_dataset_positions_and_exposes_global_indexes() -> None:
    selected = IndexSelection(_Dataset(range(5)), (1, 3, 4))

    assert _values(selected) == [1, 3, 4]
    assert selected.indices == (1, 3, 4)
    assert selected.global_index(1) == 3


def test_empty_selection() -> None:
    selected = IndexSelection(_Dataset(range(3)), ())

    assert len(selected) == 0
    assert list(selected) == []


@pytest.mark.parametrize("indices", [(1, 1), (2, 1), (-1,), (3,), (True,)])
def test_rejects_invalid_indices(indices: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        IndexSelection(_Dataset(range(3)), indices)  # type: ignore[arg-type]


def test_pickle_round_trip() -> None:
    selected = IndexSelection(_Dataset(range(5)), (1, 4))

    restored = pickle.loads(pickle.dumps(selected))

    assert restored.indices == (1, 4)
    assert _values(restored) == [1, 4]


def test_shuffle_preserves_base_groups_and_distributes_selected_positions() -> None:
    selected = IndexSelection(_Dataset(range(5)), (0, 2, 3, 4))

    first = _shuffled(selected, rank=0)
    second = _shuffled(selected, rank=1)

    assert first == (1, 0)
    assert second == (2, 3)
    assert set(first).isdisjoint(second)
    assert set(first) | set(second) == set(range(len(selected)))
