from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Callable, cast

from ..types.item import Sample
from ._selection import selected_index_groups
from .abc import MapStyleABC


class IndexSelection(MapStyleABC):
    """Read a stable set of positions from a map-style dataset."""

    __slots__ = ("_dataset", "_indices")

    def __init__(self, dataset: MapStyleABC, indices: Sequence[int]) -> None:
        if not isinstance(dataset, MapStyleABC):
            raise TypeError("dataset must be a MapStyleABC.")
        self._dataset = dataset
        self._indices = _indices(indices, len(dataset))

    @property
    def dataset(self) -> MapStyleABC:
        return self._dataset

    @property
    def indices(self) -> tuple[int, ...]:
        return self._indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Sample:
        return self.dataset[self._indices[index]]

    def cost_row(self, index: int) -> Any:
        return self.dataset.cost_row(self._indices[index])

    def global_index(self, index: int) -> int:
        position = self._indices[index]
        method = getattr(self.dataset, "global_index", None)
        if not callable(method):
            return position
        return cast(Callable[[int], int], method)(position)

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        yield from selected_index_groups(
            self.dataset,
            self._indices,
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            num_replicas=num_replicas,
            rank=rank,
        )


def _indices(values: Sequence[int], length: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("indices must be a sequence of integers.")
    output: list[int] = []
    previous = -1
    for position, value in enumerate(values):
        if type(value) is not int:
            raise TypeError(f"indices[{position}] must be an integer.")
        if value <= previous:
            raise ValueError("indices must be strictly increasing.")
        if value >= length:
            raise ValueError("index exceeds the dataset length.")
        output.append(value)
        previous = value
    return tuple(output)


__all__ = ["IndexSelection"]
