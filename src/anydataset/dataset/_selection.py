from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .abc import MapStyleABC


def selected_index_groups(
    dataset: MapStyleABC,
    indices: Sequence[int],
    *,
    shuffle: bool,
    seed: int,
    epoch: int,
    num_replicas: int,
    rank: int,
) -> Iterator[Sequence[int]]:
    """Map base index groups into a selected dataset's position space."""

    positions = {index: position for position, index in enumerate(indices)}
    selected_position = 0
    groups = dataset._shuffle(
        shuffle=shuffle,
        seed=seed,
        epoch=epoch,
        num_replicas=1,
        rank=0,
    )
    for group in groups:
        selected = [positions[index] for index in group if index in positions]
        if not selected:
            continue
        offset = (rank - selected_position) % num_replicas
        selected_position += len(selected)
        if offset < len(selected):
            yield selected[offset::num_replicas]


__all__: list[str] = []
