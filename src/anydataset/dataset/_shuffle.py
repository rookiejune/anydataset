from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence


def index_groups(length: int, size: int) -> Iterator[range]:
    for start in range(0, length, size):
        yield range(start, min(start + size, length))


def shuffle_index_groups(
    groups: Iterable[Sequence[int]],
    *,
    shuffle: bool = True,
    seed: int,
    epoch: int,
    num_replicas: int,
    rank: int,
) -> Iterator[Sequence[int]]:
    if shuffle:
        shuffled = [group for group in groups if len(group) > 0]
        rng = random.Random(seed + epoch)
        rng.shuffle(shuffled)
        ordered: Iterable[Sequence[int]] = shuffled
    else:
        ordered = groups
        rng = None

    position = 0
    for group in ordered:
        size = len(group)
        if size == 0:
            continue
        group_seed = None if rng is None else rng.getrandbits(64)
        offset = (rank - position) % num_replicas
        position += size
        if offset >= size:
            continue
        if group_seed is None:
            yield group[offset::num_replicas]
            continue
        indexes = list(group)
        random.Random(group_seed).shuffle(indexes)
        yield indexes[offset::num_replicas]
