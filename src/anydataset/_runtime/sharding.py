from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from torch import distributed as dist
from torch.utils.data import get_worker_info


def validate_shard(num_shards: int, shard_id: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards.")


def validated_shard_rows(
    rows: Any,
    *,
    num_shards: int,
    shard_id: int,
    label: str,
) -> Iterator[tuple[int, Any]]:
    validate_shard(num_shards, shard_id)
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise TypeError(f"{label} must return an iterable.") from exc
    expected = shard_id
    for entry in iterator:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"{label} must yield (sample_index, value) tuples.")
        sample_index, value = entry
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError(f"{label} sample indexes must be integers.")
        if sample_index != expected:
            raise ValueError(
                f"{label} must yield dense global sample indexes: "
                f"expected {expected}, got {sample_index}."
            )
        yield sample_index, value
        expected += num_shards


def validated_range_rows(
    rows: Any,
    *,
    start: int,
    stop: int,
    label: str,
) -> Iterator[tuple[int, Any]]:
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise TypeError(f"{label} must return an iterable.") from exc
    expected = start
    for entry in iterator:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"{label} must yield (sample_index, value) tuples.")
        sample_index, value = entry
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError(f"{label} sample indexes must be integers.")
        if expected >= stop:
            raise ValueError(
                f"{label} must stop at sample index {stop}; got extra index "
                f"{sample_index}."
            )
        if sample_index != expected:
            raise ValueError(
                f"{label} must yield dense range indexes: expected {expected}, "
                f"got {sample_index}."
            )
        yield sample_index, value
        expected += 1
    if expected != stop:
        raise ValueError(
            f"{label} must cover [{start}, {stop}); stopped before index {expected}."
        )


def validate_range(sample_count: int, start: int, stop: int) -> None:
    if start < 0 or stop < start or stop > sample_count:
        raise ValueError("range must satisfy 0 <= start <= stop <= len(dataset).")


def iter_map_style_range(
    dataset: Any,
    start: int,
    stop: int,
) -> Iterator[tuple[int, Any]]:
    validate_range(len(dataset), start, stop)
    for index in range(start, stop):
        yield index, dataset[index]


def iter_map_style_shard(
    dataset: Any,
    num_shards: int,
    shard_id: int,
) -> Iterator[tuple[int, Any]]:
    validate_shard(num_shards, shard_id)
    for index in range(shard_id, len(dataset), num_shards):
        yield index, dataset[index]


@dataclass(frozen=True)
class Shard:
    count: int = 1
    index: int = 0
    rank_count: int = 1
    rank_index: int = 0
    worker_count: int = 1
    worker_index: int = 0

    def __post_init__(self) -> None:
        validate_shard(self.count, self.index)
        validate_shard(self.rank_count, self.rank_index)
        validate_shard(self.worker_count, self.worker_index)

    def split(self, count: int, index: int) -> Shard:
        validate_shard(count, index)
        return Shard(
            count=self.count * count,
            index=self.index * count + index,
            rank_count=self.rank_count,
            rank_index=self.rank_index,
            worker_count=self.worker_count,
            worker_index=self.worker_index,
        )

    @property
    def flat_count(self) -> int:
        return self.rank_count * self.worker_count

    @property
    def flat_index(self) -> int:
        return self.worker_index * self.rank_count + self.rank_index


def runtime_shard() -> Shard:
    rank_count, rank_index = runtime_rank()

    worker_count = 1
    worker_index = 0

    worker = get_worker_info()
    if worker is not None:
        worker_count = worker.num_workers
        worker_index = worker.id

    return Shard(
        count=rank_count * worker_count,
        index=worker_index * rank_count + rank_index,
        rank_count=rank_count,
        rank_index=rank_index,
        worker_count=worker_count,
        worker_index=worker_index,
    )


def runtime_rank() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    world_size = _optional_env_int("WORLD_SIZE")
    rank = _optional_env_int("RANK")
    if world_size is None and rank is None:
        return 1, 0
    if world_size is None or rank is None:
        raise ValueError("RANK and WORLD_SIZE must be set together.")
    return world_size, rank


def _optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
