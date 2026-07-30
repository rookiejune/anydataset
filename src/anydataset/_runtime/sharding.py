from __future__ import annotations

import os
from dataclasses import dataclass

from torch import distributed as dist
from torch.utils.data import get_worker_info


def validate_shard(num_shards: int, shard_id: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards.")


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
    rank_count, rank_index = _runtime_rank()

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


def _runtime_rank() -> tuple[int, int]:
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
