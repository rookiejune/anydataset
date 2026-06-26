from __future__ import annotations

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

    def __post_init__(self) -> None:
        validate_shard(self.count, self.index)

    def split(self, count: int, index: int) -> Shard:
        validate_shard(count, index)
        return Shard(
            count=self.count * count,
            index=self.index * count + index,
        )


def runtime_shard() -> Shard:
    shard = Shard()

    if dist.is_available() and dist.is_initialized():
        shard = shard.split(dist.get_world_size(), dist.get_rank())

    worker = get_worker_info()
    if worker is not None:
        shard = shard.split(worker.num_workers, worker.id)

    return shard
