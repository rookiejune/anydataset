"""Distributed plan synchronization for cost-aware dataloaders."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar

import torch
import torch.distributed as dist


_PLAN_WINDOW = 128
_T = TypeVar("_T")


def synchronized_plans(
    plans: Iterable[_T],
    *,
    drop_tail: bool,
) -> Iterator[_T]:
    if not dist.is_available() or not dist.is_initialized():
        yield from plans
        return

    source = iter(plans)
    world_size = dist.get_world_size()
    while True:
        local = list(islice(source, _PLAN_WINDOW))
        counts = plan_counts(len(local), world_size)
        kept = min(counts)
        if any(count != kept for count in counts):
            if not drop_tail:
                raise RuntimeError(
                    "dataloader cannot keep equal rank-local batch counts."
                )
            if len(local) > kept:
                warnings.warn(
                    "dataloader dropped rank-local final batches for equal DDP steps.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        yield from local[:kept]
        if kept < _PLAN_WINDOW:
            return


def plan_counts(local_count: int, world_size: int) -> tuple[int, ...]:
    backend = str(dist.get_backend()).lower()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    local = torch.tensor([local_count], dtype=torch.int64, device=device)
    gathered = torch.empty(world_size, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local)
    return tuple(int(value) for value in gathered.cpu().tolist())


def rank() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        return 1, 0
    return dist.get_world_size(), dist.get_rank()
