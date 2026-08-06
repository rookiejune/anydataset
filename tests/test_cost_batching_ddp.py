from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from anydataset.dataset._ddp import synchronized_plans


def _interleaved_collective_worker(
    rank: int,
    world_size: int,
    init_method: str,
    model_collective_started: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=timedelta(seconds=15),
    )
    try:
        plans = iter(synchronized_plans(range(65), drop_tail=True))
        assert [next(plans) for _ in range(32)] == list(range(32))

        value = torch.tensor(rank + 1, dtype=torch.int64)
        if rank == 1:
            work = dist.all_reduce(value, async_op=True)
            model_collective_started.set()
            work.wait()
        else:
            assert model_collective_started.wait(timeout=5)
            assert next(plans) == 32
            dist.all_reduce(value)

        assert value.item() == 3
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo distributed backend is unavailable.",
)
def test_training_collective_cannot_interleave_with_late_plan_collective(
    tmp_path: Path,
) -> None:
    context = mp.get_context("spawn")
    model_collective_started = context.Event()
    init_method = f"file://{tmp_path / 'ddp-init'}"

    mp.spawn(
        _interleaved_collective_worker,
        args=(2, init_method, model_collective_started),
        nprocs=2,
        join=True,
    )
