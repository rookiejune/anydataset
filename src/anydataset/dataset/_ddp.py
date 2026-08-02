"""Distributed plan synchronization for cost-aware dataloaders."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from itertools import islice
import logging
import os
from time import perf_counter
from typing import TypeVar

import torch
import torch.distributed as dist

from .._runtime.sharding import runtime_rank


_DEFAULT_PLAN_WINDOW = 32
_DEBUG_ENV = "ANYDATASET_DEBUG_DDP_PLANS"
_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")


def synchronized_plans(
    plans: Iterable[_T],
    *,
    drop_tail: bool,
    plan_window: int = _DEFAULT_PLAN_WINDOW,
) -> Iterator[_T]:
    if not dist.is_available() or not dist.is_initialized():
        yield from plans
        return

    source = iter(plans)
    world_size = dist.get_world_size()
    window = _positive_plan_window(plan_window)
    chunk = 0
    while True:
        if _debug_enabled():
            _log_debug_plan(
                f"collecting rank-local plans rank={_rank_id()} "
                f"chunk={chunk} window={window}"
            )
        started = perf_counter()
        local = list(islice(source, window))
        elapsed = perf_counter() - started
        counts = plan_counts(len(local), world_size)
        kept = min(counts)
        if _debug_enabled():
            _log_debug_plan(
                "synchronized rank-local plans "
                f"rank={_rank_id()} chunk={chunk} window={window} "
                f"local_count={len(local)} counts={counts} kept={kept} "
                f"collect_seconds={elapsed:.3f}"
            )
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
        if kept < window:
            return
        chunk += 1


def _positive_plan_window(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("distributed plan_window must be an integer.")
    if value <= 0:
        raise ValueError("distributed plan_window must be positive.")
    return value


def _debug_enabled() -> bool:
    value = os.environ.get(_DEBUG_ENV)
    return value is not None and value.lower() not in {"", "0", "false", "no"}


def _log_debug_plan(message: str) -> None:
    _LOGGER.info(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _rank_id() -> int:
    try:
        return dist.get_rank()
    except RuntimeError:
        return int(os.environ.get("RANK", "0"))


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
    return runtime_rank()
