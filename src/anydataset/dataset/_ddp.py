"""Distributed plan synchronization for cost-aware dataloaders."""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterable, Iterator
from itertools import islice
from time import perf_counter
from typing import Literal, TypeVar

import torch
import torch.distributed as dist

from .._runtime.sharding import runtime_rank


_DEFAULT_PLAN_WINDOW = 32
PLAN_DEBUG_ENV = "ANYDATASET_DEBUG_DDP_PLANS"
_DEBUG_ENV = PLAN_DEBUG_ENV
_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")


def synchronized_plans(
    plans: Iterable[_T],
    *,
    drop_tail: bool,
    mode: Literal["epoch", "window"] = "epoch",
    plan_window: int = _DEFAULT_PLAN_WINDOW,
) -> Iterator[_T]:
    if not dist.is_available() or not dist.is_initialized():
        yield from plans
        return

    sync_mode = _plan_sync_mode(mode)
    world_size = dist.get_world_size()
    if world_size == 1:
        yield from plans
        return
    window = _positive_plan_window(plan_window)
    if sync_mode == "epoch":
        yield from _synchronized_epoch(
            plans,
            drop_tail=drop_tail,
            world_size=world_size,
        )
        return

    yield from _synchronized_windows(
        plans,
        drop_tail=drop_tail,
        world_size=world_size,
        window=window,
    )


def _synchronized_epoch(
    plans: Iterable[_T],
    *,
    drop_tail: bool,
    world_size: int,
) -> Iterator[_T]:
    rank_id = _rank_id()
    if debug_plans_enabled():
        log_debug_plan(f"epoch planning start rank={rank_id}")

    started = perf_counter()
    local: list[_T] = []
    local_error: Exception | None = None
    try:
        local.extend(plans)
    except Exception as error:
        local_error = error
        local.clear()
    planning_elapsed = perf_counter() - started

    local_count = -1 if local_error is not None else len(local)
    sync_started = perf_counter()
    counts = plan_counts(local_count, world_size)
    sync_elapsed = perf_counter() - sync_started
    kept = None if any(count < 0 for count in counts) else min(counts)
    if debug_plans_enabled():
        log_debug_plan(
            "epoch planning completion "
            f"rank={rank_id} local_count={local_count} counts={counts} kept={kept} "
            f"planning_seconds={planning_elapsed:.3f} "
            f"sync_seconds={sync_elapsed:.3f}"
        )

    if local_error is not None:
        raise local_error
    if -1 in counts:
        raise RuntimeError(
            "dataloader detected a remote planning failure: "
            f"rank={rank_id} local_count={local_count} counts={counts}. "
            f"Enable {PLAN_DEBUG_ENV}=1 and inspect dataset planning errors."
        )
    if any(count < 0 for count in counts):
        raise RuntimeError(
            "dataloader received invalid rank-local plan counts: "
            f"rank={rank_id} local_count={local_count} counts={counts}. "
            f"Enable {PLAN_DEBUG_ENV}=1 and inspect dataset length/index groups."
        )

    kept_count = min(counts)
    if kept_count == 0:
        if any(count > 0 for count in counts):
            raise RuntimeError(
                "dataloader rank-local planning produced zero batches on at least "
                "one DDP rank: "
                f"rank={rank_id} local_count={local_count} counts={counts} "
                f"kept={kept_count}. "
                f"Enable {PLAN_DEBUG_ENV}=1 and inspect dataset length/index groups."
            )
        return

    if any(count != kept_count for count in counts):
        if not drop_tail:
            raise RuntimeError("dataloader cannot keep equal rank-local batch counts.")
        if len(local) > kept_count:
            warnings.warn(
                "dataloader dropped rank-local final batches for equal DDP steps.",
                RuntimeWarning,
                stacklevel=2,
            )
    yield from local[:kept_count]


def _synchronized_windows(
    plans: Iterable[_T],
    *,
    drop_tail: bool,
    world_size: int,
    window: int,
) -> Iterator[_T]:
    source = iter(plans)
    chunk = 0
    while True:
        if debug_plans_enabled():
            log_debug_plan(
                f"collecting rank-local plans rank={_rank_id()} "
                f"chunk={chunk} window={window}"
            )
        started = perf_counter()
        local = list(islice(source, window))
        elapsed = perf_counter() - started
        counts = plan_counts(len(local), world_size)
        if any(count < 0 or count > window for count in counts):
            raise RuntimeError(
                "dataloader received invalid rank-local plan counts: "
                f"rank={_rank_id()} chunk={chunk} window={window} "
                f"local_count={len(local)} counts={counts}. "
                f"Enable {PLAN_DEBUG_ENV}=1 and inspect dataset length/index groups."
            )
        kept = min(counts)
        if debug_plans_enabled():
            log_debug_plan(
                "synchronized rank-local plans "
                f"rank={_rank_id()} chunk={chunk} window={window} "
                f"local_count={len(local)} counts={counts} kept={kept} "
                f"collect_seconds={elapsed:.3f}"
            )
        if chunk == 0 and kept == 0 and any(count > 0 for count in counts):
            raise RuntimeError(
                "dataloader rank-local planning produced zero batches on at least "
                "one DDP rank: "
                f"rank={_rank_id()} chunk={chunk} window={window} "
                f"local_count={len(local)} counts={counts} kept={kept}. "
                f"Enable {PLAN_DEBUG_ENV}=1 and inspect dataset length/index groups."
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


def _plan_sync_mode(value: str) -> Literal["epoch", "window"]:
    if value == "epoch":
        return "epoch"
    if value == "window":
        return "window"
    raise ValueError("distributed plan sync mode must be 'epoch' or 'window'.")


def _positive_plan_window(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("distributed plan_window must be an integer.")
    if value <= 0:
        raise ValueError("distributed plan_window must be positive.")
    return value


def debug_plans_enabled() -> bool:
    value = os.environ.get(PLAN_DEBUG_ENV)
    return value is not None and value.lower() not in {"", "0", "false", "no"}


def _debug_enabled() -> bool:
    return debug_plans_enabled()


def log_debug_plan(message: str) -> None:
    _LOGGER.info(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _rank_id() -> int:
    try:
        return dist.get_rank()
    except RuntimeError:
        return int(os.environ.get("RANK", "0"))


def plan_counts(local_count: int, world_size: int) -> tuple[int, ...]:
    device = _collective_device()
    local = torch.tensor([local_count], dtype=torch.int64, device=device)
    gathered = [
        torch.full((1,), -1, dtype=torch.int64, device=device)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, local)
    return tuple(int(value.item()) for value in gathered)


def minimum_rank_length(local_length: int) -> int:
    """Return the shortest map-style length currently visible across DDP ranks."""

    if isinstance(local_length, bool) or not isinstance(local_length, int):
        raise TypeError("local_length must be an integer.")
    if local_length < 0:
        raise ValueError("local_length must be non-negative.")
    if not dist.is_available() or not dist.is_initialized():
        return local_length
    if dist.get_world_size() == 1:
        return local_length
    value = torch.tensor(
        local_length,
        dtype=torch.int64,
        device=_collective_device(),
    )
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return int(value.item())


def _collective_device() -> torch.device:
    backend = str(dist.get_backend()).lower()
    if backend == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def rank() -> tuple[int, int]:
    return runtime_rank()
