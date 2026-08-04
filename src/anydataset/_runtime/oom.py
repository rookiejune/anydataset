"""CUDA out-of-memory recovery for ordered batch calls.

The module retries recognized OOM failures by recursively splitting an input
sequence. It preserves input order, clears exception-held references before
releasing the CUDA cache, and leaves non-OOM or single-item failures unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar

import torch

from .devices import clear_cuda_cache, release_exception

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
OomCallback = Callable[[int, int, int], None]


def iter_resilient_batch_outputs(
    inputs: Sequence[InputT],
    call: Callable[[Sequence[InputT]], Sequence[OutputT]],
    *,
    on_oom: OomCallback | None = None,
) -> Iterator[OutputT]:
    try:
        yield from call(inputs)
    except Exception as exc:
        if len(inputs) <= 1 or not is_oom_error(exc):
            raise
        release_exception(exc)
        clear_cuda_cache()
        midpoint = len(inputs) // 2
        if on_oom is not None:
            on_oom(len(inputs), midpoint, len(inputs) - midpoint)
        yield from iter_resilient_batch_outputs(
            inputs[:midpoint],
            call,
            on_oom=on_oom,
        )
        yield from iter_resilient_batch_outputs(
            inputs[midpoint:],
            call,
            on_oom=on_oom,
        )


def is_oom_error(error: BaseException) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message
