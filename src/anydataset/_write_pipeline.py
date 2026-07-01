from __future__ import annotations

"""Bounded background write pipeline for dataset-wide scans.

The module owns only submission, backpressure, worker lifetime, and exception
propagation. Callers keep domain-specific fragment, partition, and manifest
formats in their own modules.
"""

import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Generic, TypeVar

from ._parallel import StartMethod, multiprocessing_context, validate_process_value
from ._validation import non_negative_int, optional_positive_int

T = TypeVar("T")


class BackgroundWriteSink(Generic[T]):
    def __init__(
        self,
        write: Callable[[T], None],
        *,
        workers: int,
        start_method: StartMethod,
        max_pending: int | None = None,
        on_submit: Callable[[T, int], None] | None = None,
        on_complete: Callable[[T, int, float], None] | None = None,
    ) -> None:
        self.write = write
        self.workers = non_negative_int("write_workers", workers)
        self.max_pending = optional_positive_int("max_pending", max_pending)
        self.start_method = start_method
        self.on_submit = on_submit
        self.on_complete = on_complete
        self._executor: ProcessPoolExecutor | None = None
        self._pending: deque[tuple[T, Future[None], float]] = deque()
        self._closed = False

    def __enter__(self) -> BackgroundWriteSink[T]:
        if self.workers == 0:
            return self
        validate_process_value(
            "write",
            self.write,
            context="background writes",
            start_method=self.start_method,
        )
        context = multiprocessing_context(self.start_method)
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        self.abort()

    def submit(self, job: T) -> None:
        if self._closed:
            raise RuntimeError("background write sink is already closed.")
        executor = self._executor
        if executor is None:
            start = time.perf_counter()
            self._on_submit(job, 1)
            self.write(job)
            self._on_complete(job, 0, time.perf_counter() - start)
            return
        self._drain_ready()
        while len(self._pending) >= self._pending_limit:
            self._drain_one()
        future = executor.submit(self.write, job)
        self._pending.append((job, future, time.perf_counter()))
        self._on_submit(job, len(self._pending))

    def close(self) -> None:
        if self._closed:
            return
        try:
            while self._pending:
                self._drain_ready()
                if not self._pending:
                    break
                self._drain_one()
        finally:
            if self._executor is not None:
                self._executor.shutdown()
                self._executor = None
            self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self._pending.clear()
        if self._executor is not None:
            self._executor.shutdown(cancel_futures=True)
            self._executor = None
        self._closed = True

    @property
    def _pending_limit(self) -> int:
        if self.max_pending is not None:
            return self.max_pending
        return max(1, self.workers * 2)

    def _drain_one(self) -> None:
        job, future, start = self._pending.popleft()
        future.result()
        self._on_complete(job, len(self._pending), time.perf_counter() - start)

    def _drain_ready(self) -> None:
        ready: deque[tuple[T, Future[None], float]] = deque()
        while self._pending:
            job, future, start = self._pending.popleft()
            if future.done():
                future.result()
                self._on_complete(
                    job,
                    len(self._pending) + len(ready),
                    time.perf_counter() - start,
                )
                continue
            ready.append((job, future, start))
        self._pending = ready

    def _on_submit(self, job: T, pending: int) -> None:
        if self.on_submit is not None:
            self.on_submit(job, pending)

    def _on_complete(self, job: T, pending: int, elapsed: float) -> None:
        if self.on_complete is not None:
            self.on_complete(job, pending, elapsed)
