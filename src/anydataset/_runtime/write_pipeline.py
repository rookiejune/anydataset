"""Bounded background write pipeline for dataset-wide scans.

The module owns only submission, backpressure, worker lifetime, and exception
propagation. Callers keep domain-specific fragment, partition, and manifest
formats in their own modules.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from queue import Empty, SimpleQueue
from typing import Generic, Literal, TypeVar

from .parallel import StartMethod, multiprocessing_context, validate_process_value
from .._validation import non_negative_int, optional_positive_int

T = TypeVar("T")
WriteBackend = Literal["thread", "process"]


class BackgroundWriteSink(Generic[T]):
    def __init__(
        self,
        write: Callable[[T], None],
        *,
        workers: int,
        start_method: StartMethod,
        backend: WriteBackend = "thread",
        max_pending: int | None = None,
        on_submit: Callable[[T, int], None] | None = None,
        on_complete: Callable[[T, int, float], None] | None = None,
        on_backpressure: Callable[[float], None] | None = None,
    ) -> None:
        self.write = write
        self.workers = non_negative_int("write_workers", workers)
        self.max_pending = optional_positive_int("max_pending", max_pending)
        self.start_method: StartMethod = start_method
        self.backend = backend
        self.on_submit = on_submit
        self.on_complete = on_complete
        self.on_backpressure = on_backpressure
        self._executor: Executor | None = None
        self._pending: dict[Future[None], tuple[T, float]] = {}
        self._ready_queue: SimpleQueue[Future[None]] = SimpleQueue()
        self._error: BaseException | None = None
        self._closed = False

    def __enter__(self) -> BackgroundWriteSink[T]:
        if self.workers == 0:
            return self
        if self.backend == "thread":
            self._executor = ThreadPoolExecutor(max_workers=self.workers)
            return self
        if self.backend != "process":
            raise ValueError(f"Unsupported write backend: {self.backend!r}.")
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
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError("background write sink is already closed.")
        executor = self._executor
        if executor is None:
            start = time.perf_counter()
            self._on_submit(job, 1)
            try:
                self.write(job)
            except BaseException as exc:
                self._remember_error(exc)
                raise
            self._on_complete(job, 0, time.perf_counter() - start)
            return
        backpressure_started: float | None = None
        while len(self._pending) >= self._pending_limit:
            if backpressure_started is None:
                backpressure_started = time.perf_counter()
            self._drain_one()
            self._raise_error()
        if backpressure_started is not None:
            self._on_backpressure(time.perf_counter() - backpressure_started)
        future = executor.submit(self.write, job)
        self._pending[future] = (job, time.perf_counter())
        future.add_done_callback(self._mark_ready)
        self._on_submit(job, len(self._pending))

    def flush(self) -> None:
        """Wait for submitted writes while keeping the sink open for reuse."""

        self._drain_ready()
        if self._closed:
            self._raise_error()
            return
        while self._pending:
            self._drain_ready()
            if not self._pending:
                break
            self._drain_one()
        self._raise_error()

    def raise_if_failed(self) -> None:
        """Raise the first completed write failure without waiting for pending jobs."""

        self._drain_ready()
        self._raise_error()

    def close(self) -> None:
        if self._closed:
            self._raise_error()
            return
        try:
            self.flush()
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
        while self._pending:
            future = self._ready_queue.get()
            if future in self._pending:
                self._complete(future)
                return

    def _complete(self, future: Future[None]) -> None:
        job, start = self._pending.pop(future)
        try:
            future.result()
        except BaseException as exc:
            self._remember_error(exc)
            return
        self._on_complete(job, len(self._pending), time.perf_counter() - start)

    def _drain_ready(self) -> None:
        while True:
            try:
                future = self._ready_queue.get_nowait()
            except Empty:
                return
            if future in self._pending:
                self._complete(future)

    def _mark_ready(self, future: Future[None]) -> None:
        self._ready_queue.put(future)

    def _remember_error(self, error: BaseException) -> None:
        if self._error is None:
            self._error = error

    def _raise_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _on_submit(self, job: T, pending: int) -> None:
        if self.on_submit is not None:
            self.on_submit(job, pending)

    def _on_complete(self, job: T, pending: int, elapsed: float) -> None:
        if self.on_complete is not None:
            self.on_complete(job, pending, elapsed)

    def _on_backpressure(self, elapsed: float) -> None:
        if self.on_backpressure is not None:
            self.on_backpressure(elapsed)
