"""Low-overhead counters shared by dataset runtime performance logs.

The counters measure stage boundaries owned by AnyDataset. They deliberately do
not inspect provider or predicate internals, device telemetry, or sample data.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


def mean(total: int | float, count: int | float) -> float:
    """Return a zero-safe arithmetic mean."""

    if count <= 0:
        return 0.0
    return total / count


def rate(total: int | float, elapsed: int | float) -> float:
    """Return a zero-safe rate for a wall-clock or service interval."""

    if elapsed <= 0:
        return 0.0
    return total / elapsed


@dataclass
class BatchStats:
    """Accumulate call counts, batch sizes, and boundary service time."""

    calls: int = 0
    samples: int = 0
    batch_size_min: int | None = None
    batch_size_max: int | None = None
    elapsed_seconds: float = 0.0

    def record(self, batch_size: int, elapsed: float) -> None:
        self.calls += 1
        self.samples += batch_size
        self.elapsed_seconds += elapsed
        if self.batch_size_min is None:
            self.batch_size_min = batch_size
        else:
            self.batch_size_min = min(self.batch_size_min, batch_size)
        if self.batch_size_max is None:
            self.batch_size_max = batch_size
        else:
            self.batch_size_max = max(self.batch_size_max, batch_size)

    @property
    def batch_size_mean(self) -> float:
        return mean(self.samples, self.calls)

    @property
    def call_seconds_mean(self) -> float:
        return mean(self.elapsed_seconds, self.calls)

    @property
    def service_samples_per_second(self) -> float:
        return rate(self.samples, self.elapsed_seconds)


@dataclass
class PipelineWorkerStats:
    """Shared reader/operation/writer counters for one pipeline worker."""

    worker: int
    operation: str
    operation_name: str
    requested_batch_size: int
    num_workers: int
    prefetch_factor: int | None
    device: str | None = None
    setup_seconds: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)
    processed_samples: int = 0
    selected_samples: int = 0
    loader: BatchStats = field(default_factory=BatchStats)
    operation_calls: BatchStats = field(default_factory=BatchStats)
    writer: BatchStats = field(default_factory=BatchStats)
    writer_pending: int | None = None
    writer_pending_max: int = 0
    writer_backpressure_seconds: float = 0.0
    oom_count: int = 0
    output_queue_blocked_seconds: float = 0.0

    def record_loader(self, batch_size: int, elapsed: float) -> None:
        self.loader.record(batch_size, elapsed)

    def record_operation(self, batch_size: int, elapsed: float) -> None:
        self.operation_calls.record(batch_size, elapsed)

    def record_scalar_operations(self, calls: int, elapsed: float) -> None:
        if calls <= 0:
            return
        self.operation_calls.calls += calls
        self.operation_calls.samples += calls
        self.operation_calls.elapsed_seconds += elapsed
        self.operation_calls.batch_size_min = 1
        self.operation_calls.batch_size_max = 1

    def record_writer_submit(self, pending: int) -> None:
        self.writer_pending = pending
        self.writer_pending_max = max(self.writer_pending_max, pending)

    def record_writer_complete(
        self,
        samples: int,
        pending: int,
        elapsed: float,
    ) -> None:
        self.writer.record(samples, elapsed)
        self.writer_pending = pending
        self.writer_pending_max = max(self.writer_pending_max, pending)

    def record_writer_backpressure(self, elapsed: float) -> None:
        self.writer_backpressure_seconds += elapsed

    @property
    def split_call_ratio(self) -> float:
        return rate(self.oom_count, self.operation_calls.calls)

    def fields(
        self,
        *,
        status: str,
        error_type: str | None = None,
    ) -> dict[str, object]:
        elapsed = time.perf_counter() - self.started_at
        operation = self.operation
        fields: dict[str, object] = {
            "worker": self.worker,
            "status": status,
            operation: self.operation_name,
            "requested_batch_size": self.requested_batch_size,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "processed_samples": self.processed_samples,
            "selected_samples": self.selected_samples,
            "loader_batches": self.loader.calls,
            "loader_samples": self.loader.samples,
            "loader_batch_size_min": self.loader.batch_size_min,
            "loader_batch_size_mean": self.loader.batch_size_mean,
            "loader_batch_size_max": self.loader.batch_size_max,
            "loader_wait_seconds": self.loader.elapsed_seconds,
            "loader_wait_seconds_mean": self.loader.call_seconds_mean,
            "loader_service_samples_per_second": (
                self.loader.service_samples_per_second
            ),
            f"{operation}_setup_seconds": self.setup_seconds,
            f"{operation}_calls": self.operation_calls.calls,
            f"{operation}_samples": self.operation_calls.samples,
            f"{operation}_batch_size_min": self.operation_calls.batch_size_min,
            f"{operation}_batch_size_mean": self.operation_calls.batch_size_mean,
            f"{operation}_batch_size_max": self.operation_calls.batch_size_max,
            f"{operation}_seconds": self.operation_calls.elapsed_seconds,
            f"{operation}_call_seconds_mean": self.operation_calls.call_seconds_mean,
            f"{operation}_service_samples_per_second": (
                self.operation_calls.service_samples_per_second
            ),
            "writer_jobs": self.writer.calls,
            "writer_samples": self.writer.samples,
            "writer_job_seconds": self.writer.elapsed_seconds,
            "writer_job_seconds_mean": self.writer.call_seconds_mean,
            "writer_service_samples_per_second": (
                self.writer.service_samples_per_second
            ),
            "writer_pending": self.writer_pending,
            "writer_pending_max": self.writer_pending_max,
            "writer_backpressure_seconds": self.writer_backpressure_seconds,
            "oom_count": self.oom_count,
            "split_call_ratio": self.split_call_ratio,
            "output_queue_blocked_seconds": self.output_queue_blocked_seconds,
            "elapsed_seconds": elapsed,
            "worker_elapsed_seconds": elapsed,
            "wall_clock_samples_per_second": rate(
                self.processed_samples,
                elapsed,
            ),
            # Compatibility with the original filter worker summary.
            "samples_per_second": rate(self.processed_samples, elapsed),
        }
        if self.device is not None:
            fields["device"] = self.device
        if operation == "provider":
            fields["provider_load_seconds"] = self.setup_seconds
        if error_type is not None:
            fields["error_type"] = error_type
        return fields


def aggregate_worker_summaries(
    summaries: Sequence[Mapping[str, object]],
    *,
    operation: str,
) -> dict[str, object]:
    """Aggregate ``PipelineWorkerStats.fields()`` dictionaries."""

    operation_samples = _sum_int(summaries, f"{operation}_samples")
    operation_calls = _sum_int(summaries, f"{operation}_calls")
    operation_seconds = _sum_float(summaries, f"{operation}_seconds")
    loader_samples = _sum_int(summaries, "loader_samples")
    loader_batches = _sum_int(summaries, "loader_batches")
    writer_samples = _sum_int(summaries, "writer_samples")
    writer_jobs = _sum_int(summaries, "writer_jobs")
    writer_seconds = _sum_float(summaries, "writer_job_seconds")
    setup_seconds = _sum_float(summaries, f"{operation}_setup_seconds")
    fields: dict[str, object] = {
        "worker_count": len(summaries),
        "processed_samples": _sum_int(summaries, "processed_samples"),
        "selected_samples": _sum_int(summaries, "selected_samples"),
        "loader_batches": loader_batches,
        "loader_samples": loader_samples,
        "loader_batch_size_min": _summary_min(summaries, "loader_batch_size_min"),
        "loader_batch_size_mean": mean(loader_samples, loader_batches),
        "loader_batch_size_max": _summary_max(summaries, "loader_batch_size_max"),
        "loader_wait_seconds": _sum_float(summaries, "loader_wait_seconds"),
        "loader_wait_seconds_mean": mean(
            _sum_float(summaries, "loader_wait_seconds"), loader_batches
        ),
        "loader_service_samples_per_second": rate(
            loader_samples, _sum_float(summaries, "loader_wait_seconds")
        ),
        f"{operation}_setup_seconds": setup_seconds,
        f"{operation}_setup_seconds_max": _summary_max_float(
            summaries, f"{operation}_setup_seconds"
        ),
        f"{operation}_calls": operation_calls,
        f"{operation}_samples": operation_samples,
        f"{operation}_batch_size_min": _summary_min(
            summaries, f"{operation}_batch_size_min"
        ),
        f"{operation}_batch_size_mean": mean(operation_samples, operation_calls),
        f"{operation}_batch_size_max": _summary_max(
            summaries, f"{operation}_batch_size_max"
        ),
        f"{operation}_seconds": operation_seconds,
        f"{operation}_call_seconds_mean": mean(operation_seconds, operation_calls),
        f"{operation}_service_samples_per_second": rate(
            operation_samples, operation_seconds
        ),
        "writer_jobs": writer_jobs,
        "writer_samples": writer_samples,
        "writer_job_seconds": writer_seconds,
        "writer_job_seconds_mean": mean(writer_seconds, writer_jobs),
        "writer_service_samples_per_second": rate(writer_samples, writer_seconds),
        "writer_pending_max": _summary_max_value(summaries, "writer_pending_max"),
        "writer_backpressure_seconds": _sum_float(
            summaries, "writer_backpressure_seconds"
        ),
        "oom_count": _sum_int(summaries, "oom_count"),
        "output_queue_blocked_seconds": _sum_float(
            summaries, "output_queue_blocked_seconds"
        ),
        "worker_elapsed_seconds_sum": _sum_float(summaries, "worker_elapsed_seconds"),
        "worker_elapsed_seconds_max": _summary_max_float(
            summaries, "worker_elapsed_seconds"
        ),
    }
    if operation == "provider":
        fields["provider_load_seconds"] = setup_seconds
        fields["provider_load_seconds_max"] = _summary_max_float(
            summaries, "provider_load_seconds"
        )
    return fields


def _sum_int(summaries: Sequence[Mapping[str, object]], key: str) -> int:
    return sum(
        int(value) for summary in summaries if type(value := summary.get(key, 0)) is int
    )


def _sum_float(summaries: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(
        float(value)
        for summary in summaries
        if isinstance((value := summary.get(key, 0.0)), (int, float))
        and not isinstance(value, bool)
    )


def _summary_max_value(summaries: Sequence[Mapping[str, object]], key: str) -> int:
    values = tuple(
        value for summary in summaries if type(value := summary.get(key)) is int
    )
    return max(values) if values else 0


def _summary_min(summaries: Sequence[Mapping[str, object]], key: str) -> int | None:
    values = tuple(
        value for summary in summaries if type(value := summary.get(key)) is int
    )
    return min(values) if values else None


def _summary_max(summaries: Sequence[Mapping[str, object]], key: str) -> int | None:
    values = tuple(
        value for summary in summaries if type(value := summary.get(key)) is int
    )
    return max(values) if values else None


def _summary_max_float(summaries: Sequence[Mapping[str, object]], key: str) -> float:
    values = tuple(
        float(value)
        for summary in summaries
        if isinstance((value := summary.get(key)), (int, float))
        and not isinstance(value, bool)
    )
    return max(values) if values else 0.0
