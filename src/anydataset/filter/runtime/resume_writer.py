from __future__ import annotations

import time
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

from ..._runtime.parallel import can_select_indexes, validate_process_value
from ..._runtime.logging import write_event, write_info, write_warning
from ..._runtime.progress import Progress, ProgressDashboard
from ..._runtime.resume import (
    dataset_sample_count,
    indexes_complete,
    log_resume_summary,
    missing_indexes,
)
from ..._runtime.write_pipeline import BackgroundWriteSink
from ...runtime import Runtime
from .collect import _FilterCollectionStats, collect_ranges, collect_ranges_parallel
from ..cache.identity import FilterBase
from ..cache.resume import (
    completed_filter_indexes,
    iter_filter_fragment_chunks,
    prepare_filter_resume_dir,
    write_filter_fragment,
)
from ..cache.storage import MetricsWriter, PartitionWriter
from ..types import DatasetFactory, _FilterChunk, _FilterMetricsRow

if TYPE_CHECKING:
    from ..api import FilterRule


_PROGRESS_STAGES = ("scan", "writer")
_FILTER_PROGRESS_EVENT_INTERVAL = 30.0


@dataclass
class _FilterRunStats:
    expected_samples: int
    resumed_samples: int
    collection: _FilterCollectionStats = field(
        default_factory=_FilterCollectionStats
    )
    started_at: float = field(default_factory=time.perf_counter)
    scan_samples: int = 0
    writer_samples: int = 0
    writer_jobs: int = 0
    writer_elapsed_seconds: float = 0.0
    writer_pending: int = 0
    writer_pending_max: int = 0
    writer_backpressure_seconds: float = 0.0
    fragment_seconds: float = 0.0
    replay_seconds: float = 0.0
    _last_progress_event_at: float = field(init=False)

    def __post_init__(self) -> None:
        self._last_progress_event_at = self.started_at

    @property
    def target_samples(self) -> int:
        return max(0, self.expected_samples - self.resumed_samples)

    def record_scan(self, samples: int) -> None:
        self.scan_samples += samples
        self.progress()

    def record_writer_submit(self, pending: int) -> None:
        self.writer_jobs += 1
        self.writer_pending = pending
        self.writer_pending_max = max(self.writer_pending_max, pending)
        self.progress()

    def record_writer_complete(
        self, samples: int, pending: int, elapsed: float
    ) -> None:
        self.writer_samples += samples
        self.writer_elapsed_seconds += elapsed
        self.writer_pending = pending
        self.progress()

    def record_writer_backpressure(self, elapsed: float) -> None:
        self.writer_backpressure_seconds += elapsed

    def progress(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if (
            not force
            and now - self._last_progress_event_at
            < _FILTER_PROGRESS_EVENT_INTERVAL
        ):
            return
        self._last_progress_event_at = now
        elapsed = now - self.started_at
        write_event(
            "filter",
            "filter_progress",
            {
                "expected_samples": self.expected_samples,
                "resumed_samples": self.resumed_samples,
                "target_samples": self.target_samples,
                "scan_samples": self.scan_samples,
                "scan_samples_per_second": _rate(self.scan_samples, elapsed),
                "writer_samples": self.writer_samples,
                "writer_samples_per_second": _rate(
                    self.writer_samples, elapsed
                ),
                "writer_pending": self.writer_pending,
                "writer_pending_max": self.writer_pending_max,
                "writer_backpressure_seconds": self.writer_backpressure_seconds,
                "elapsed_seconds": elapsed,
            },
        )


def _log_filter_run_summary(
    stats: _FilterRunStats,
    *,
    status: str,
    error_type: str | None,
    devices: Sequence[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    write_workers: int,
    write_prefetch: int | None,
) -> None:
    elapsed = time.perf_counter() - stats.started_at
    collection = stats.collection.fields()
    predicate_calls = cast(int, collection["predicate_calls"])
    predicate_seconds = cast(float, collection["predicate_seconds"])
    loader_batches = cast(int, collection["loader_batches"])
    loader_wait_seconds = cast(float, collection["loader_wait_seconds"])
    oom_count = cast(int, collection["oom_count"])
    fields: dict[str, object] = {
        "status": status,
        "expected_samples": stats.expected_samples,
        "resumed_samples": stats.resumed_samples,
        "target_samples": stats.target_samples,
        "scan_samples": stats.scan_samples,
        "scan_samples_per_second": _rate(stats.scan_samples, elapsed),
        "writer_samples": stats.writer_samples,
        "writer_samples_per_second": _rate(stats.writer_samples, elapsed),
        "completed_samples": stats.resumed_samples + stats.writer_samples,
        "writer_jobs": stats.writer_jobs,
        "writer_job_seconds": stats.writer_elapsed_seconds,
        "writer_job_seconds_mean": _rate(
            stats.writer_elapsed_seconds, stats.writer_jobs
        ),
        "writer_pending": stats.writer_pending,
        "writer_pending_max": stats.writer_pending_max,
        "writer_backpressure_seconds": stats.writer_backpressure_seconds,
        "fragment_seconds": stats.fragment_seconds,
        "replay_seconds": stats.replay_seconds,
        "elapsed_seconds": elapsed,
        "devices": list(devices),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "write_workers": write_workers,
        "write_prefetch": write_prefetch,
        **collection,
        "split_call_ratio": _rate(oom_count, predicate_calls),
        "predicate_call_seconds_mean": _rate(
            predicate_seconds, predicate_calls
        ),
        "loader_wait_seconds_mean": _rate(
            loader_wait_seconds,
            loader_batches,
        ),
    }
    if error_type is not None:
        fields["error_type"] = error_type
    message = (
        "filter run summary: "
        f"status={status!r} scan={stats.scan_samples} "
        f"writer={stats.writer_samples} workers={collection['worker_count']} "
        f"predicate_calls={predicate_calls} oom_count={oom_count} "
        f"split/call={_rate(oom_count, predicate_calls):.6f} "
        f"elapsed={elapsed:.3f}s"
    )
    write = write_info if status == "complete" else write_warning
    write(
        "filter",
        message,
        event="filter_run_summary",
        fields=fields,
    )


def _rate(total: int | float, elapsed: int | float) -> float:
    if elapsed <= 0:
        return 0.0
    return total / elapsed


def write_partitions(
    path: Path,
    dataset: FilterBase,
    rule: FilterRule,
    *,
    cache_path: Path,
    metadata: Mapping[str, object],
    metrics: bool,
    devices: tuple[str, ...],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    commit_samples: int,
    max_shard_samples: int | None,
    write_workers: int,
    write_prefetch: int | None,
    worker_timeout: float | None,
    runtime: Runtime,
    dataset_factory: DatasetFactory,
) -> None:
    resume_dir = prepare_filter_resume_dir(cache_path, metadata, metrics=metrics)
    expected = dataset_sample_count(dataset, context="filter")
    completed = completed_filter_indexes(resume_dir, expected=expected)
    stats = _FilterRunStats(
        expected_samples=expected,
        resumed_samples=len(completed),
    )
    status = "complete"
    error_type: str | None = None
    try:
        if not indexes_complete(completed, expected):
            missing = missing_indexes(completed, expected)
            use_map_style_loader = can_select_indexes(dataset)
            log_resume_summary(
                "filter",
                expected=expected,
                completed_count=len(completed),
                missing=missing,
                use_map_style_loader=use_map_style_loader,
            )
            writer = _FilterResumeFragmentWriter(
                path=resume_dir,
                dataset=dataset,
                rule=rule,
                metrics=metrics,
                devices=devices,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                commit_samples=commit_samples,
                runtime=runtime,
                dataset_factory=dataset_factory,
                completed=completed,
                missing=missing,
                worker_timeout=worker_timeout,
            )
            fragment_started = time.perf_counter()
            try:
                writer.write(
                    write_workers=write_workers,
                    write_prefetch=write_prefetch,
                    stats=stats,
                )
            finally:
                stats.fragment_seconds = time.perf_counter() - fragment_started
            completed = completed_filter_indexes(resume_dir, expected=expected)
        if not indexes_complete(completed, expected):
            raise RuntimeError("Filter resume fragments do not cover all samples.")
        replay_started = time.perf_counter()
        replay_filter_resume_fragments(
            path,
            resume_dir,
            metrics=metrics,
            max_shard_samples=max_shard_samples,
        )
        stats.replay_seconds = time.perf_counter() - replay_started
    except (GeneratorExit, KeyboardInterrupt) as exc:
        status = "interrupted"
        error_type = type(exc).__name__
        raise
    except BaseException as exc:
        status = "failed"
        error_type = type(exc).__name__
        raise
    finally:
        stats.progress(force=True)
        _log_filter_run_summary(
            stats,
            status=status,
            error_type=error_type,
            devices=devices,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            write_workers=write_workers,
            write_prefetch=write_prefetch,
        )


@dataclass(frozen=True)
class _FilterResumeFragmentWriter:
    path: Path
    dataset: FilterBase
    rule: FilterRule
    metrics: bool
    devices: tuple[str, ...]
    batch_size: int
    num_workers: int
    prefetch_factor: int | None
    commit_samples: int
    runtime: Runtime
    dataset_factory: DatasetFactory
    completed: Collection[int]
    missing: Sequence[int]
    worker_timeout: float | None

    def write(
        self,
        *,
        write_workers: int,
        write_prefetch: int | None,
        stats: _FilterRunStats,
    ) -> None:
        with ProgressDashboard(
            desc="filter samples",
            total=len(self.dataset),
            count_stage="writer",
            initial=len(self.completed),
            stages=_PROGRESS_STAGES,
        ) as progress:
            def on_submit(_job: FilterFragmentJob, pending: int) -> None:
                stats.record_writer_submit(pending)
                progress.put(
                    Progress(0, 0, False, None, stage="writer", pending=pending)
                )

            def on_complete(
                job: FilterFragmentJob, pending: int, elapsed: float
            ) -> None:
                samples = len(job.scan_indexes)
                stats.record_writer_complete(samples, pending, elapsed)
                progress.put(
                    Progress(
                        0,
                        samples,
                        False,
                        None,
                        stage="writer",
                        elapsed=elapsed,
                        pending=pending,
                    )
                )

            sink = BackgroundWriteSink(
                write_filter_fragment_job,
                workers=write_workers,
                max_pending=write_prefetch,
                start_method=self.runtime.writer_worker_start_method,
                on_submit=on_submit,
                on_complete=on_complete,
                on_backpressure=stats.record_writer_backpressure,
            )
            with sink:
                self.write_jobs(sink, progress, stats)

    def write_jobs(
        self,
        sink: BackgroundWriteSink[FilterFragmentJob],
        progress: ProgressDashboard,
        stats: _FilterRunStats,
    ) -> None:
        for chunk in self._chunks(progress, stats):
            sink.submit(self._job(chunk))

    def _chunks(
        self,
        progress: ProgressDashboard,
        stats: _FilterRunStats | None = None,
    ) -> Iterable[_FilterChunk]:
        use_map_style_loader = can_select_indexes(self.dataset)
        skip_indexes: Collection[int] = () if use_map_style_loader else self.completed
        if len(self.devices) == 1 or len(self.dataset) == 0:
            chunks = collect_ranges(
                self.dataset,
                self.rule.factory,
                self.devices[0],
                self.metrics,
                self.commit_samples,
                skip_indexes=skip_indexes,
                sample_indexes=self.missing if use_map_style_loader else None,
                dataset_factory=self.dataset_factory,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                runtime=self.runtime,
                collection_stats=None if stats is None else stats.collection,
            )
            for chunk in chunks:
                samples = len(filter_chunk_indexes(chunk))
                if stats is not None:
                    stats.record_scan(samples)
                progress.put(
                    Progress(
                        0,
                        samples,
                        False,
                        None,
                        stage="scan",
                    )
                )
                yield chunk
            return

        factory = parallel_dataset_factory(self.dataset_factory, self.runtime)
        chunks = collect_ranges_parallel(
            factory,
            self.rule.factory,
            self.devices,
            self.metrics,
            self.commit_samples,
            sample_count=len(self.dataset),
            skip_indexes=skip_indexes,
            sample_indexes=self.missing if use_map_style_loader else None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            runtime=self.runtime,
            use_map_style_loader=use_map_style_loader,
            worker_timeout=self.worker_timeout,
            collection_stats=None if stats is None else stats.collection,
        )
        for chunk in chunks:
            samples = len(filter_chunk_indexes(chunk))
            if stats is not None:
                stats.record_scan(samples)
            progress.put(
                Progress(
                    0,
                    samples,
                    False,
                    None,
                    stage="scan",
                )
            )
            yield chunk

    def _job(self, chunk: _FilterChunk) -> FilterFragmentJob:
        return FilterFragmentJob(
            path=self.path,
            scan_indexes=filter_chunk_indexes(chunk),
            chunk=global_filter_chunk(self.dataset, chunk),
        )


@dataclass(frozen=True)
class FilterFragmentJob:
    path: Path
    scan_indexes: tuple[int, ...]
    chunk: _FilterChunk


def write_filter_fragment_job(job: FilterFragmentJob) -> None:
    write_filter_fragment(job.path, job.scan_indexes, job.chunk)


def replay_filter_resume_fragments(
    path: Path,
    resume_dir: Path,
    *,
    metrics: bool,
    max_shard_samples: int | None,
) -> None:
    writer = PartitionWriter(path, max_shard_samples=max_shard_samples)
    metrics_writer = (
        MetricsWriter(path / "metrics", max_shard_samples=max_shard_samples)
        if metrics
        else None
    )
    try:
        for chunk in iter_filter_fragment_chunks(resume_dir, metrics=metrics):
            write_filter_chunk(writer, metrics_writer, chunk, metrics=metrics)
        writer.close()
        if metrics_writer is not None:
            metrics_writer.close()
    except Exception:
        writer.abort()
        if metrics_writer is not None:
            metrics_writer.abort()
        raise


def filter_chunk_indexes(chunk: _FilterChunk) -> tuple[int, ...]:
    indexes = {
        int(index) for positions in chunk.partitions.values() for index in positions
    }
    return tuple(sorted(indexes))


def parallel_dataset_factory(
    factory: DatasetFactory, runtime: Runtime
) -> DatasetFactory:
    validate_process_value(
        "dataset_factory",
        factory,
        context="multi-device filtering",
        start_method=runtime.process_start_method,
    )
    return factory


def write_filter_chunk(
    writer: PartitionWriter,
    metrics_writer: MetricsWriter | None,
    chunk: _FilterChunk,
    *,
    metrics: bool,
) -> None:
    writer.write_partitions(chunk.partitions)
    if metrics:
        if metrics_writer is None:
            raise RuntimeError("metrics writer was not initialized.")
        metrics_writer.write_rows(chunk.metrics)


def global_filter_chunk(dataset: FilterBase, chunk: _FilterChunk) -> _FilterChunk:
    global_index = getattr(dataset, "global_index", None)
    if not callable(global_index):
        return chunk
    index = cast(Callable[[int], int], global_index)
    return _FilterChunk(
        partitions={
            label: tuple(index(position) for position in positions)
            for label, positions in chunk.partitions.items()
        },
        metrics=tuple(
            _FilterMetricsRow(
                index=index(row.index),
                label=row.label,
                metrics=row.metrics,
            )
            for row in chunk.metrics
        ),
    )
