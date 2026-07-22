from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .._parallel import can_select_indexes, validate_process_value
from .._progress import Progress, ProgressDashboard
from .._resume import (
    dataset_sample_count,
    indexes_complete,
    log_resume_summary,
    missing_indexes,
)
from .._write_pipeline import BackgroundWriteSink
from ..runtime import Runtime
from .collect import collect_ranges, collect_ranges_parallel
from ._identity import FilterBase
from .resume import (
    completed_filter_indexes,
    iter_filter_fragment_chunks,
    prepare_filter_resume_dir,
    write_filter_fragment,
)
from .storage import MetricsWriter, PartitionWriter
from .types import DatasetFactory, _FilterChunk, _FilterMetricsRow

if TYPE_CHECKING:
    from .api import FilterRule


_PROGRESS_STAGES = ("scan", "writer")


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
        writer.write(write_workers=write_workers, write_prefetch=write_prefetch)
        completed = completed_filter_indexes(resume_dir, expected=expected)
    if not indexes_complete(completed, expected):
        raise RuntimeError("Filter resume fragments do not cover all samples.")
    replay_filter_resume_fragments(
        path,
        resume_dir,
        metrics=metrics,
        max_shard_samples=max_shard_samples,
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
    completed: frozenset[int]
    missing: Sequence[int]
    worker_timeout: float | None

    def write(self, *, write_workers: int, write_prefetch: int | None) -> None:
        with ProgressDashboard(
            desc="filter samples",
            total=len(self.dataset),
            count_stage="writer",
            initial=len(self.completed),
            stages=_PROGRESS_STAGES,
        ) as progress:
            sink = BackgroundWriteSink(
                write_filter_fragment_job,
                workers=write_workers,
                max_pending=write_prefetch,
                start_method=self.runtime.writer_worker_start_method,
                on_submit=lambda job, pending: progress.put(
                    Progress(0, 0, False, None, stage="writer", pending=pending)
                ),
                on_complete=lambda job, pending, elapsed: progress.put(
                    Progress(
                        0,
                        len(job.scan_indexes),
                        False,
                        None,
                        stage="writer",
                        elapsed=elapsed,
                        pending=pending,
                    )
                ),
            )
            with sink:
                self.write_jobs(sink, progress)

    def write_jobs(
        self,
        sink: BackgroundWriteSink[FilterFragmentJob],
        progress: ProgressDashboard,
    ) -> None:
        for chunk in self._chunks(progress):
            sink.submit(self._job(chunk))

    def _chunks(self, progress: ProgressDashboard) -> Iterable[_FilterChunk]:
        use_map_style_loader = can_select_indexes(self.dataset)
        if len(self.devices) == 1 or len(self.dataset) == 0:
            chunks = collect_ranges(
                self.dataset,
                self.rule.factory,
                self.devices[0],
                self.metrics,
                self.commit_samples,
                skip_indexes=self.completed,
                sample_indexes=self.missing if use_map_style_loader else None,
                dataset_factory=self.dataset_factory,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                runtime=self.runtime,
            )
            for chunk in chunks:
                progress.put(
                    Progress(
                        0,
                        len(filter_chunk_indexes(chunk)),
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
            skip_indexes=self.completed,
            sample_indexes=self.missing if use_map_style_loader else None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            runtime=self.runtime,
            use_map_style_loader=use_map_style_loader,
            worker_timeout=self.worker_timeout,
        )
        for chunk in chunks:
            progress.put(
                Progress(
                    0,
                    len(filter_chunk_indexes(chunk)),
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
        int(index)
        for positions in chunk.partitions.values()
        for index in positions
    }
    return tuple(sorted(indexes))


def parallel_dataset_factory(factory: DatasetFactory, runtime: Runtime) -> DatasetFactory:
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
    return _FilterChunk(
        partitions={
            label: tuple(global_index(position) for position in positions)
            for label, positions in chunk.partitions.items()
        },
        metrics=tuple(
            _FilterMetricsRow(
                index=global_index(row.index),
                label=row.label,
                metrics=row.metrics,
            )
            for row in chunk.metrics
        ),
    )
