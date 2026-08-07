from __future__ import annotations

import multiprocessing
import os
import queue
import time
import traceback
from array import array
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from ..._runtime.logging import (
    run_logs_dir,
    use_run_logs_dir,
    worker_logger,
    write_info,
    write_warning,
)
from ..._runtime.oom import iter_resilient_batch_outputs
from ..._runtime.performance import (
    PipelineWorkerStats,
    aggregate_worker_summaries,
)
from ..._runtime.parallel import (
    DeviceWorker,
    ProcessHandle,
    multiprocessing_context,
    restore_environment,
    sample_index_loader,
    set_single_worker_environment,
    set_worker_environment,
    validate_process_value,
    worker_configs,
)
from ..._runtime.progress import write_progress_message
from ...runtime import Runtime
from ...types import Sample
from ..rules import label
from ..cache.storage import validate_metrics
from ..types import (
    BatchFilterPredicate,
    DatasetFactory,
    FilterDecision,
    FilterFactory,
    FilterPredicate,
    FilterOutput,
    JsonValue,
    _FilterChunk,
    _FilterDecision,
    _FilterMetricsRow,
)

_DONE = "__done__"
_WORKER_QUEUE_SIZE = 2
_PARALLEL_WORKER_COMMIT_SAMPLES = 8_192
_WORKER_SHUTDOWN_TIMEOUT = 5.0


@dataclass(frozen=True)
class _FilterRow:
    index: int
    label: str
    metrics: Mapping[str, JsonValue] | None


@dataclass(frozen=True)
class _IndexedFilterChunk:
    rank: int
    rows: Sequence[_FilterRow]


@dataclass(frozen=True)
class _FilterWorkerConfig:
    worker: DeviceWorker
    batch_size: int
    num_workers: int
    prefetch_factor: int | None
    runtime: Runtime
    sample_count: int
    use_map_style_loader: bool
    skip_indexes: Collection[int]
    sample_indexes: Sequence[int] | None
    logs_dir: Path
    worker_logs_dir: Path


class _FilterWorkerStats(PipelineWorkerStats):
    """Filter compatibility facade over the shared pipeline counters."""

    def __init__(
        self,
        *,
        worker: int,
        predicate: str,
        requested_batch_size: int,
        num_workers: int,
        prefetch_factor: int | None,
        predicate_setup_seconds: float = 0.0,
    ) -> None:
        super().__init__(
            worker=worker,
            operation="predicate",
            operation_name=predicate,
            requested_batch_size=requested_batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            setup_seconds=predicate_setup_seconds,
        )

    predicate = property(lambda self: self.operation_name)
    predicate_setup_seconds = property(lambda self: self.setup_seconds)
    predicate_calls = property(lambda self: self.operation_calls.calls)
    predicate_samples = property(lambda self: self.operation_calls.samples)
    predicate_batch_size_min = property(
        lambda self: self.operation_calls.batch_size_min
    )
    predicate_batch_size_max = property(
        lambda self: self.operation_calls.batch_size_max
    )
    predicate_seconds = property(lambda self: self.operation_calls.elapsed_seconds)
    loader_batches = property(lambda self: self.loader.calls)
    loader_samples = property(lambda self: self.loader.samples)
    loader_batch_size_min = property(lambda self: self.loader.batch_size_min)
    loader_batch_size_max = property(lambda self: self.loader.batch_size_max)
    loader_wait_seconds = property(lambda self: self.loader.elapsed_seconds)

    @property
    def oom_splits(self) -> int:
        return self.oom_count

    @oom_splits.setter
    def oom_splits(self, value: int) -> None:
        self.oom_count = value

    def record_loader_batch(self, batch_size: int, elapsed: float) -> None:
        self.record_loader(batch_size, elapsed)

    def record_predicate_call(self, batch_size: int, elapsed: float) -> None:
        self.record_operation(batch_size, elapsed)

    def record_scalar_predicates(self, calls: int, elapsed: float) -> None:
        self.record_scalar_operations(calls, elapsed)


@dataclass
class _FilterCollectionStats:
    workers: dict[int, Mapping[str, object]] = field(default_factory=dict)

    def add(self, fields: Mapping[str, object]) -> None:
        worker = fields.get("worker")
        if not isinstance(worker, int):
            raise TypeError("filter worker summary must contain an integer worker.")
        self.workers[worker] = dict(fields)

    def fields(self) -> dict[str, object]:
        summaries = tuple(self.workers.values())
        fields = aggregate_worker_summaries(summaries, operation="predicate")
        # Filter fragment writes run in the parent and are reported by
        # _FilterRunStats rather than by predicate workers.
        for key in tuple(fields):
            if key.startswith("writer_"):
                del fields[key]
        return fields


def collect_ranges(
    dataset,
    factory: FilterFactory,
    device: str,
    metrics: bool,
    commit_samples: int,
    *,
    skip_indexes: Collection[int] = frozenset(),
    sample_indexes: Sequence[int] | None = None,
    dataset_factory: DatasetFactory,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    runtime: Runtime,
    collection_stats: _FilterCollectionStats | None = None,
) -> Iterable[_FilterChunk]:
    env = set_single_worker_environment(device, device_env="ANYDATASET_FILTER_DEVICE")
    try:
        setup_started = time.perf_counter()
        predicate = factory()
        setup_elapsed = time.perf_counter() - setup_started
        yield from collect_ranges_sequential(
            dataset,
            predicate,
            metrics,
            commit_samples,
            skip_indexes=skip_indexes,
            sample_indexes=sample_indexes,
            dataset_factory=dataset_factory,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            runtime=runtime,
            predicate_setup_seconds=setup_elapsed,
            collection_stats=collection_stats,
        )
    finally:
        restore_environment(env)


def collect_ranges_sequential(
    dataset,
    predicate: FilterPredicate,
    write_metrics: bool,
    commit_samples: int,
    *,
    skip_indexes: Collection[int] = frozenset(),
    sample_indexes: Sequence[int] | None = None,
    dataset_factory: DatasetFactory,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    runtime: Runtime,
    predicate_setup_seconds: float = 0.0,
    collection_stats: _FilterCollectionStats | None = None,
) -> Iterable[_FilterChunk]:
    partitions: dict[str, array[int]] = {}
    metric_rows: list[_FilterMetricsRow] = []
    sample_count = 0
    stats = _FilterWorkerStats(
        worker=0,
        predicate=type(predicate).__name__,
        requested_batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        predicate_setup_seconds=predicate_setup_seconds,
    )
    loader = _filter_loader(
        dataset,
        dataset_factory=dataset_factory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        runtime=runtime,
        sample_indexes=sample_indexes,
    )
    status = "complete"
    error_type: str | None = None
    try:
        for batch in _timed_filter_batches(loader, stats):
            selected = tuple(
                (index, sample) for index, sample in batch if index not in skip_indexes
            )
            stats.selected_samples += len(selected)
            outputs = _predicate_outputs(
                predicate,
                tuple(sample for _index, sample in selected),
                worker_id=0,
                stats=stats,
            )
            for (index, _sample), value in zip(selected, outputs):
                output = decision(value, metrics=write_metrics)
                if output.label not in partitions:
                    partitions[output.label] = array("q")
                partitions[output.label].append(index)
                sample_count += 1
                stats.processed_samples += 1
                if write_metrics and output.metrics is None:
                    raise TypeError(
                        "filter predicate must return FilterDecision when metrics=True."
                    )
                if output.metrics is not None:
                    metric_rows.append(
                        _FilterMetricsRow(
                            index=index,
                            label=output.label,
                            metrics=output.metrics,
                        )
                    )
                if sample_count == commit_samples:
                    yield _FilterChunk(partitions=partitions, metrics=metric_rows)
                    partitions = {}
                    metric_rows = []
                    sample_count = 0
        if partitions or metric_rows:
            yield _FilterChunk(partitions=partitions, metrics=metric_rows)
    except GeneratorExit:
        status = "interrupted"
        raise
    except BaseException as exc:
        status = "failed"
        error_type = type(exc).__name__
        raise
    finally:
        fields = _log_filter_worker_summary(stats, status=status, error_type=error_type)
        if collection_stats is not None:
            collection_stats.add(fields)


def collect_ranges_parallel(
    dataset_factory: DatasetFactory,
    factory: FilterFactory,
    devices: tuple[str, ...],
    metrics: bool,
    commit_samples: int,
    *,
    sample_count: int,
    skip_indexes: Collection[int] = frozenset(),
    sample_indexes: Sequence[int] | None = None,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    runtime: Runtime,
    use_map_style_loader: bool,
    worker_timeout: float | None = None,
    collection_stats: _FilterCollectionStats | None = None,
) -> Iterable[_FilterChunk]:
    workers = min(len(devices), sample_count)
    validate_process_value(
        "dataset_factory",
        dataset_factory,
        context="multi-device filtering",
        start_method=runtime.process_start_method,
    )
    validate_process_value(
        "factory",
        factory,
        context="multi-device filtering",
        start_method=runtime.process_start_method,
    )
    context = multiprocessing_context(runtime.process_start_method)
    worker_commit_samples = _worker_commit_samples(commit_samples)
    outputs = tuple(
        context.Queue(maxsize=_WORKER_QUEUE_SIZE) for _rank in range(workers)
    )
    logs_dir = run_logs_dir()
    worker_logs_dir = logs_dir / "filter"
    processes = [
        context.Process(
            target=_filter_worker,
            args=(
                dataset_factory,
                factory,
                metrics,
                worker_commit_samples,
                _FilterWorkerConfig(
                    worker=worker,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    runtime=runtime,
                    sample_count=sample_count,
                    use_map_style_loader=use_map_style_loader,
                    skip_indexes=skip_indexes,
                    sample_indexes=sample_indexes,
                    logs_dir=logs_dir,
                    worker_logs_dir=worker_logs_dir,
                ),
                outputs[rank],
            ),
            name=f"anydataset-filter-{rank}",
        )
        for rank, worker in enumerate(worker_configs(devices[:workers]))
    ]
    started: list[ProcessHandle] = []
    completed = False
    forced: tuple[ProcessHandle, ...] = ()
    try:
        for process in processes:
            process.start()
            started.append(process)
        yield from _ordered_worker_chunks(
            outputs,
            processes,
            workers=workers,
            sample_count=sample_count,
            commit_samples=commit_samples,
            skip_indexes=skip_indexes,
            sample_indexes=sample_indexes,
            worker_timeout=worker_timeout,
            collection_stats=collection_stats,
        )
        completed = True
    finally:
        if not completed:
            for process in started:
                if process.is_alive():
                    process.terminate()
            for process in started:
                process.join()
        else:
            forced = _join_completed_workers(started)
    failed = [
        process
        for process in processes
        if process.exitcode != 0 and process not in forced
    ]
    if failed:
        details = ", ".join(
            f"{process.name} exited with {process.exitcode}" for process in failed
        )
        raise RuntimeError(f"Filter workers failed: {details}.")


def _join_completed_workers(
    processes: Sequence[ProcessHandle],
) -> tuple[ProcessHandle, ...]:
    _join_until(processes, timeout=_WORKER_SHUTDOWN_TIMEOUT)
    lingering = tuple(process for process in processes if process.is_alive())
    for process in lingering:
        write_warning(
            "filter",
            "terminating completed filter worker after shutdown timeout: "
            f"name={process.name!r} pid={process.pid!r}",
            event="filter_worker_terminated_after_completion",
            fields={"name": process.name, "pid": process.pid},
        )
        process.terminate()
    _join_until(lingering, timeout=_WORKER_SHUTDOWN_TIMEOUT)
    stubborn = tuple(process for process in lingering if process.is_alive())
    for process in stubborn:
        write_warning(
            "filter",
            "killing completed filter worker after terminate timeout: "
            f"name={process.name!r} pid={process.pid!r}",
            event="filter_worker_killed_after_completion",
            fields={"name": process.name, "pid": process.pid},
        )
        process.kill()
    for process in stubborn:
        process.join()
    return lingering


def _join_until(processes: Sequence[ProcessHandle], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(deadline - time.monotonic(), 0.0))


def decision(value: FilterOutput, *, metrics: bool) -> _FilterDecision:
    if isinstance(value, FilterDecision):
        return _FilterDecision(
            label=label(value.label),
            metrics=validate_metrics(value.metrics) if metrics else None,
        )
    return _FilterDecision(label=label(value), metrics=None)


def _filter_worker(
    dataset_factory: DatasetFactory,
    factory: FilterFactory,
    metrics: bool,
    commit_samples: int,
    config: _FilterWorkerConfig,
    output: multiprocessing.Queue,
) -> None:
    worker = config.worker
    with use_run_logs_dir(config.logs_dir):
        logger = worker_logger("filter", config.worker_logs_dir, worker.rank)
        logger.info(
            "starting shard %s/%s on %s map_style=%s",
            worker.rank,
            worker.world_size,
            worker.device,
            config.use_map_style_loader,
        )
        env = set_worker_environment(worker, device_env="ANYDATASET_FILTER_DEVICE")
        processed = 0
        stats: _FilterWorkerStats | None = None
        status = "complete"
        error_type: str | None = None
        error: str | None = None
        try:
            setup_started = time.perf_counter()
            predicate = factory()
            stats = _FilterWorkerStats(
                worker=worker.rank,
                predicate=type(predicate).__name__,
                requested_batch_size=config.batch_size,
                num_workers=config.num_workers,
                prefetch_factor=config.prefetch_factor,
                predicate_setup_seconds=time.perf_counter() - setup_started,
            )
            for chunk in collect_shard(
                dataset_factory,
                predicate,
                metrics,
                commit_samples,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                prefetch_factor=config.prefetch_factor,
                runtime=config.runtime,
                sample_count=config.sample_count,
                use_map_style_loader=config.use_map_style_loader,
                skip_indexes=config.skip_indexes,
                sample_indexes=config.sample_indexes,
                stats=stats,
            ):
                processed += len(chunk.rows)
                blocked_started = time.perf_counter()
                output.put(chunk)
                stats.output_queue_blocked_seconds += (
                    time.perf_counter() - blocked_started
                )
            logger.info("finished shard %s processed=%s", worker.rank, processed)
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            error = traceback.format_exc()
            logger.error("worker failed processed=%s\n%s", processed, error)
        finally:
            summary: Mapping[str, object] | None = None
            if stats is not None:
                summary = _log_filter_worker_summary(
                    stats,
                    status=status,
                    error_type=error_type,
                )
            restore_environment(env)
        output.put((_DONE, worker.rank, error, summary))


def collect_shard(
    dataset_factory: DatasetFactory,
    predicate: FilterPredicate,
    write_metrics: bool,
    commit_samples: int,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    runtime: Runtime,
    sample_count: int | None = None,
    use_map_style_loader: bool | None = None,
    skip_indexes: Collection[int] = frozenset(),
    sample_indexes: Sequence[int] | None = None,
    stats: _FilterWorkerStats | None = None,
) -> Iterable[_IndexedFilterChunk]:
    rows: list[_FilterRow] = []
    if stats is None:
        stats = _FilterWorkerStats(
            worker=int(os.environ["RANK"]),
            predicate=type(predicate).__name__,
            requested_batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    dataset = None
    if use_map_style_loader is None or sample_count is None:
        dataset = dataset_factory()
    loader = _filter_loader(
        dataset,
        dataset_factory=dataset_factory,
        sample_count=sample_count,
        use_map_style_loader=use_map_style_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        runtime=runtime,
        sample_indexes=sample_indexes,
    )
    for batch in _timed_filter_batches(loader, stats):
        selected = tuple(
            (index, sample) for index, sample in batch if index not in skip_indexes
        )
        stats.selected_samples += len(selected)
        outputs = _predicate_outputs(
            predicate,
            tuple(sample for _index, sample in selected),
            worker_id=int(os.environ["RANK"]),
            stats=stats,
        )
        for (index, _sample), value in zip(selected, outputs):
            output = decision(value, metrics=write_metrics)
            if write_metrics and output.metrics is None:
                raise TypeError(
                    "filter predicate must return FilterDecision when metrics=True."
                )
            rows.append(
                _FilterRow(
                    index=index,
                    label=output.label,
                    metrics=output.metrics,
                )
            )
            stats.processed_samples += 1
            if len(rows) == commit_samples:
                yield _IndexedFilterChunk(
                    rank=int(os.environ["RANK"]), rows=tuple(rows)
                )
                rows = []
    if rows:
        yield _IndexedFilterChunk(rank=int(os.environ["RANK"]), rows=tuple(rows))


def _predicate_outputs(
    predicate: FilterPredicate,
    samples: Sequence[Sample],
    *,
    worker_id: int,
    stats: _FilterWorkerStats,
) -> Sequence[FilterOutput]:
    if not samples:
        return ()
    if not isinstance(predicate, BatchFilterPredicate):
        started = time.perf_counter()
        try:
            return tuple(predicate(sample) for sample in samples)
        finally:
            stats.record_scalar_predicates(len(samples), time.perf_counter() - started)

    predicate_name = type(predicate).__name__

    def call(batch: Sequence[Sample]) -> Sequence[FilterOutput]:
        started = time.perf_counter()
        try:
            return _validated_predicate_outputs(predicate, batch)
        finally:
            stats.record_predicate_call(len(batch), time.perf_counter() - started)

    def on_oom(batch_size: int, left_size: int, right_size: int) -> None:
        stats.oom_splits += 1
        ratio = stats.split_call_ratio
        write_progress_message(
            "filter",
            "predicate OOM: "
            f"worker={worker_id} predicate={predicate_name} "
            f"batch_size={batch_size}; retrying as {left_size}+{right_size} "
            "after cache cleanup; "
            f"oom_count={stats.oom_splits} "
            f"predicate_calls={stats.predicate_calls} "
            f"split/call={ratio:.6f}",
        )
        write_warning(
            "filter",
            "predicate OOM split: "
            f"worker={worker_id} predicate={predicate_name} "
            f"batch_size={batch_size} retry={left_size}+{right_size} "
            f"oom_count={stats.oom_splits} "
            f"predicate_calls={stats.predicate_calls} "
            f"split/call={ratio:.6f}",
            event="filter_predicate_oom_split",
            fields={
                "worker": worker_id,
                "predicate": predicate_name,
                "batch_size": batch_size,
                "left_size": left_size,
                "right_size": right_size,
                "oom_count": stats.oom_splits,
                "predicate_calls": stats.predicate_calls,
                "split_call_ratio": ratio,
            },
        )

    return tuple(
        iter_resilient_batch_outputs(
            samples,
            call,
            on_oom=on_oom,
        )
    )


def _validated_predicate_outputs(
    predicate: BatchFilterPredicate,
    samples: Sequence[Sample],
) -> Sequence[FilterOutput]:
    outputs = predicate.call_batch(samples)
    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
        raise TypeError(
            "filter predicate call_batch() must return an ordered sequence."
        )
    if len(outputs) != len(samples):
        raise ValueError(
            "filter predicate call_batch() must return one output per input "
            f"sample; received {len(outputs)} outputs for {len(samples)} samples."
        )
    return outputs


def _timed_filter_batches(
    loader: Iterable[Sequence[tuple[int, Sample]]],
    stats: _FilterWorkerStats,
) -> Iterable[Sequence[tuple[int, Sample]]]:
    iterator = iter(loader)
    while True:
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        stats.record_loader_batch(len(batch), time.perf_counter() - started)
        yield batch


def _log_filter_worker_summary(
    stats: _FilterWorkerStats,
    *,
    status: str,
    error_type: str | None,
) -> dict[str, object]:
    fields = stats.fields(status=status, error_type=error_type)
    elapsed = cast(float, fields["elapsed_seconds"])
    message = (
        "filter worker summary: "
        f"worker={stats.worker} status={status!r} "
        f"samples={stats.processed_samples} "
        f"predicate_calls={stats.predicate_calls} "
        f"oom_count={stats.oom_splits} "
        f"split/call={stats.split_call_ratio:.6f} "
        f"elapsed={elapsed:.3f}s"
    )
    write = write_info if status == "complete" else write_warning
    write(
        "filter",
        message,
        event="filter_worker_summary",
        fields=fields,
    )
    return fields


def _worker_commit_samples(commit_samples: int) -> int:
    return min(commit_samples, _PARALLEL_WORKER_COMMIT_SAMPLES)


def _filter_loader(
    dataset,
    *,
    dataset_factory: DatasetFactory,
    sample_count: int | None = None,
    use_map_style_loader: bool | None = None,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    runtime: Runtime,
    sample_indexes: Sequence[int] | None = None,
):
    return sample_index_loader(
        dataset_factory,
        dataset=dataset,
        sample_count=sample_count,
        sample_indexes=sample_indexes,
        use_map_style_loader=use_map_style_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        start_method=runtime.reader_worker_start_method,
    )


def _ordered_worker_chunks(
    outputs: Sequence[multiprocessing.Queue],
    processes: Sequence[ProcessHandle],
    *,
    workers: int,
    sample_count: int,
    commit_samples: int,
    skip_indexes: Collection[int],
    sample_indexes: Sequence[int] | None = None,
    worker_timeout: float | None = None,
    collection_stats: _FilterCollectionStats | None = None,
) -> Iterable[_FilterChunk]:
    buffers: tuple[dict[int, _FilterRow], ...] = tuple({} for _rank in range(workers))
    done: set[int] = set()
    rows: list[_FilterRow] = []
    last_message = time.monotonic()

    for rank, next_index in _ordered_worker_targets(
        sample_count,
        workers=workers,
        skip_indexes=skip_indexes,
        sample_indexes=sample_indexes,
    ):
        buffer = buffers[rank]
        while next_index not in buffer:
            if rank in done:
                raise RuntimeError(
                    f"Filter workers finished before emitting sample {next_index}."
                )
            last_message = _read_worker_message(
                outputs[rank],
                processes,
                buffer,
                done,
                rank=rank,
                validate_modulo=sample_indexes is None,
                worker_timeout=worker_timeout,
                last_message=last_message,
                collection_stats=collection_stats,
            )
        rows.append(buffer.pop(next_index))
        if len(rows) == commit_samples:
            yield _chunk_from_rows(rows)
            rows = []

    if rows:
        yield _chunk_from_rows(rows)

    for rank, output in enumerate(outputs):
        while rank not in done:
            last_message = _read_worker_message(
                output,
                processes,
                buffers[rank],
                done,
                rank=rank,
                validate_modulo=sample_indexes is None,
                worker_timeout=worker_timeout,
                last_message=last_message,
                collection_stats=collection_stats,
            )
    for buffer in buffers:
        if buffer:
            unexpected = min(buffer)
            raise RuntimeError(f"Filter worker emitted unexpected sample {unexpected}.")


def _ordered_worker_targets(
    sample_count: int,
    *,
    workers: int,
    skip_indexes: Collection[int],
    sample_indexes: Sequence[int] | None,
) -> Iterable[tuple[int, int]]:
    if sample_indexes is None:
        for index in range(sample_count):
            if index not in skip_indexes:
                yield index % workers, index
        return

    previous: int | None = None
    for position, index in enumerate(sample_indexes):
        if index < 0 or index >= sample_count:
            raise ValueError("sample index must satisfy 0 <= index < sample_count.")
        if previous is not None and index <= previous:
            raise ValueError("sample indexes must be strictly increasing.")
        previous = index
        if index not in skip_indexes:
            yield position % workers, index


def _read_worker_message(
    output: multiprocessing.Queue,
    processes: Sequence[ProcessHandle],
    buffer: dict[int, _FilterRow],
    done: set[int],
    *,
    rank: int,
    validate_modulo: bool,
    worker_timeout: float | None,
    last_message: float,
    collection_stats: _FilterCollectionStats | None = None,
) -> float:
    try:
        message = output.get(timeout=0.2)
    except queue.Empty:
        dead = [process for process in processes if process.exitcode not in (None, 0)]
        if dead:
            details = ", ".join(
                f"{process.name} exited with {process.exitcode}" for process in dead
            )
            raise RuntimeError(f"Filter worker exited early: {details}.")
        if (
            worker_timeout is not None
            and time.monotonic() - last_message > worker_timeout
        ):
            raise TimeoutError(f"Filter worker {rank} timed out.")
        return last_message
    last_message = time.monotonic()
    if isinstance(message, _IndexedFilterChunk):
        if message.rank != rank:
            raise RuntimeError(
                f"Filter worker {rank} queue received chunk from worker {message.rank}."
            )
        for row in message.rows:
            if validate_modulo and row.index % len(processes) != rank:
                raise RuntimeError(
                    f"Filter worker {rank} emitted sample {row.index} outside its shard."
                )
            if row.index in buffer:
                raise RuntimeError(f"Duplicate filtered sample index: {row.index}.")
            buffer[row.index] = row
        return last_message
    if not _done_message(message):
        raise RuntimeError(
            f"Filter worker {rank} queue received unexpected message: "
            f"{type(message).__name__}."
        )
    _, message_rank, error, *rest = message
    if message_rank != rank:
        raise RuntimeError(
            f"Filter worker {rank} queue received completion from worker "
            f"{message_rank}."
        )
    summary = rest[0] if rest else None
    if summary is not None:
        if not isinstance(summary, Mapping):
            raise RuntimeError(f"Filter worker {rank} returned an invalid summary.")
        if collection_stats is not None:
            collection_stats.add(summary)
    done.add(rank)
    if error is not None:
        raise RuntimeError(f"Filter worker {rank} failed.\n{error}")
    return last_message


def _chunk_from_rows(rows: Sequence[_FilterRow]) -> _FilterChunk:
    partitions: dict[str, array[int]] = {}
    metric_rows: list[_FilterMetricsRow] = []
    for row in rows:
        if row.label not in partitions:
            partitions[row.label] = array("q")
        partitions[row.label].append(row.index)
        if row.metrics is not None:
            metric_rows.append(
                _FilterMetricsRow(
                    index=row.index,
                    label=row.label,
                    metrics=row.metrics,
                )
            )
    return _FilterChunk(partitions=partitions, metrics=metric_rows)


def _done_message(message: object) -> bool:
    return isinstance(message, tuple) and len(message) in (3, 4) and message[0] == _DONE
