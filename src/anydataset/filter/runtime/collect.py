from __future__ import annotations

import multiprocessing
import os
import queue
import time
import traceback
from array import array
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..._runtime.logging import (
    run_logs_dir,
    use_run_logs_dir,
    worker_logger,
    write_warning,
)
from ..._runtime.oom import iter_resilient_batch_outputs
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


@dataclass
class _BatchCallStats:
    predicate_calls: int = 0
    oom_splits: int = 0

    @property
    def split_call_ratio(self) -> float:
        if self.predicate_calls == 0:
            return 0.0
        return self.oom_splits / self.predicate_calls


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
) -> Iterable[_FilterChunk]:
    env = set_single_worker_environment(device, device_env="ANYDATASET_FILTER_DEVICE")
    try:
        predicate = factory()
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
) -> Iterable[_FilterChunk]:
    partitions: dict[str, array[int]] = {}
    metric_rows: list[_FilterMetricsRow] = []
    sample_count = 0
    batch_stats = _BatchCallStats()
    loader = _filter_loader(
        dataset,
        dataset_factory=dataset_factory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        runtime=runtime,
        sample_indexes=sample_indexes,
    )
    for batch in loader:
        selected = tuple(
            (index, sample) for index, sample in batch if index not in skip_indexes
        )
        outputs = _predicate_outputs(
            predicate,
            tuple(sample for _index, sample in selected),
            worker_id=0,
            batch_stats=batch_stats,
        )
        for (index, _sample), value in zip(selected, outputs):
            output = decision(value, metrics=write_metrics)
            if output.label not in partitions:
                partitions[output.label] = array("q")
            partitions[output.label].append(index)
            sample_count += 1
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
        try:
            predicate = factory()
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
            ):
                processed += len(chunk.rows)
                output.put(chunk)
            logger.info("finished shard %s processed=%s", worker.rank, processed)
            output.put((_DONE, worker.rank, None))
        except Exception:
            error = traceback.format_exc()
            logger.error("worker failed processed=%s\n%s", processed, error)
            output.put((_DONE, worker.rank, error))
        finally:
            restore_environment(env)


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
) -> Iterable[_IndexedFilterChunk]:
    rows: list[_FilterRow] = []
    batch_stats = _BatchCallStats()
    dataset = None
    if use_map_style_loader is None or sample_count is None:
        dataset = dataset_factory()
    for batch in _filter_loader(
        dataset,
        dataset_factory=dataset_factory,
        sample_count=sample_count,
        use_map_style_loader=use_map_style_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        runtime=runtime,
        sample_indexes=sample_indexes,
    ):
        selected = tuple(
            (index, sample) for index, sample in batch if index not in skip_indexes
        )
        outputs = _predicate_outputs(
            predicate,
            tuple(sample for _index, sample in selected),
            worker_id=int(os.environ["RANK"]),
            batch_stats=batch_stats,
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
    batch_stats: _BatchCallStats,
) -> Sequence[FilterOutput]:
    if not samples:
        return ()
    if not isinstance(predicate, BatchFilterPredicate):
        return tuple(predicate(sample) for sample in samples)

    predicate_name = type(predicate).__name__

    def call(batch: Sequence[Sample]) -> Sequence[FilterOutput]:
        batch_stats.predicate_calls += 1
        return _validated_predicate_outputs(predicate, batch)

    def on_oom(batch_size: int, left_size: int, right_size: int) -> None:
        batch_stats.oom_splits += 1
        ratio = batch_stats.split_call_ratio
        write_progress_message(
            "filter",
            "predicate OOM: "
            f"worker={worker_id} predicate={predicate_name} "
            f"batch_size={batch_size}; retrying as {left_size}+{right_size} "
            "after cache cleanup; "
            f"oom_count={batch_stats.oom_splits} "
            f"predicate_calls={batch_stats.predicate_calls} "
            f"split/call={ratio:.6f}",
        )
        write_warning(
            "filter",
            "predicate OOM split: "
            f"worker={worker_id} predicate={predicate_name} "
            f"batch_size={batch_size} retry={left_size}+{right_size} "
            f"oom_count={batch_stats.oom_splits} "
            f"predicate_calls={batch_stats.predicate_calls} "
            f"split/call={ratio:.6f}",
            event="filter_predicate_oom_split",
            fields={
                "worker": worker_id,
                "predicate": predicate_name,
                "batch_size": batch_size,
                "left_size": left_size,
                "right_size": right_size,
                "oom_count": batch_stats.oom_splits,
                "predicate_calls": batch_stats.predicate_calls,
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
    _, rank, error = message
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
    return isinstance(message, tuple) and len(message) == 3 and message[0] == _DONE
