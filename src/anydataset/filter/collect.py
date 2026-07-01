from __future__ import annotations

import multiprocessing
import os
import queue
import traceback
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .._parallel import (
    DeviceWorker,
    indexed_loader,
    map_style_indexed_loader,
    multiprocessing_context,
    restore_environment,
    set_single_worker_environment,
    set_worker_environment,
    validate_process_value,
    worker_configs,
)
from ..dataset.abc import uses_default_indexed_shard
from ..runtime import Runtime
from .rules import label
from .storage import validate_metrics
from .types import (
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


@dataclass(frozen=True)
class _FilterRow:
    index: int
    label: str
    metrics: Mapping[str, JsonValue] | None


@dataclass(frozen=True)
class _IndexedFilterChunk:
    rank: int
    rows: Sequence[_FilterRow]


def collect_ranges(
    dataset,
    factory: FilterFactory,
    device: str,
    metrics: bool,
    commit_samples: int,
    *,
    skip_indexes: frozenset[int] = frozenset(),
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
    skip_indexes: frozenset[int] = frozenset(),
    dataset_factory: DatasetFactory,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    runtime: Runtime,
) -> Iterable[_FilterChunk]:
    partitions: dict[str, array[int]] = {}
    metric_rows: list[_FilterMetricsRow] = []
    sample_count = 0
    loader = _filter_loader(
        dataset,
        dataset_factory=dataset_factory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        runtime=runtime,
    )
    for batch in loader:
        for index, sample in batch:
            if index in skip_indexes:
                continue
            output = decision(predicate(sample), metrics=write_metrics)
            if output.label not in partitions:
                partitions[output.label] = array("q")
            partitions[output.label].append(index)
            sample_count += 1
            if write_metrics and output.metrics is None:
                raise TypeError("filter predicate must return FilterDecision when metrics=True.")
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
    skip_indexes: frozenset[int] = frozenset(),
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    runtime: Runtime,
    use_map_style_loader: bool,
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
    outputs = tuple(context.Queue(maxsize=_WORKER_QUEUE_SIZE) for _rank in range(workers))
    processes = [
        context.Process(
            target=_filter_worker,
            args=(
                dataset_factory,
                factory,
                metrics,
                worker_commit_samples,
                worker,
                batch_size,
                num_workers,
                prefetch_factor,
                runtime,
                sample_count,
                use_map_style_loader,
                skip_indexes,
                outputs[rank],
            ),
            name=f"anydataset-filter-{rank}",
        )
        for rank, worker in enumerate(worker_configs(devices[:workers]))
    ]
    for process in processes:
        process.start()
    completed = False
    try:
        yield from _ordered_worker_chunks(
            outputs,
            processes,
            workers=workers,
            sample_count=sample_count,
            commit_samples=commit_samples,
            skip_indexes=skip_indexes,
        )
        completed = True
    finally:
        if not completed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join()
    failed = [process for process in processes if process.exitcode != 0]
    if failed:
        details = ", ".join(
            f"{process.name} exited with {process.exitcode}" for process in failed
        )
        raise RuntimeError(f"Filter workers failed: {details}.")


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
    worker: DeviceWorker,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    runtime: Runtime,
    sample_count: int,
    use_map_style_loader: bool,
    skip_indexes: frozenset[int],
    output: multiprocessing.Queue,
) -> None:
    env = set_worker_environment(worker, device_env="ANYDATASET_FILTER_DEVICE")
    try:
        predicate = factory()
        for chunk in collect_indexed_shard(
            dataset_factory,
            predicate,
            metrics,
            commit_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            runtime=runtime,
            sample_count=sample_count,
            use_map_style_loader=use_map_style_loader,
            skip_indexes=skip_indexes,
        ):
            output.put(chunk)
        output.put((_DONE, worker.rank, None))
    except Exception:
        output.put((_DONE, worker.rank, traceback.format_exc()))
    finally:
        restore_environment(env)


def collect_indexed_shard(
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
    skip_indexes: frozenset[int] = frozenset(),
) -> Iterable[_IndexedFilterChunk]:
    rows: list[_FilterRow] = []
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
    ):
        for index, sample in batch:
            if index in skip_indexes:
                continue
            output = decision(predicate(sample), metrics=write_metrics)
            if write_metrics and output.metrics is None:
                raise TypeError("filter predicate must return FilterDecision when metrics=True.")
            rows.append(
                _FilterRow(
                    index=index,
                    label=output.label,
                    metrics=output.metrics,
                )
            )
            if len(rows) == commit_samples:
                yield _IndexedFilterChunk(rank=int(os.environ["RANK"]), rows=tuple(rows))
                rows = []
    if rows:
        yield _IndexedFilterChunk(rank=int(os.environ["RANK"]), rows=tuple(rows))


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
):
    if use_map_style_loader is None:
        use_map_style_loader = uses_default_indexed_shard(dataset)
    if use_map_style_loader:
        if sample_count is None:
            sample_count = len(dataset)
        return map_style_indexed_loader(
            dataset_factory,
            sample_count=sample_count,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            start_method=runtime.reader_worker_start_method,
            dataset=dataset,
        )
    return indexed_loader(
        dataset_factory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        start_method=runtime.reader_worker_start_method,
    )


def _ordered_worker_chunks(
    outputs: Sequence[multiprocessing.Queue],
    processes: list[multiprocessing.Process],
    *,
    workers: int,
    sample_count: int,
    commit_samples: int,
    skip_indexes: frozenset[int],
) -> Iterable[_FilterChunk]:
    buffers: tuple[dict[int, _FilterRow], ...] = tuple({} for _rank in range(workers))
    done: set[int] = set()
    next_index = 0
    rows: list[_FilterRow] = []

    while next_index < sample_count:
        if next_index in skip_indexes:
            next_index += 1
            continue
        rank = next_index % workers
        buffer = buffers[rank]
        while next_index not in buffer:
            if rank in done:
                raise RuntimeError(
                    f"Filter workers finished before emitting sample {next_index}."
                )
            _read_worker_message(outputs[rank], processes, buffer, done, rank=rank)
        rows.append(buffer.pop(next_index))
        next_index += 1
        if len(rows) == commit_samples:
            yield _chunk_from_rows(rows)
            rows = []

    if rows:
        yield _chunk_from_rows(rows)

    for rank, output in enumerate(outputs):
        while rank not in done:
            _read_worker_message(output, processes, buffers[rank], done, rank=rank)


def _read_worker_message(
    output: multiprocessing.Queue,
    processes: list[multiprocessing.Process],
    buffer: dict[int, _FilterRow],
    done: set[int],
    *,
    rank: int,
) -> None:
    try:
        message = output.get(timeout=0.2)
    except queue.Empty:
        dead = [
            process
            for process in processes
            if process.exitcode not in (None, 0)
        ]
        if dead:
            details = ", ".join(
                f"{process.name} exited with {process.exitcode}" for process in dead
            )
            raise RuntimeError(f"Filter worker exited early: {details}.")
        return
    if isinstance(message, _IndexedFilterChunk):
        if message.rank != rank:
            raise RuntimeError(
                f"Filter worker {rank} queue received chunk from worker {message.rank}."
            )
        for row in message.rows:
            if row.index % len(processes) != rank:
                raise RuntimeError(
                    f"Filter worker {rank} emitted sample {row.index} outside its shard."
                )
            if row.index in buffer:
                raise RuntimeError(f"Duplicate filtered sample index: {row.index}.")
            buffer[row.index] = row
        return
    if not _done_message(message):
        return
    _, rank, error = message
    done.add(rank)
    if error is not None:
        raise RuntimeError(f"Filter worker {rank} failed.\n{error}")


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
    return (
        isinstance(message, tuple)
        and len(message) == 3
        and message[0] == _DONE
    )
