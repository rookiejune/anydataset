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
    multiprocessing_context,
    restore_environment,
    set_single_worker_environment,
    set_worker_environment,
    validate_spawn_value,
    worker_configs,
)
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
    dataset_factory: DatasetFactory,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
) -> Iterable[_FilterChunk]:
    env = set_single_worker_environment(device, device_env="ANYDATASET_FILTER_DEVICE")
    try:
        predicate = factory()
        yield from collect_ranges_sequential(
            dataset,
            predicate,
            metrics,
            commit_samples,
            dataset_factory=dataset_factory,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    finally:
        restore_environment(env)


def collect_ranges_sequential(
    dataset,
    predicate: FilterPredicate,
    write_metrics: bool,
    commit_samples: int,
    *,
    dataset_factory: DatasetFactory,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
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
    )
    for batch in loader:
        for index, sample in batch:
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
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
) -> Iterable[_FilterChunk]:
    workers = min(len(devices), sample_count)
    validate_spawn_value(
        "dataset_factory",
        dataset_factory,
        context="multi-device filtering",
    )
    validate_spawn_value("factory", factory, context="multi-device filtering")
    context = multiprocessing_context()
    output = context.Queue()
    processes = [
        context.Process(
            target=_filter_worker,
            args=(
                dataset_factory,
                factory,
                metrics,
                commit_samples,
                worker,
                batch_size,
                num_workers,
                prefetch_factor,
                output,
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
            output,
            processes,
            workers=workers,
            sample_count=sample_count,
            commit_samples=commit_samples,
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
) -> Iterable[_IndexedFilterChunk]:
    rows: list[_FilterRow] = []
    for batch in indexed_loader(
        dataset_factory=dataset_factory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    ):
        for index, sample in batch:
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


def _filter_loader(
    dataset,
    *,
    dataset_factory: DatasetFactory,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
):
    return indexed_loader(
        dataset_factory,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def _ordered_worker_chunks(
    output: multiprocessing.Queue,
    processes: list[multiprocessing.Process],
    *,
    workers: int,
    sample_count: int,
    commit_samples: int,
) -> Iterable[_FilterChunk]:
    buffer: dict[int, _FilterRow] = {}
    done: set[int] = set()
    next_index = 0
    rows: list[_FilterRow] = []

    while next_index < sample_count:
        while next_index not in buffer:
            if len(done) == workers:
                raise RuntimeError(
                    f"Filter workers finished before emitting sample {next_index}."
                )
            _read_worker_message(output, processes, buffer, done)
        rows.append(buffer.pop(next_index))
        next_index += 1
        if len(rows) == commit_samples:
            yield _chunk_from_rows(rows)
            rows = []

    if rows:
        yield _chunk_from_rows(rows)

    while len(done) < workers:
        _read_worker_message(output, processes, buffer, done)


def _read_worker_message(
    output: multiprocessing.Queue,
    processes: list[multiprocessing.Process],
    buffer: dict[int, _FilterRow],
    done: set[int],
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
        for row in message.rows:
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
