from __future__ import annotations

import multiprocessing
import os
import pickle
import queue
import socket
import traceback
from array import array
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ..dataset.abc import AnyDataset
from .rules import label
from .storage import validate_metrics
from .types import (
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
class _WorkerConfig:
    device: str
    master_addr: str
    master_port: str
    rank: int
    world_size: int


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
    dataset: AnyDataset,
    factory: FilterFactory,
    device: str,
    metrics: bool,
    commit_samples: int,
) -> Iterable[_FilterChunk]:
    env = _set_worker_environment(_single_worker_config(device))
    try:
        predicate = factory()
        for start, stop in range_chunks(len(dataset), commit_samples):
            yield collect_range(dataset, predicate, metrics, start, stop)
    finally:
        _restore_environment(env)


def collect_ranges_parallel(
    dataset: AnyDataset,
    factory: FilterFactory,
    devices: tuple[str, ...],
    metrics: bool,
    commit_samples: int,
) -> Iterable[_FilterChunk]:
    sample_count = len(dataset)
    workers = min(len(devices), sample_count)
    master_addr = "127.0.0.1"
    master_port = _free_port()
    _validate_spawn_value("dataset", dataset)
    _validate_spawn_value("factory", factory)
    context = _multiprocessing_context()
    output = context.Queue()
    processes = [
        context.Process(
            target=_filter_worker,
            args=(
                dataset,
                factory,
                metrics,
                commit_samples,
                _WorkerConfig(
                    device=device,
                    master_addr=master_addr,
                    master_port=master_port,
                    rank=rank,
                    world_size=workers,
                ),
                output,
            ),
            name=f"anydataset-filter-{rank}",
        )
        for rank, device in enumerate(devices[:workers])
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


def collect_range(
    dataset: AnyDataset,
    predicate: FilterPredicate,
    write_metrics: bool,
    start: int,
    stop: int,
) -> _FilterChunk:
    partitions: dict[str, array[int]] = {}
    metric_rows: list[_FilterMetricsRow] = []
    for index in range(start, stop):
        output = decision(predicate(dataset[index]), metrics=write_metrics)
        if output.label not in partitions:
            partitions[output.label] = array("q")
        partitions[output.label].append(index)
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
    return _FilterChunk(partitions=partitions, metrics=metric_rows)


def decision(value: FilterOutput, *, metrics: bool) -> _FilterDecision:
    if isinstance(value, FilterDecision):
        return _FilterDecision(
            label=label(value.label),
            metrics=validate_metrics(value.metrics) if metrics else None,
        )
    return _FilterDecision(label=label(value), metrics=None)


def range_chunks(sample_count: int, chunk_samples: int) -> Iterable[tuple[int, int]]:
    for start in range(0, sample_count, chunk_samples):
        yield start, min(start + chunk_samples, sample_count)


def _filter_worker(
    dataset: AnyDataset,
    factory: FilterFactory,
    metrics: bool,
    commit_samples: int,
    config: _WorkerConfig,
    output: multiprocessing.Queue,
) -> None:
    env = _set_worker_environment(config)
    try:
        predicate = factory()
        for chunk in collect_indexed_shard(
            dataset,
            predicate,
            metrics,
            commit_samples,
            num_shards=config.world_size,
            shard_id=config.rank,
        ):
            output.put(chunk)
        output.put((_DONE, config.rank, None))
    except Exception:
        output.put((_DONE, config.rank, traceback.format_exc()))
    finally:
        _restore_environment(env)


def collect_indexed_shard(
    dataset: AnyDataset,
    predicate: FilterPredicate,
    write_metrics: bool,
    commit_samples: int,
    *,
    num_shards: int,
    shard_id: int,
) -> Iterable[_IndexedFilterChunk]:
    rows: list[_FilterRow] = []
    for index in range(shard_id, len(dataset), num_shards):
        output = decision(predicate(dataset[index]), metrics=write_metrics)
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
            yield _IndexedFilterChunk(rank=shard_id, rows=tuple(rows))
            rows = []
    if rows:
        yield _IndexedFilterChunk(rank=shard_id, rows=tuple(rows))


def _ordered_worker_chunks(
    output: multiprocessing.Queue,
    processes: list[multiprocessing.Process],
    *,
    workers: int,
    sample_count: int,
    commit_samples: int,
) -> Iterable[_FilterChunk]:
    buffers = [deque() for _ in range(workers)]
    done: set[int] = set()
    next_index = 0
    rows: list[_FilterRow] = []

    while next_index < sample_count:
        rank = next_index % workers
        while not buffers[rank] or buffers[rank][0].index != next_index:
            if rank in done:
                raise RuntimeError(
                    f"Filter worker {rank} finished before emitting sample {next_index}."
                )
            _read_worker_message(output, processes, buffers, done)
        rows.append(buffers[rank].popleft())
        next_index += 1
        if len(rows) == commit_samples:
            yield _chunk_from_rows(rows)
            rows = []

    if rows:
        yield _chunk_from_rows(rows)

    while len(done) < workers:
        _read_worker_message(output, processes, buffers, done)


def _read_worker_message(
    output: multiprocessing.Queue,
    processes: list[multiprocessing.Process],
    buffers: Sequence[deque],
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
        buffers[message.rank].extend(message.rows)
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


def _multiprocessing_context():
    return multiprocessing.get_context("spawn")


def _set_worker_environment(config: _WorkerConfig) -> dict[str, str | None]:
    previous = {
        name: os.environ.get(name)
        for name in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "ANYDATASET_FILTER_DEVICE",
        )
    }
    os.environ["RANK"] = str(config.rank)
    os.environ["LOCAL_RANK"] = _local_rank(config.device, config.rank)
    os.environ["WORLD_SIZE"] = str(config.world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(config.world_size)
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    os.environ["ANYDATASET_FILTER_DEVICE"] = config.device
    return previous


def _restore_environment(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _local_rank(device: str, fallback: int) -> str:
    prefix = "cuda:"
    if device.startswith(prefix):
        return device.removeprefix(prefix)
    return str(fallback)


def _single_worker_config(device: str) -> _WorkerConfig:
    return _WorkerConfig(
        device=device,
        master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        master_port=os.environ.get("MASTER_PORT", _free_port()),
        rank=0,
        world_size=1,
    )


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _validate_spawn_value(name: str, value: object) -> None:
    try:
        pickle.dumps(value)
    except Exception as exc:
        raise TypeError(f"{name} must be picklable for multi-device filtering.") from exc
