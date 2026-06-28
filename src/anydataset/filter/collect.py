from __future__ import annotations

import multiprocessing
import os
import pickle
import socket
from array import array
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from ..dataset.abc import AnyDataset
from .rules import label
from .storage import validate_metrics
from .types import (
    FilterDecision,
    FilterFactory,
    FilterPredicate,
    FilterOutput,
    _FilterChunk,
    _FilterDecision,
    _FilterMetricsRow,
)

_WORKER_DATASET: AnyDataset | None = None
_WORKER_PREDICATE: FilterPredicate | None = None
_WORKER_METRICS = False


@dataclass(frozen=True)
class _WorkerConfig:
    device: str
    master_addr: str
    master_port: str
    rank: int
    world_size: int


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
    chunk_samples = min(commit_samples, (sample_count + workers - 1) // workers)
    master_addr = "127.0.0.1"
    master_port = _free_port()
    _validate_spawn_value("dataset", dataset)
    _validate_spawn_value("factory", factory)
    context = _multiprocessing_context()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_filter_worker,
        initargs=(
            dataset,
            factory,
            metrics,
            tuple(devices[:workers]),
            master_addr,
            master_port,
        ),
    ) as executor:
        yield from _map_range_chunks(
            executor,
            range_chunks(sample_count, chunk_samples),
            max_pending=workers * 2,
        )


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


def _map_range_chunks(
    executor: ProcessPoolExecutor,
    chunks: Iterable[tuple[int, int]],
    *,
    max_pending: int,
) -> Iterable[_FilterChunk]:
    chunk_iter = iter(chunks)
    pending = {}
    next_submit = 0
    next_yield = 0

    def submit_next() -> None:
        nonlocal next_submit
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            return
        pending[next_submit] = executor.submit(_collect_worker_range, chunk)
        next_submit += 1

    for _ in range(max_pending):
        submit_next()

    while pending:
        future = pending.pop(next_yield)
        yield future.result()
        next_yield += 1
        submit_next()


def _init_filter_worker(
    dataset: AnyDataset,
    factory: FilterFactory,
    metrics: bool,
    devices: tuple[str, ...],
    master_addr: str,
    master_port: str,
) -> None:
    global _WORKER_DATASET, _WORKER_PREDICATE, _WORKER_METRICS
    _WORKER_DATASET = dataset
    process = multiprocessing.current_process()
    rank = process._identity[0] - 1 if process._identity else 0
    rank = rank % len(devices)
    _set_worker_environment(
        _WorkerConfig(
            device=devices[rank],
            master_addr=master_addr,
            master_port=master_port,
            rank=rank,
            world_size=len(devices),
        )
    )
    _WORKER_PREDICATE = factory()
    _WORKER_METRICS = metrics


def _collect_worker_range(bounds: tuple[int, int]) -> _FilterChunk:
    if _WORKER_DATASET is None or _WORKER_PREDICATE is None:
        raise RuntimeError("filter worker was not initialized.")
    start, stop = bounds
    return collect_range(
        _WORKER_DATASET,
        _WORKER_PREDICATE,
        _WORKER_METRICS,
        start,
        stop,
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
