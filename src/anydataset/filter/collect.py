from __future__ import annotations

import multiprocessing
from array import array
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor

from ..dataset.abc import AnyDataset
from .rules import label
from .storage import validate_metrics
from .types import (
    FilterDecision,
    FilterPredicate,
    FilterOutput,
    _FilterChunk,
    _FilterDecision,
    _FilterMetricsRow,
)

_WORKER_DATASET: AnyDataset | None = None
_WORKER_PREDICATE: FilterPredicate | None = None
_WORKER_METRICS = False


def collect_ranges(
    dataset: AnyDataset,
    predicate: FilterPredicate,
    metrics: bool,
    commit_samples: int,
) -> Iterable[_FilterChunk]:
    for start, stop in range_chunks(len(dataset), commit_samples):
        yield collect_range(dataset, predicate, metrics, start, stop)


def collect_ranges_parallel(
    dataset: AnyDataset,
    predicate: FilterPredicate,
    num_workers: int,
    metrics: bool,
    commit_samples: int,
) -> Iterable[_FilterChunk]:
    sample_count = len(dataset)
    workers = min(num_workers, sample_count)
    chunk_samples = min(commit_samples, (sample_count + workers - 1) // workers)
    context = _multiprocessing_context()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_filter_worker,
        initargs=(dataset, predicate, metrics),
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
    predicate: FilterPredicate,
    metrics: bool,
) -> None:
    global _WORKER_DATASET, _WORKER_PREDICATE, _WORKER_METRICS
    _WORKER_DATASET = dataset
    _WORKER_PREDICATE = predicate
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
    if "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()
