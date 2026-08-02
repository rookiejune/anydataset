from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from ..._runtime.logging import run_logs_dir
from ..._runtime.devices import Devices, resolve_devices
from ..._runtime.resume import dataset_sample_count
from ..._validation import (
    non_negative_int,
    optional_positive_float,
    optional_positive_int,
    positive_int,
)
from ...cache import FileLock
from ...runtime import Runtime
from ..cache.ready import (
    log_filter_cache_miss,
    metrics_path,
    ready_filter_generation,
    write_cache,
)
from ..cache.generations import FilterGeneration
from ..cache.identity import (
    FilterBase,
    filter_base,
    filter_identity,
    filter_lock_path,
    filter_path as filter_path,
    metadata,
)
from ..cache.storage import read_partitions
from ..types import DatasetFactory, FilterApplyReport

if TYPE_CHECKING:
    from ..api import FilterRule, _FilterCache

_CACHE_LOCK_TIMEOUT = 3600.0
_CACHE_LOCK_POLL = 0.2


@dataclass(frozen=True)
class _AppliedFilter:
    cache: _FilterCache
    report: FilterApplyReport | None


@dataclass(frozen=True)
class _EnsuredFilter:
    generation: FilterGeneration
    sample_count: int
    cache_hit: bool
    logs_dir: Path | None
    cache_lookup_seconds: float
    cache_build_seconds: float


def apply_filter(
    rule: FilterRule,
    *,
    input_id: str | None,
    metrics: bool,
    device: Devices,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    commit_samples: int,
    max_shard_samples: int | None,
    write_workers: int,
    write_prefetch: int | None,
    worker_timeout: float | None,
    runtime: Runtime,
    rebuild: bool,
    dataset_factory: DatasetFactory,
    with_report: bool = False,
) -> _AppliedFilter:
    from ..api import _FilterCache

    started_at = perf_counter()
    dataset_started_at = perf_counter()
    dataset = filter_base(dataset_factory())
    dataset_seconds = perf_counter() - dataset_started_at
    ensured = ensure_filter(
        dataset,
        rule,
        input_id=input_id,
        metrics=metrics,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        commit_samples=commit_samples,
        max_shard_samples=max_shard_samples,
        write_workers=write_workers,
        write_prefetch=write_prefetch,
        worker_timeout=worker_timeout,
        runtime=runtime,
        rebuild=rebuild,
        dataset_factory=dataset_factory,
    )
    try:
        partition_started_at = perf_counter()
        partitions = read_partitions(ensured.generation.path)
        partition_read_seconds = perf_counter() - partition_started_at
        cache = _FilterCache(
            dataset,
            partitions,
            rule,
            ensured.generation.path,
            lease=ensured.generation.lease,
            metrics_path=metrics_path(ensured.generation.path) if metrics else None,
            dataset_factory=dataset_factory,
            input_id=input_id,
        )
        report = None
        if with_report:
            report = FilterApplyReport(
                logs_dir=ensured.logs_dir,
                elapsed_seconds=perf_counter() - started_at,
                dataset_seconds=dataset_seconds,
                cache_lookup_seconds=ensured.cache_lookup_seconds,
                cache_build_seconds=ensured.cache_build_seconds,
                partition_read_seconds=partition_read_seconds,
                sample_count=ensured.sample_count,
                cache_hit=ensured.cache_hit,
                cache_path=ensured.generation.path,
            )
        return _AppliedFilter(cache=cache, report=report)
    except Exception:
        ensured.generation.lease.close()
        raise


def ensure_filter(
    dataset: FilterBase,
    rule: FilterRule,
    *,
    input_id: str | None,
    metrics: bool,
    device: Devices,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    commit_samples: int,
    max_shard_samples: int | None,
    write_workers: int,
    write_prefetch: int | None,
    worker_timeout: float | None,
    runtime: Runtime,
    rebuild: bool,
    dataset_factory: DatasetFactory,
) -> _EnsuredFilter:
    from ..api import FilterRule
    from ..cache.resume import cleanup_filter_resume_dir

    dataset = filter_base(dataset)
    if not isinstance(rule, FilterRule):
        raise TypeError("rule must be a FilterRule.")
    if not isinstance(metrics, bool):
        raise TypeError("metrics must be a bool.")
    if not isinstance(rebuild, bool):
        raise TypeError("rebuild must be a bool.")
    devices = resolve_devices(device)
    batch_size = positive_int("batch_size", batch_size)
    num_workers = non_negative_int("num_workers", num_workers)
    prefetch_factor = optional_positive_int("prefetch_factor", prefetch_factor)
    commit_samples = positive_int("commit_samples", commit_samples)
    max_shard_samples = optional_positive_int(
        "max_shard_samples",
        max_shard_samples,
    )
    write_workers = non_negative_int("write_workers", write_workers)
    write_prefetch = optional_positive_int("write_prefetch", write_prefetch)
    worker_timeout = optional_positive_float("worker_timeout", worker_timeout)

    lookup_started_at = perf_counter()
    identity = filter_identity(dataset, input_id=input_id)
    base_count = dataset_sample_count(dataset, context="filter")
    expected = metadata(identity, base_count, rule)
    cache_path = filter_path(rule, identity)

    if not rebuild:
        generation, reason = ready_filter_generation(
            cache_path,
            expected,
            metrics=metrics,
        )
        if generation is not None:
            return _EnsuredFilter(
                generation=generation,
                sample_count=base_count,
                cache_hit=True,
                logs_dir=None,
                cache_lookup_seconds=perf_counter() - lookup_started_at,
                cache_build_seconds=0.0,
            )

    lock_path = filter_lock_path(rule, identity)
    with FileLock(
        lock_path,
        wait_timeout=_CACHE_LOCK_TIMEOUT,
        poll_interval=_CACHE_LOCK_POLL,
    ):
        if rebuild:
            cleanup_filter_resume_dir(cache_path)
            current = cache_path / "current.json"
            if current.is_file():
                current.unlink()
            reason = "rebuild requested"
        else:
            generation, reason = ready_filter_generation(
                cache_path,
                expected,
                metrics=metrics,
            )
            if generation is not None:
                return _EnsuredFilter(
                    generation=generation,
                    sample_count=base_count,
                    cache_hit=True,
                    logs_dir=None,
                    cache_lookup_seconds=perf_counter() - lookup_started_at,
                    cache_build_seconds=0.0,
                )
            if reason is None:
                raise RuntimeError("filter cache miss must include a reason.")
        build_started_at = perf_counter()
        logs_dir = run_logs_dir()
        log_filter_cache_miss(
            cache_path,
            rule,
            identity,
            base_count=base_count,
            metrics=metrics,
            reason=reason,
        )
        generation = write_cache(
            cache_path,
            expected,
            dataset,
            rule,
            metrics=metrics,
            devices=devices,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            commit_samples=commit_samples,
            max_shard_samples=max_shard_samples,
            write_workers=write_workers,
            write_prefetch=write_prefetch,
            worker_timeout=worker_timeout,
            runtime=runtime,
            dataset_factory=dataset_factory,
        )
        return _EnsuredFilter(
            generation=generation,
            sample_count=base_count,
            cache_hit=False,
            logs_dir=logs_dir,
            cache_lookup_seconds=build_started_at - lookup_started_at,
            cache_build_seconds=perf_counter() - build_started_at,
        )
