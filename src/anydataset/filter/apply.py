from __future__ import annotations

from typing import TYPE_CHECKING

from .._devices import Devices, resolve_devices
from .._resume import dataset_sample_count
from .._validation import (
    non_negative_int,
    optional_positive_float,
    optional_positive_int,
    positive_int,
)
from ..cache import FileLock
from ..runtime import Runtime
from ._cache import (
    log_filter_cache_miss,
    metrics_path,
    ready_filter_generation,
    write_cache,
)
from .generations import FilterGeneration
from ._identity import (
    FilterBase,
    filter_base,
    filter_identity,
    filter_lock_path,
    filter_path as filter_path,
    metadata,
)
from .storage import read_partitions
from .types import DatasetFactory

if TYPE_CHECKING:
    from .api import FilterRule, _FilterCache

_CACHE_LOCK_TIMEOUT = 3600.0
_CACHE_LOCK_POLL = 0.2


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
    dataset_factory: DatasetFactory,
) -> _FilterCache:
    from .api import _FilterCache

    dataset = filter_base(dataset_factory())
    generation = ensure_filter(
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
        dataset_factory=dataset_factory,
    )
    try:
        return _FilterCache(
            dataset,
            read_partitions(generation.path),
            rule,
            generation.path,
            lease=generation.lease,
            metrics_path=metrics_path(generation.path) if metrics else None,
            dataset_factory=dataset_factory,
            input_id=input_id,
        )
    except Exception:
        generation.lease.close()
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
    dataset_factory: DatasetFactory,
) -> FilterGeneration:
    from .api import FilterRule

    dataset = filter_base(dataset)
    if not isinstance(rule, FilterRule):
        raise TypeError("rule must be a FilterRule.")
    if not isinstance(metrics, bool):
        raise TypeError("metrics must be a bool.")
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

    identity = filter_identity(dataset, input_id=input_id)
    base_count = dataset_sample_count(dataset, context="filter")
    expected = metadata(identity, base_count, rule)
    cache_path = filter_path(rule, identity)

    generation, reason = ready_filter_generation(
        cache_path,
        expected,
        metrics=metrics,
    )
    if generation is not None:
        return generation

    lock_path = filter_lock_path(rule, identity)
    with FileLock(
        lock_path,
        wait_timeout=_CACHE_LOCK_TIMEOUT,
        poll_interval=_CACHE_LOCK_POLL,
    ):
        generation, reason = ready_filter_generation(
            cache_path,
            expected,
            metrics=metrics,
        )
        if generation is not None:
            return generation
        log_filter_cache_miss(
            cache_path,
            rule,
            identity,
            base_count=base_count,
            metrics=metrics,
            reason=reason,
        )
        return write_cache(
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
