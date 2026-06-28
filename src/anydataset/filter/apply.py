from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._devices import Devices, resolve_devices
from ..cache import CacheManager, CacheManifest, FileLock
from ..dataset.abc import AnyDataset, SampleDataset
from ..store.jsonio import read_json, write_json
from ..store.reader import StoreDataset
from ..types import Source, Spec
from .collect import collect_ranges, collect_ranges_parallel
from .rules import optional_positive_int, positive_int, rule_cache_key
from .storage import (
    MetricsWriter,
    PartitionWriter,
    metrics_ready,
    partition_files,
    read_partitions,
)
from .types import _FilterChunk

if TYPE_CHECKING:
    from .api import FilterResult, FilterRule

FilterBase = SampleDataset

_DEFAULT_MAX_SHARD_SAMPLES = 1_000_000
_DEFAULT_COMMIT_SAMPLES = 100_000
_FILTER_VIEW_SCHEMA_VERSION = 1


def apply_filter(
    dataset: FilterBase,
    rule: FilterRule,
    *,
    metrics: bool,
    device: Devices,
    commit_samples: int,
    max_shard_samples: int | None,
    cache_root: str | Path | None,
) -> FilterResult:
    from .api import FilterResult

    cache_path, metric_path = ensure_filter(
        dataset,
        rule,
        metrics=metrics,
        device=device,
        commit_samples=commit_samples,
        max_shard_samples=max_shard_samples,
        cache_root=cache_root,
    )
    return FilterResult(
        dataset,
        read_partitions(cache_path),
        rule,
        cache_path,
        metrics_path=metric_path,
    )


def ensure_filter(
    dataset: FilterBase,
    rule: FilterRule,
    *,
    metrics: bool,
    device: Devices,
    commit_samples: int,
    max_shard_samples: int | None,
    cache_root: str | Path | None,
) -> tuple[Path, Path | None]:
    from .api import FilterRule

    dataset = filter_base(dataset)
    if not isinstance(rule, FilterRule):
        raise TypeError("rule must be a FilterRule.")
    if not isinstance(metrics, bool):
        raise TypeError("metrics must be a bool.")
    devices = resolve_devices(device)
    commit_samples = positive_int("commit_samples", commit_samples)
    max_shard_samples = optional_positive_int(
        "max_shard_samples",
        max_shard_samples,
    )

    spec = filter_spec(dataset)
    identity = filter_identity(dataset)
    cache = filter_cache_manager(dataset, cache_root).prepare(spec)
    base_count = len(dataset)
    expected = metadata(identity, base_count, rule)
    cache_path = filter_path(cache, rule, identity)
    metric_path = metrics_path(cache_path) if metrics else None

    if is_ready(cache_path, expected, metrics=metrics):
        return cache_path, metric_path

    lock_path = filter_lock_path(cache, rule, identity)
    with FileLock(lock_path):
        if is_ready(cache_path, expected, metrics=metrics):
            return cache_path, metric_path
        write_cache(
            cache_path,
            expected,
            dataset,
            rule,
            metrics=metrics,
            devices=devices,
            commit_samples=commit_samples,
            max_shard_samples=max_shard_samples,
        )
        return cache_path, metric_path


def filter_base(dataset: object) -> FilterBase:
    if isinstance(dataset, SampleDataset):
        return dataset
    raise TypeError("dataset must be a SampleDataset.")


def filter_spec(dataset: FilterBase) -> Spec:
    if isinstance(dataset, AnyDataset):
        return dataset.spec
    if isinstance(dataset, StoreDataset):
        return Spec(
            source=Source.STORE,
            path=str(dataset.root),
            split=dataset.manifest.split,
        )
    if _is_filtered_dataset(dataset):
        return filter_spec(dataset.base)
    raise TypeError("dataset must be an AnyDataset or StoreDataset.")


def filter_identity(dataset: FilterBase) -> dict[str, Any]:
    if _is_filtered_dataset(dataset):
        return {
            "view_schema_version": _FILTER_VIEW_SCHEMA_VERSION,
            "kind": "filtered",
            "base": filter_identity(dataset.base),
            "rule": {"name": dataset.rule.name},
            "labels": list(dataset.labels),
            "cache_key": dataset.cache_path.name,
            "sample_count": len(dataset),
        }
    spec = filter_spec(dataset)
    return {
        "kind": "physical",
        "spec_id": spec.id,
        "spec": spec.to_dict(),
    }


def filter_cache_manager(
    dataset: FilterBase,
    cache_root: str | Path | None,
) -> CacheManager:
    if cache_root is not None:
        return CacheManager(cache_root)
    if isinstance(dataset, AnyDataset):
        return dataset.cache_manager
    if _is_filtered_dataset(dataset):
        return filter_cache_manager(dataset.base, cache_root)
    return CacheManager()


def metadata(
    identity: Mapping[str, Any],
    base_count: int,
    rule: FilterRule,
) -> dict[str, Any]:
    output = {
        "schema_version": 4,
        "base": {
            "sample_count": base_count,
        },
        "rule": {"name": rule.name},
    }
    if identity.get("kind") == "physical":
        output["base"]["spec_id"] = identity["spec_id"]
    else:
        output["base"]["view"] = dict(identity)
    return output


def filter_path(
    cache: CacheManifest,
    rule: FilterRule,
    identity: Mapping[str, Any],
) -> Path:
    cache_name = rule.name
    if identity.get("kind") != "physical":
        cache_name = f"{filter_identity_key(identity)}:{rule.name}"
    return cache.cache_path / "filters" / rule_cache_key(cache_name)


def filter_lock_path(
    cache: CacheManifest,
    rule: FilterRule,
    identity: Mapping[str, Any],
) -> Path:
    path = filter_path(cache, rule, identity)
    return path.with_name(f".{path.name}.lock")


def filter_identity_key(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _is_filtered_dataset(dataset: object) -> bool:
    from .api import FilteredDataset

    return isinstance(dataset, FilteredDataset)


def metrics_path(cache_path: Path) -> Path:
    return cache_path / "metrics"


def is_ready(path: Path, expected: Mapping[str, Any], *, metrics: bool) -> bool:
    if not (path / ".ready").is_file():
        return False
    metadata_path = path / "rule.json"
    if not metadata_path.is_file():
        return False
    manifest_path = path / "partitions.json"
    if not manifest_path.is_file():
        return False
    if read_json(metadata_path) != expected:
        return False
    manifest = read_json(manifest_path)
    if not all((path / relpath).is_file() for relpath in partition_files(manifest)):
        return False
    if metrics and not metrics_ready(metrics_path(path)):
        return False
    return True


def write_cache(
    path: Path,
    metadata: Mapping[str, Any],
    dataset: FilterBase,
    rule: FilterRule,
    *,
    metrics: bool,
    devices: tuple[str, ...],
    commit_samples: int,
    max_shard_samples: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        write_json(tmp / "rule.json", dict(metadata))
        write_partitions(
            tmp,
            dataset,
            rule,
            metrics=metrics,
            devices=devices,
            commit_samples=commit_samples,
            max_shard_samples=max_shard_samples,
        )
        (tmp / ".ready").write_text("ready\n", encoding="utf-8")
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise


def write_partitions(
    path: Path,
    dataset: FilterBase,
    rule: FilterRule,
    *,
    metrics: bool,
    devices: tuple[str, ...],
    commit_samples: int,
    max_shard_samples: int | None,
) -> None:
    writer = PartitionWriter(path, max_shard_samples=max_shard_samples)
    metrics_writer = (
        MetricsWriter(metrics_path(path), max_shard_samples=max_shard_samples)
        if metrics
        else None
    )
    try:
        if len(devices) == 1 or len(dataset) == 0:
            for chunk in collect_ranges(
                dataset,
                rule.factory,
                devices[0],
                metrics,
                commit_samples,
            ):
                write_filter_chunk(writer, metrics_writer, chunk, metrics=metrics)
        else:
            for chunk in collect_ranges_parallel(
                dataset,
                rule.factory,
                devices,
                metrics,
                commit_samples,
            ):
                write_filter_chunk(writer, metrics_writer, chunk, metrics=metrics)
        writer.close()
        if metrics_writer is not None:
            metrics_writer.close()
    except Exception:
        writer.abort()
        if metrics_writer is not None:
            metrics_writer.abort()
        raise


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
