from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._io.atomic import replace_existing_dir
from .._devices import Devices, resolve_devices
from .._logging import write_info
from .._parallel import validate_process_value
from .._resume import (
    dataset_sample_count,
    indexes_complete,
    log_resume_summary,
    missing_indexes,
)
from .._validation import (
    non_negative_int,
    optional_positive_float,
    optional_positive_int,
    positive_int,
)
from .._write_pipeline import BackgroundWriteSink
from ..cache import FileLock, anydataset_home
from ..dataset.abc import (
    AnyDataset,
    MapStyleABC,
    MergedDataset,
    uses_default_indexed_shard,
)
from ..runtime import Runtime
from ..store.jsonio import read_json, write_json
from ..store.reader import StoreDataset
from ..types import Source, Spec
from .collect import collect_ranges, collect_ranges_parallel
from .resume import (
    cleanup_filter_resume_dir,
    completed_filter_indexes,
    iter_filter_fragment_chunks,
    prepare_filter_resume_dir,
    write_filter_fragment,
)
from .rules import rule_cache_key
from .storage import (
    MetricsWriter,
    PartitionWriter,
    metrics_ready,
    partition_files,
    read_partitions,
)
from .types import DatasetFactory, _FilterChunk, _FilterMetricsRow

if TYPE_CHECKING:
    from .api import FilterRule, _FilterCache

FilterBase = MapStyleABC

_DEFAULT_MAX_SHARD_SAMPLES = 1_000_000
_DEFAULT_COMMIT_SAMPLES = 100_000
_FILTER_VIEW_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _FilteredDatasetFactory:
    base: DatasetFactory
    rule_name: str
    labels: tuple[str, ...]
    cache_path: Path

    def __call__(self) -> FilterBase:
        from .api import FilteredDataset, FilterRule

        return FilteredDataset._from_partitions(
            self.base(),
            FilterRule(self.rule_name, _unavailable_filter_factory),
            self.cache_path,
            read_partitions(self.cache_path),
            self.labels,
            dataset_factory=self.base,
        )


def apply_filter(
    rule: FilterRule,
    *,
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
    cache_path, metric_path = ensure_filter(
        dataset,
        rule,
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
    return _FilterCache(
        dataset,
        read_partitions(cache_path),
        rule,
        cache_path,
        metrics_path=metric_path,
        dataset_factory=dataset_factory,
    )


def ensure_filter(
    dataset: FilterBase,
    rule: FilterRule,
    *,
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
) -> tuple[Path, Path | None]:
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

    identity = filter_identity(dataset)
    base_count = dataset_sample_count(dataset, context="filter")
    expected = metadata(identity, base_count, rule)
    cache_path = filter_path(rule, identity)
    metric_path = metrics_path(cache_path) if metrics else None

    reason = not_ready_reason(cache_path, expected, metrics=metrics)
    if reason is None:
        return cache_path, metric_path

    lock_path = filter_lock_path(rule, identity)
    with FileLock(lock_path):
        reason = not_ready_reason(cache_path, expected, metrics=metrics)
        if reason is None:
            return cache_path, metric_path
        log_filter_cache_miss(
            cache_path,
            rule,
            identity,
            base_count=base_count,
            metrics=metrics,
            reason=reason,
        )
        write_cache(
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
        return cache_path, metric_path


def filter_base(dataset: object) -> FilterBase:
    if isinstance(dataset, MapStyleABC):
        return dataset
    raise TypeError("dataset must be a MapStyleABC.")


def filter_universe(dataset: FilterBase) -> FilterBase:
    dataset = filter_base(dataset)
    if _is_filtered_dataset(dataset):
        return filter_universe(dataset.base)
    return dataset


def filter_spec(dataset: FilterBase) -> Spec:
    if isinstance(dataset, AnyDataset):
        return dataset.spec
    if isinstance(dataset, StoreDataset):
        return Spec(
            source=Source.STORE,
            path=str(dataset.root),
            split=dataset.manifest.split,
        )
    if isinstance(dataset, MergedDataset):
        return filter_spec(dataset.left)
    if _is_filtered_dataset(dataset):
        return filter_spec(dataset.base)
    raise TypeError("dataset must be an AnyDataset, StoreDataset, or MergedDataset.")


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
    if isinstance(dataset, MergedDataset):
        children = sorted(
            (dataset_identity(child) for child in merged_children(dataset)),
            key=filter_identity_key,
        )
        return {
            "view_schema_version": _FILTER_VIEW_SCHEMA_VERSION,
            "kind": "merged",
            "children": children,
            "sample_count": len(dataset),
        }
    spec = filter_spec(dataset)
    return {
        "kind": "physical",
        "type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "spec_id": spec.id,
        "spec": spec.to_dict(),
    }


def dataset_identity(dataset: Any) -> dict[str, Any]:
    if isinstance(dataset, MapStyleABC):
        return filter_identity(dataset)
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TypeError("merged dataset inputs must be map-style datasets.")
    return {
        "view_schema_version": _FILTER_VIEW_SCHEMA_VERSION,
        "kind": "map_style",
        "type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "sample_count": len(dataset),
        "object_id": id(dataset),
    }


def merged_children(dataset: MergedDataset) -> tuple[Any, ...]:
    children: list[Any] = []
    for child in (dataset.left, dataset.right):
        if isinstance(child, MergedDataset):
            children.extend(merged_children(child))
            continue
        children.append(child)
    return tuple(children)


def metadata(
    identity: Mapping[str, Any],
    base_count: int,
    rule: FilterRule,
) -> dict[str, Any]:
    output = {
        "schema_version": 5,
        "base": {
            "identity": dict(identity),
            "identity_id": filter_identity_key(identity),
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
    rule: FilterRule,
    identity: Mapping[str, Any],
) -> Path:
    return (
        anydataset_home()
        / "cache"
        / "filters"
        / filter_identity_key(identity)
        / rule_cache_key(rule.name)
    )


def filter_lock_path(
    rule: FilterRule,
    identity: Mapping[str, Any],
) -> Path:
    path = filter_path(rule, identity)
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
    return not_ready_reason(path, expected, metrics=metrics) is None


def not_ready_reason(path: Path, expected: Mapping[str, Any], *, metrics: bool) -> str | None:
    if not (path / ".ready").is_file():
        return "ready marker is missing"
    metadata_path = path / "rule.json"
    if not metadata_path.is_file():
        return "rule metadata is missing"
    manifest_path = path / "partitions.json"
    if not manifest_path.is_file():
        return "partition manifest is missing"
    if metadata_mismatch(read_json(metadata_path), expected):
        return "rule metadata does not match current dataset identity"
    manifest = read_json(manifest_path)
    for relpath in partition_files(manifest):
        if not (path / relpath).is_file():
            return f"partition shard is missing: {relpath}"
    if metrics and not metrics_ready(metrics_path(path)):
        return "metrics cache is missing or incomplete"
    return None


def metadata_mismatch(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return actual != expected


def log_filter_cache_miss(
    path: Path,
    rule: FilterRule,
    identity: Mapping[str, Any],
    *,
    base_count: int,
    metrics: bool,
    reason: str,
) -> None:
    fields = {
        "rule": rule.name,
        "cache_path": str(path),
        "identity_id": filter_identity_key(identity),
        "identity_kind": identity.get("kind"),
        "sample_count": base_count,
        "metrics": metrics,
        "reason": reason,
    }
    spec_id = identity.get("spec_id")
    if spec_id is not None:
        fields["spec_id"] = spec_id
    write_info(
        "filter",
        "building filter cache: "
        + " ".join(f"{key}={value!r}" for key, value in fields.items()),
    )


def write_cache(
    path: Path,
    metadata: Mapping[str, Any],
    dataset: FilterBase,
    rule: FilterRule,
    *,
    metrics: bool,
    devices: tuple[str, ...],
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
) -> None:
    replace_existing_dir(
        path,
        lambda tmp: _write_cache_tmp(
            tmp,
            path,
            metadata,
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
        ),
    )
    cleanup_filter_resume_dir(path)


def _write_cache_tmp(
    tmp: Path,
    cache_path: Path,
    metadata: Mapping[str, Any],
    dataset: FilterBase,
    rule: FilterRule,
    *,
    metrics: bool,
    devices: tuple[str, ...],
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
) -> None:
    write_json(tmp / "rule.json", dict(metadata))
    write_partitions(
        tmp,
        dataset,
        rule,
        cache_path=cache_path,
        metadata=metadata,
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
    (tmp / ".ready").write_text("ready\n", encoding="utf-8")


def write_partitions(
    path: Path,
    dataset: FilterBase,
    rule: FilterRule,
    *,
    cache_path: Path,
    metadata: Mapping[str, Any],
    metrics: bool,
    devices: tuple[str, ...],
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
) -> None:
    resume_dir = prepare_filter_resume_dir(cache_path, metadata, metrics=metrics)
    expected = dataset_sample_count(dataset, context="filter")
    completed = completed_filter_indexes(resume_dir, expected=expected)
    if not indexes_complete(completed, expected):
        missing = missing_indexes(completed, expected)
        use_map_style_loader = uses_default_indexed_shard(dataset)
        log_resume_summary(
            "filter",
            expected=expected,
            completed_count=len(completed),
            missing=missing,
            use_map_style_loader=use_map_style_loader,
        )
        writer = _FilterResumeFragmentWriter(
            path=resume_dir,
            dataset=dataset,
            rule=rule,
            metrics=metrics,
            devices=devices,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            commit_samples=commit_samples,
            runtime=runtime,
            dataset_factory=dataset_factory,
            completed=completed,
            missing=missing,
            worker_timeout=worker_timeout,
        )
        writer.write(write_workers=write_workers, write_prefetch=write_prefetch)
        completed = completed_filter_indexes(resume_dir, expected=expected)
    if not indexes_complete(completed, expected):
        raise RuntimeError("Filter resume fragments do not cover all samples.")
    replay_filter_resume_fragments(
        path,
        resume_dir,
        metrics=metrics,
        max_shard_samples=max_shard_samples,
    )


@dataclass(frozen=True)
class _FilterResumeFragmentWriter:
    path: Path
    dataset: FilterBase
    rule: FilterRule
    metrics: bool
    devices: tuple[str, ...]
    batch_size: int
    num_workers: int
    prefetch_factor: int | None
    commit_samples: int
    runtime: Runtime
    dataset_factory: DatasetFactory
    completed: frozenset[int]
    missing: tuple[int, ...]
    worker_timeout: float | None

    def write(self, *, write_workers: int, write_prefetch: int | None) -> None:
        sink = BackgroundWriteSink(
            write_filter_fragment_job,
            workers=write_workers,
            max_pending=write_prefetch,
            start_method=self.runtime.writer_worker_start_method,
        )
        with sink:
            self.write_jobs(sink)

    def write_jobs(self, sink: BackgroundWriteSink[FilterFragmentJob]) -> None:
        for chunk in self._chunks():
            sink.submit(self._job(chunk))

    def _chunks(self) -> Iterable[_FilterChunk]:
        use_map_style_loader = uses_default_indexed_shard(self.dataset)
        if len(self.devices) == 1 or len(self.dataset) == 0:
            yield from collect_ranges(
                self.dataset,
                self.rule.factory,
                self.devices[0],
                self.metrics,
                self.commit_samples,
                skip_indexes=self.completed,
                sample_indexes=self.missing if use_map_style_loader else None,
                dataset_factory=self.dataset_factory,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                runtime=self.runtime,
            )
            return

        factory = parallel_dataset_factory(self.dataset_factory, self.runtime)
        yield from collect_ranges_parallel(
            factory,
            self.rule.factory,
            self.devices,
            self.metrics,
            self.commit_samples,
            sample_count=len(self.dataset),
            skip_indexes=self.completed,
            sample_indexes=self.missing if use_map_style_loader else None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            runtime=self.runtime,
            use_map_style_loader=use_map_style_loader,
            worker_timeout=self.worker_timeout,
        )

    def _job(self, chunk: _FilterChunk) -> FilterFragmentJob:
        return FilterFragmentJob(
            path=self.path,
            scan_indexes=filter_chunk_indexes(chunk),
            chunk=global_filter_chunk(self.dataset, chunk),
        )


@dataclass(frozen=True)
class FilterFragmentJob:
    path: Path
    scan_indexes: tuple[int, ...]
    chunk: _FilterChunk


def write_filter_fragment_job(job: FilterFragmentJob) -> None:
    write_filter_fragment(job.path, job.scan_indexes, job.chunk)


def replay_filter_resume_fragments(
    path: Path,
    resume_dir: Path,
    *,
    metrics: bool,
    max_shard_samples: int | None,
) -> None:
    writer = PartitionWriter(path, max_shard_samples=max_shard_samples)
    metrics_writer = (
        MetricsWriter(metrics_path(path), max_shard_samples=max_shard_samples)
        if metrics
        else None
    )
    try:
        for chunk in iter_filter_fragment_chunks(resume_dir):
            write_filter_chunk(writer, metrics_writer, chunk, metrics=metrics)
        writer.close()
        if metrics_writer is not None:
            metrics_writer.close()
    except Exception:
        writer.abort()
        if metrics_writer is not None:
            metrics_writer.abort()
        raise


def filter_chunk_indexes(chunk: _FilterChunk) -> tuple[int, ...]:
    indexes = {
        int(index)
        for positions in chunk.partitions.values()
        for index in positions
    }
    return tuple(sorted(indexes))


def parallel_dataset_factory(factory: DatasetFactory, runtime: Runtime) -> DatasetFactory:
    validate_process_value(
        "dataset_factory",
        factory,
        context="multi-device filtering",
        start_method=runtime.process_start_method,
    )
    return factory


def make_filtered_dataset_factory(
    base: DatasetFactory,
    rule: FilterRule,
    labels: tuple[str, ...],
    cache_path: Path,
) -> _FilteredDatasetFactory:
    return _FilteredDatasetFactory(
        base=base,
        rule_name=rule.name,
        labels=labels,
        cache_path=Path(cache_path),
    )


def _unavailable_filter_factory():
    raise RuntimeError("cached filtered-view factory cannot rebuild its upstream rule.")


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


def global_filter_chunk(dataset: FilterBase, chunk: _FilterChunk) -> _FilterChunk:
    global_index = getattr(dataset, "global_index", None)
    if not callable(global_index):
        return chunk
    return _FilterChunk(
        partitions={
            label: tuple(global_index(position) for position in positions)
            for label, positions in chunk.partitions.items()
        },
        metrics=tuple(
            _FilterMetricsRow(
                index=global_index(row.index),
                label=row.label,
                metrics=row.metrics,
            )
            for row in chunk.metrics
        ),
    )
