from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._logging import write_info
from ..runtime import Runtime
from ..store.jsonio import read_json, write_json
from .generations import (
    FilterGeneration,
    GenerationUnavailable,
    cleanup_filter_generations_locked,
    create_filter_generation,
    lease_current_filter_generation,
)
from ._identity import FilterBase, filter_identity_key
from .resume import cleanup_filter_resume_dir
from .storage import metrics_ready, partition_count, partition_files
from .types import DatasetFactory

if TYPE_CHECKING:
    from .api import FilterRule


def metrics_path(cache_path: Path) -> Path:
    return cache_path / "metrics"


def ready_filter_generation(
    path: Path,
    expected: Mapping[str, Any],
    *,
    metrics: bool,
) -> tuple[FilterGeneration | None, str | None]:
    try:
        generation = lease_current_filter_generation(path)
    except FileNotFoundError:
        return None, "current generation pointer is missing"
    except (GenerationUnavailable, TypeError, ValueError) as exc:
        return None, f"current generation pointer is invalid: {exc}"
    reason = not_ready_reason(generation.path, expected, metrics=metrics)
    if reason is not None:
        generation.lease.close()
        return None, reason
    return generation, None


def is_ready(path: Path, expected: Mapping[str, Any], *, metrics: bool) -> bool:
    generation, _ = ready_filter_generation(path, expected, metrics=metrics)
    if generation is None:
        return False
    generation.lease.close()
    return True


def not_ready_reason(
    path: Path,
    expected: Mapping[str, Any],
    *,
    metrics: bool,
) -> str | None:
    if not (path / ".ready").is_file():
        return "ready marker is missing"
    metadata_path = path / "rule.json"
    if not metadata_path.is_file():
        return "rule metadata is missing"
    manifest_path = path / "partitions.json"
    if not manifest_path.is_file():
        return "partition manifest is missing"
    try:
        actual = read_json(metadata_path)
        manifest = read_json(manifest_path)
    except FileNotFoundError:
        return "cache snapshot changed during readiness check"
    if metadata_mismatch(actual, expected):
        return "rule metadata does not match current dataset identity"
    expected_count = int(expected["base"]["sample_count"])
    if partition_count(manifest) != expected_count:
        return "partition sample count does not match current dataset"
    for relpath in partition_files(manifest):
        if not (path / relpath).is_file():
            return f"partition shard is missing: {relpath}"
    if metrics:
        try:
            if not metrics_ready(metrics_path(path), expected_count=expected_count):
                return "metrics cache is missing or incomplete"
        except FileNotFoundError:
            return "cache snapshot changed during readiness check"
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
) -> FilterGeneration:
    generation = create_filter_generation(
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
    try:
        cleanup_filter_resume_dir(path)
        cleanup_filter_generations_locked(path)
        return generation
    except Exception:
        generation.lease.close()
        raise


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
    from ._resume_writer import write_partitions

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
