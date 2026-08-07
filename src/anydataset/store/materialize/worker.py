from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ..._runtime.logging import (
    use_run_logs_dir,
    worker_logger,
    write_info,
    write_warning,
)
from ..._runtime.performance import PipelineWorkerStats
from ..._runtime.parallel import (
    DeviceWorker,
    restore_environment,
    set_torch_device,
    set_worker_environment,
)
from ..._runtime.progress import Progress, ProgressWriter, put_progress
from ...runtime import Runtime
from ...types.item import Role, Schema, View
from .types import MaterializerProvider
from ..part.commit import commit_fragment_part, store_fragments

if TYPE_CHECKING:
    from .materializer import MaterializerWorker

DatasetFactory = Callable[[], Any]
ProviderFactory = Callable[[str], MaterializerProvider]
MaterializerMode = Literal["view", "modality", "sample"]


@dataclass(frozen=True)
class WorkerConfig:
    output_dir: Path
    dataset_id: str
    split: str | None
    provenance: Mapping[str, str]
    max_shard_samples: int
    max_shard_bytes: int | None
    batch_size: int
    commit_samples: int
    num_workers: int
    prefetch_factor: int | None
    write_workers: int
    write_prefetch: int | None
    keep_schema: Schema | None
    output: View | None
    schema: Schema | None
    roles: frozenset[Role] | None
    mode: MaterializerMode
    runtime: Runtime
    use_map_style_loader: bool
    completed_indexes: Sequence[int] | None
    missing_indexes: Sequence[int]
    fragments_dir: Path
    parts_dir: Path
    expected: int
    logs_dir: Path
    worker_logs_dir: Path
    device: str
    num_shards: int
    shard_id: int
    master_addr: str
    master_port: str
    finalize: bool


def materialize_worker(
    config: WorkerConfig,
    dataset_factory: DatasetFactory,
    provider_factory: ProviderFactory,
    progress: ProgressWriter[Progress],
    barrier: Any,
) -> None:
    with use_run_logs_dir(config.logs_dir):
        logger = worker_logger("materializer", config.worker_logs_dir, config.shard_id)
        logger.info(
            "starting shard %s/%s on %s missing=%s map_style=%s",
            config.shard_id,
            config.num_shards,
            config.device,
            shard_missing_count(
                config.missing_indexes, config.num_shards, config.shard_id
            ),
            config.use_map_style_loader,
        )
        env = set_worker_environment(
            DeviceWorker(
                device=config.device,
                rank=config.shard_id,
                world_size=config.num_shards,
                master_addr=config.master_addr,
                master_port=config.master_port,
            ),
            device_env="ANYDATASET_MATERIALIZE_DEVICE",
        )
        performance = PipelineWorkerStats(
            worker=config.shard_id,
            operation="provider",
            operation_name=type(provider_factory).__name__,
            requested_batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            device=config.device,
        )
        status = "complete"
        error_type: str | None = None
        error: str | None = None
        try:
            if config.runtime.uses_local_device:
                set_torch_device(config.device)
            logger.info("loading provider on %s", config.device)
            setup_started = time.perf_counter()
            try:
                provider = provider_factory(config.device)
            finally:
                performance.setup_seconds = time.perf_counter() - setup_started
            performance.operation_name = type(provider).__name__
            logger.info("loaded provider on %s", config.device)
            materializer = worker_materializer(config)
            logger.info("starting materialization on %s", config.device)
            materializer.write_batches(
                provider,
                dataset_factory=dataset_factory,
                sample_count=config.expected,
                use_map_style_loader=config.use_map_style_loader,
                completed_indexes=config.completed_indexes,
                sample_indexes=config.missing_indexes,
                fragments_dir=config.fragments_dir,
                expected=config.expected,
                progress=progress,
                worker_id=config.shard_id,
                performance=performance,
            )
            if config.finalize:
                logger.info("waiting to merge shard %s fragments", config.shard_id)
                barrier.wait()
                fragments = store_fragments(
                    config.fragments_dir,
                    dataset_id=materializer.dataset_id,
                    split=config.split,
                    shard_id=config.shard_id,
                    num_shards=config.num_shards,
                )
                commit_fragment_part(
                    config.parts_dir / f"part-{config.shard_id:05d}",
                    fragments,
                    dataset_id=materializer.dataset_id,
                    split=config.split,
                    shard_id=config.shard_id,
                    num_shards=config.num_shards,
                    max_shard_samples=config.max_shard_samples,
                    max_shard_bytes=config.max_shard_bytes,
                    provenance=config.provenance,
                )
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            error = traceback.format_exc()
            logger.error("worker failed\n%s", error)
            raise
        finally:
            fields = performance.fields(status=status, error_type=error_type)
            fields.update(
                {
                    "expected_samples": config.expected,
                    "target_samples": shard_missing_count(
                        config.missing_indexes,
                        config.num_shards,
                        config.shard_id,
                    ),
                    "commit_samples": config.commit_samples,
                    "write_workers": config.write_workers,
                    "write_prefetch": config.write_prefetch,
                }
            )
            elapsed = cast(float, fields["elapsed_seconds"])
            message = (
                "materializer worker summary: "
                f"worker={config.shard_id} status={status!r} "
                f"samples={performance.processed_samples} "
                f"provider_calls={performance.operation_calls.calls} "
                f"elapsed={elapsed:.3f}s"
            )
            write = write_info if status == "complete" else write_warning
            write(
                "materializer",
                message,
                event="materializer_worker_summary",
                fields=fields,
            )
            put_progress(
                progress,
                Progress(
                    config.shard_id,
                    0,
                    True,
                    error,
                    details=fields,
                ),
            )
            restore_environment(env)
        logger.info("finished shard %s", config.shard_id)


def worker_materializer(config: WorkerConfig) -> MaterializerWorker:
    from .materializer import (
        MaterializerWorker,
        ModalityMaterializer,
        SampleMaterializer,
        ViewMaterializer,
    )

    options: dict[str, Any] = {
        "output_dir": config.output_dir,
        "dataset_id": config.dataset_id,
        "split": config.split,
        "input_id": config.provenance.get("input_id"),
        "provider_id": config.provenance.get("provider_id"),
        "max_shard_samples": config.max_shard_samples,
        "max_shard_bytes": config.max_shard_bytes,
        "batch_size": config.batch_size,
        "commit_samples": config.commit_samples,
        "num_workers": config.num_workers,
        "prefetch_factor": config.prefetch_factor,
        "write_workers": config.write_workers,
        "write_prefetch": config.write_prefetch,
        "keep_schema": config.keep_schema,
        "output": config.output,
        "schema": config.schema,
        "runtime": config.runtime,
    }
    if config.mode == "modality":
        materializer = ModalityMaterializer(
            **options,
            roles=config.roles,
        )
    elif config.mode == "sample":
        materializer = SampleMaterializer(**options)
    elif config.mode == "view":
        materializer = ViewMaterializer(**options)
    else:
        raise ValueError(f"Unsupported materializer mode: {config.mode!r}.")
    return MaterializerWorker(materializer)


def shard_missing_count(indexes: Sequence[int], num_shards: int, shard_id: int) -> int:
    if shard_id >= len(indexes):
        return 0
    return (len(indexes) - 1 - shard_id) // num_shards + 1
