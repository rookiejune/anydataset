from __future__ import annotations

import multiprocessing
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .._logging import use_run_logs_dir, worker_logger
from .._parallel import (
    DeviceWorker,
    restore_environment,
    set_torch_device,
    set_worker_environment,
)
from .._progress import Progress, put_progress
from ..runtime import Runtime
from ..types.item import Schema
from ._types import MaterializerProvider
from ._part_commit import commit_fragment_part, store_fragments

DatasetFactory = Callable[[], Any]
ProviderFactory = Callable[[str], MaterializerProvider]
MaterializerMode = Literal["view", "modality"]


@dataclass(frozen=True)
class WorkerConfig:
    output_dir: Path
    split: str | None
    max_shard_samples: int
    batch_size: int
    commit_samples: int
    num_workers: int
    prefetch_factor: int | None
    write_workers: int
    write_prefetch: int | None
    keep_schema: Schema | None
    mode: MaterializerMode
    runtime: Runtime
    use_map_style_loader: bool
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


def materialize_worker(
    config: WorkerConfig,
    dataset_factory: DatasetFactory,
    provider_factory: ProviderFactory,
    progress: multiprocessing.Queue,
    barrier: Any,
) -> None:
    with use_run_logs_dir(config.logs_dir):
        logger = worker_logger("materializer", config.worker_logs_dir, config.shard_id)
        logger.info(
            "starting shard %s/%s on %s missing=%s map_style=%s",
            config.shard_id,
            config.num_shards,
            config.device,
            shard_missing_count(config.missing_indexes, config.num_shards, config.shard_id),
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
        try:
            if config.runtime.uses_local_device:
                set_torch_device(config.device)
            logger.info("loading provider on %s", config.device)
            provider = provider_factory(config.device)
            logger.info("loaded provider on %s", config.device)
            materializer = worker_materializer(config)
            logger.info("starting materialization on %s", config.device)
            materializer._write_resumable_loader_batches(
                provider,
                dataset_factory=dataset_factory,
                sample_count=config.expected,
                use_map_style_loader=config.use_map_style_loader,
                sample_indexes=config.missing_indexes,
                fragments_dir=config.fragments_dir,
                expected=config.expected,
                progress=progress,
                worker_id=config.shard_id,
            )
            logger.info("waiting to merge shard %s fragments", config.shard_id)
            barrier.wait()
            fragments = store_fragments(
                config.fragments_dir,
                dataset_id=materializer._dataset_id,
                split=config.split,
            )
            assigned = fragments[config.shard_id :: config.num_shards]
            commit_fragment_part(
                config.parts_dir / f"part-{config.shard_id:05d}",
                assigned,
                dataset_id=materializer._dataset_id,
                split=config.split,
                shard_id=config.shard_id,
                num_shards=config.num_shards,
            )
        except Exception:
            error = traceback.format_exc()
            logger.error("worker failed\n%s", error)
            put_progress(progress, Progress(config.shard_id, 0, True, error))
            raise
        finally:
            restore_environment(env)
        logger.info("finished shard %s", config.shard_id)
        put_progress(progress, Progress(config.shard_id, 0, True, None))


def worker_materializer(config: WorkerConfig):
    from .materializer import ModalityMaterializer, ViewMaterializer

    cls = ModalityMaterializer if config.mode == "modality" else ViewMaterializer
    return cls(
        config.output_dir,
        split=config.split,
        max_shard_samples=config.max_shard_samples,
        batch_size=config.batch_size,
        commit_samples=config.commit_samples,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        write_workers=config.write_workers,
        write_prefetch=config.write_prefetch,
        keep_schema=config.keep_schema,
        runtime=config.runtime,
    )


def shard_missing_count(indexes: Sequence[int], num_shards: int, shard_id: int) -> int:
    if shard_id >= len(indexes):
        return 0
    return (len(indexes) - 1 - shard_id) // num_shards + 1
