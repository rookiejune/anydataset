"""Internal parallel worker helpers for `anydataset.store.DatasetWriter`."""

from __future__ import annotations

import multiprocessing
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..._runtime.progress import Progress, iter_with_progress, put_progress, watch_workers
from ..._io.atomic import validate_empty_target
from ..._runtime.parallel import (
    DeviceWorker,
    ProcessHandle,
    free_port,
    indexed_loader,
    iter_ordered_samples,
    multiprocessing_context,
    restore_environment,
    set_worker_environment,
    validate_spawn_value,
)
from .commit import commit_store_parts
from .writer import DatasetPartWriter
from ...types.item import Modality, Role, Sample, View

DatasetFactory = Callable[[], Any]


def write_dataset_parts(
    output_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None,
    views: tuple[tuple[Role, Modality, View], ...] | None,
    max_shard_samples: int,
    num_shards: int,
    num_workers: int,
    prefetch_factor: int | None,
    provenance: Mapping[str, str],
    dataset_factory: DatasetFactory,
) -> Path:
    validate_spawn_value(
        "dataset_factory",
        dataset_factory,
        context="parallel dataset write",
    )
    output = Path(output_dir).expanduser()
    validate_empty_target(output)
    output.mkdir(exist_ok=True)
    if num_workers > 0:
        _prepare_loader_dataset(dataset_factory)
    with TemporaryDirectory(
        prefix=f".{output.name}-parts-",
        dir=str(output.parent),
    ) as tmpdir:
        parts_dir = Path(tmpdir)
        _run_parts(
            parts_dir,
            dataset_id=dataset_id,
            split=split,
            views=views,
            max_shard_samples=max_shard_samples,
            num_shards=num_shards,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            dataset_factory=dataset_factory,
        )
        return commit_store_parts(
            output,
            parts_dir,
            dataset_id=dataset_id,
            split=split,
            provenance=provenance,
        )


def _run_parts(
    parts_dir: Path,
    *,
    dataset_id: str,
    split: str | None,
    views: tuple[tuple[Role, Modality, View], ...] | None,
    max_shard_samples: int,
    num_shards: int,
    num_workers: int,
    prefetch_factor: int | None,
    dataset_factory: DatasetFactory,
) -> None:
    context = multiprocessing_context()
    progress = context.Queue()
    try:
        master_addr = "127.0.0.1"
        master_port = free_port()
        workers = [
            context.Process(
                target=_write_worker,
                args=(
                    _WorkerConfig(
                        dataset_id=dataset_id,
                        split=split,
                        views=views,
                        max_shard_samples=max_shard_samples,
                        num_shards=num_shards,
                        shard_id=shard_id,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor,
                        parts_dir=parts_dir,
                        master_addr=master_addr,
                        master_port=master_port,
                    ),
                    dataset_factory,
                    progress,
                ),
                name=f"anydataset-write-{shard_id}",
            )
            for shard_id in range(num_shards)
        ]
        started: list[ProcessHandle] = []
        completed = False
        try:
            for worker in workers:
                worker.start()
                started.append(worker)
            watch_workers(
                workers,
                progress,
                desc="write dataset",
                early_exit_message="Dataset write worker exited early.",
                failure_prefix="Dataset write worker",
            )
            completed = True
        finally:
            if not completed:
                for worker in started:
                    if worker.is_alive():
                        worker.terminate()
            for worker in started:
                worker.join()

        failed = [worker for worker in workers if worker.exitcode != 0]
        if failed:
            details = ", ".join(
                f"{worker.name} exited {worker.exitcode}" for worker in failed
            )
            raise RuntimeError(f"Dataset write workers failed: {details}.")
    finally:
        _close_queue(progress, suppress_errors=sys.exc_info()[0] is not None)


def _close_queue(progress: Any, *, suppress_errors: bool) -> None:
    error: Exception | None = None
    for cleanup in (progress.close, progress.join_thread):
        try:
            cleanup()
        except Exception as exc:
            if error is None:
                error = exc
    if error is not None and not suppress_errors:
        raise error


def _prepare_loader_dataset(dataset_factory: DatasetFactory) -> None:
    dataset = dataset_factory()
    prepare = getattr(dataset, "prepare", None)
    if callable(prepare):
        prepare()


@dataclass(frozen=True)
class _WorkerConfig:
    dataset_id: str
    split: str | None
    views: tuple[tuple[Role, Modality, View], ...] | None
    max_shard_samples: int
    num_shards: int
    shard_id: int
    num_workers: int
    prefetch_factor: int | None
    parts_dir: Path
    master_addr: str
    master_port: str


def _write_worker(
    config: _WorkerConfig,
    dataset_factory: DatasetFactory,
    progress: multiprocessing.Queue,
) -> None:
    env = set_worker_environment(
        DeviceWorker(
            device=str(config.shard_id),
            rank=config.shard_id,
            world_size=config.num_shards,
            master_addr=config.master_addr,
            master_port=config.master_port,
        ),
        device_env="ANYDATASET_WRITE_SHARD",
    )
    try:
        writer = DatasetPartWriter(
            config.parts_dir / f"part-{config.shard_id:05d}",
            dataset_id=config.dataset_id,
            split=config.split,
            shard_id=config.shard_id,
            num_shards=config.num_shards,
            views=config.views,
            max_shard_samples=config.max_shard_samples,
        )
        writer.write(
            iter_with_progress(
                _indexed_samples(
                    dataset_factory,
                    num_workers=config.num_workers,
                    prefetch_factor=config.prefetch_factor,
                ),
                worker_id=config.shard_id,
                progress=progress,
            )
        )
    except Exception:
        put_progress(
            progress,
            Progress(config.shard_id, 0, True, traceback.format_exc()),
        )
        raise
    finally:
        restore_environment(env)
    put_progress(progress, Progress(config.shard_id, 0, True, None))


def _indexed_samples(
    dataset_factory: DatasetFactory,
    *,
    num_workers: int,
    prefetch_factor: int | None,
) -> Iterator[tuple[int, Sample]]:
    for batch in indexed_loader(
        dataset_factory,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    ):
        yield from batch


def ordered_samples(dataset: Any) -> Iterator[Sample]:
    yield from iter_ordered_samples(dataset)
