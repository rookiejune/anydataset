"""Materialize sample datasets into store directories.

The module coordinates single-process and part-based parallel writes. Store file
format details stay in `anydataset.store.writer` and store internals.
"""

from __future__ import annotations

import multiprocessing
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .._progress import Progress, iter_with_progress, put_progress, watch_workers
from .._validation import non_negative_int, optional_positive_int, positive_int
from .._parallel import (
    DeviceWorker,
    free_port,
    indexed_loader,
    multiprocessing_context,
    restore_environment,
    set_worker_environment,
    validate_spawn_value,
)
from ..store._config import DEFAULT_MAX_SHARD_SAMPLES
from ..store._part_commit import commit_store_parts
from ..store._part_writer import DatasetPartWriter
from ..store._sample_write import explicit_views
from ..store.writer import DatasetWriter
from ..types.item import Modality, Role, Sample, View

DatasetFactory = Callable[[], Any]


@dataclass
class DatasetStoreWriter:
    output_dir: str | Path
    dataset_id: str | None = None
    split: str | None = None
    views: tuple[tuple[Role, Modality, View], ...] | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES
    num_shards: int = 1
    num_workers: int = 0
    prefetch_factor: int | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.dataset_id is None:
            self.dataset_id = _dataset_id(self.output_dir)
        self.views = explicit_views(self.views)
        self.max_shard_samples = positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )
        self.num_shards = positive_int("num_shards", self.num_shards)
        self.num_workers = non_negative_int("num_workers", self.num_workers)
        self.prefetch_factor = optional_positive_int(
            "prefetch_factor",
            self.prefetch_factor,
        )

    def write(
        self,
        dataset: Any | None = None,
        *,
        dataset_factory: DatasetFactory | None = None,
    ) -> Path:
        if dataset_factory is None:
            if dataset is None:
                raise TypeError("write requires dataset or dataset_factory.")
            if self.num_shards > 1 or self.num_workers > 0:
                raise TypeError(
                    "dataset_factory is required when num_shards or num_workers "
                    "is greater than one."
                )
            return self._write_single(dataset)

        if dataset is not None:
            raise TypeError("write accepts either dataset or dataset_factory, not both.")
        if self.num_shards == 1 and self.num_workers == 0:
            return self._write_single(dataset_factory())
        return self._write_parts(dataset_factory)

    def _write_single(self, dataset: Any) -> Path:
        return DatasetWriter(
            self.output_dir,
            dataset_id=self.dataset_id or _dataset_id(self.output_dir),
            split=self.split,
            views=self.views,
            max_shard_samples=self.max_shard_samples,
        ).write(_ordered_samples(dataset))

    def _write_parts(self, dataset_factory: DatasetFactory) -> Path:
        validate_spawn_value(
            "dataset_factory",
            dataset_factory,
            context="parallel dataset write",
        )
        output_dir = _prepare_output_dir(self.output_dir.expanduser())
        if self.num_workers > 0:
            _prepare_loader_dataset(dataset_factory)
        with TemporaryDirectory(
            prefix=f".{output_dir.name}-parts-",
            dir=str(output_dir.parent),
        ) as tmpdir:
            parts_dir = Path(tmpdir)
            self._run_parts(dataset_factory, parts_dir)
            return commit_store_parts(
                self.output_dir,
                parts_dir,
                dataset_id=self.dataset_id or _dataset_id(self.output_dir),
                split=self.split,
            )

    def _run_parts(self, dataset_factory: DatasetFactory, parts_dir: Path) -> None:
        context = multiprocessing_context()
        progress = context.Queue()
        master_addr = "127.0.0.1"
        master_port = free_port()
        workers = [
            context.Process(
                target=_write_worker,
                args=(
                    _WorkerConfig(
                        output_dir=self.output_dir,
                        dataset_id=self.dataset_id or _dataset_id(self.output_dir),
                        split=self.split,
                        views=self.views,
                        max_shard_samples=self.max_shard_samples,
                        num_shards=self.num_shards,
                        shard_id=shard_id,
                        num_workers=self.num_workers,
                        prefetch_factor=self.prefetch_factor,
                        parts_dir=parts_dir,
                        master_addr=master_addr,
                        master_port=master_port,
                    ),
                    dataset_factory,
                    progress,
                ),
                name=f"anydataset-write-{shard_id}",
            )
            for shard_id in range(self.num_shards)
        ]
        started: list[multiprocessing.Process] = []
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


def _prepare_loader_dataset(dataset_factory: DatasetFactory) -> None:
    dataset = dataset_factory()
    prepare = getattr(dataset, "prepare", None)
    if callable(prepare):
        prepare()


@dataclass(frozen=True)
class _WorkerConfig:
    output_dir: Path
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


def _ordered_samples(dataset: Any) -> Iterator[Sample]:
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        for index in range(len(dataset)):
            yield dataset[index]
        return
    yield from dataset


def _prepare_output_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Target path exists and is not a directory: {output_dir}")
        entries = [entry for entry in output_dir.iterdir()]
        if entries:
            raise ValueError(f"Target directory must be empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    return output_dir


def _dataset_id(output_dir: str | Path) -> str:
    return Path(output_dir).expanduser().name or "dataset"
