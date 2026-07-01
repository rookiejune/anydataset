from __future__ import annotations

import logging
import multiprocessing
import os
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from torch import distributed as dist
from torch.utils.data import DataLoader

from .._devices import Devices, resolve_devices
from .._logging import run_logs_dir, use_run_logs_dir
from .._parallel import (
    DeviceWorker,
    cuda_device,
    free_port,
    indexed_loader,
    iter_indexed_shard,
    multiprocessing_context,
    restore_environment,
    set_single_worker_environment,
    set_torch_device,
    set_worker_environment,
    validate_process_value,
)
from .._progress import Progress, ProgressDashboard, put_progress, watch_workers
from .._resume import (
    cleanup_resume_dir,
    dataset_sample_count,
    index_batch_id,
    indexes_complete,
    pending_batch,
    prepare_resume_dir,
    validate_completed_indexes,
)
from .._validation import non_negative_int, optional_positive_int, positive_int
from .._write_pipeline import BackgroundWriteSink
from ..runtime import Runtime
from ..types.item import Sample
from ..view import Provider
from ._batch import (
    indexed_sample_batches,
    validate_batch_outputs,
    with_batch_modality_provider,
    with_batch_view_provider,
    with_resilient_batch_provider,
)
from ._modality import with_modality_provider
from ._types import MaterializerProvider, ModalityProviderLike
from ._view import with_view_provider
from .parts import (
    DatasetFragmentWriter,
    commit_store_fragments,
    completed_fragment_indexes,
)
from .writer import DEFAULT_MAX_SHARD_SAMPLES, DatasetWriter

type DatasetFactory = Callable[[], Any]
type ProviderFactory = Callable[[str], MaterializerProvider]
type _MaterializerMode = Literal["view", "modality"]
type _ProgressSink = multiprocessing.Queue | ProgressDashboard

_PROGRESS_STAGES = ("reader", "provider", "writer")


@dataclass
class ViewMaterializer:
    output_dir: str | Path
    split: str | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES
    batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: int | None = None
    write_workers: int = 1
    write_prefetch: int | None = None
    runtime: Runtime = field(default_factory=Runtime)

    def __post_init__(self) -> None:
        self.max_shard_samples = positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )
        self.batch_size = positive_int("batch_size", self.batch_size)
        self.num_workers = non_negative_int("num_workers", self.num_workers)
        self.prefetch_factor = optional_positive_int(
            "prefetch_factor",
            self.prefetch_factor,
        )
        self.write_workers = non_negative_int("write_workers", self.write_workers)
        self.write_prefetch = optional_positive_int(
            "write_prefetch",
            self.write_prefetch,
        )

    @property
    def _dataset_id(self) -> str:
        return _dataset_id(self.output_dir)

    def write(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: Devices = "auto",
    ) -> Path:
        resolved = resolve_devices(devices)
        return self._write_resumable(
            dataset_factory=dataset_factory,
            provider_factory=provider_factory,
            devices=resolved,
        )

    def _write_resumable(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: tuple[str, ...],
    ) -> Path:
        if len(devices) == 1:
            device = devices[0]
            if self.runtime.uses_local_device:
                set_torch_device(device)
            return self._write_resumable_single(
                dataset_factory=dataset_factory,
                provider_factory=provider_factory,
                device=device,
            )
        return self._write_resumable_devices(
            dataset_factory=dataset_factory,
            provider_factory=provider_factory,
            devices=devices,
        )

    def _write_resumable_devices(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: tuple[str, ...],
    ) -> Path:
        validate_process_value(
            "dataset_factory",
            dataset_factory,
            context="multi-device materialization",
            start_method=self.runtime.process_start_method,
        )
        validate_process_value(
            "provider_factory",
            provider_factory,
            context="multi-device materialization",
            start_method=self.runtime.process_start_method,
        )
        expected = dataset_sample_count(dataset_factory(), context="resume")
        fragments_dir = prepare_resume_dir(self.output_dir, "fragments")
        completed = validate_completed_indexes(
            completed_fragment_indexes(
                fragments_dir,
                dataset_id=self._dataset_id,
                split=self.split,
            ),
            expected,
        )
        if indexes_complete(completed, expected):
            return self._commit_fragments(fragments_dir, expected)

        logs_dir = run_logs_dir()
        worker_logs_dir = logs_dir / "materializer"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self._run_parallel_parts(
            dataset_factory=dataset_factory,
            provider_factory=provider_factory,
            devices=devices,
            logs_dir=logs_dir,
            worker_logs_dir=worker_logs_dir,
            fragments_dir=fragments_dir,
            expected=expected,
            completed_count=len(completed),
        )
        return self._commit_fragments(fragments_dir, expected)

    def _write_resumable_single(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        device: str,
    ) -> Path:
        output_dir = Path(self.output_dir).expanduser()
        fragments_dir = prepare_resume_dir(output_dir, "fragments")
        dataset = dataset_factory()
        expected = dataset_sample_count(dataset, context="resume")
        completed = validate_completed_indexes(
            completed_fragment_indexes(
                fragments_dir,
                dataset_id=self._dataset_id,
                split=self.split,
            ),
            expected,
        )
        if indexes_complete(completed, expected):
            return self._commit_fragments(fragments_dir, expected)

        with ProgressDashboard(
            desc="materialize views",
            total=expected,
            count_stage="writer",
            initial=len(completed),
            stages=_PROGRESS_STAGES,
        ) as progress:
            provider = provider_factory(device)
            if self.num_workers > 0:
                env = set_single_worker_environment(
                    device,
                    device_env="ANYDATASET_MATERIALIZE_DEVICE",
                )
                try:
                    self._write_resumable_loader_batches(
                        provider,
                        dataset_factory=dataset_factory,
                        fragments_dir=fragments_dir,
                        expected=expected,
                        progress=progress,
                    )
                finally:
                    restore_environment(env)
            else:
                self._write_resumable_indexed_batches(
                    indexed_sample_batches(
                        iter_indexed_shard(dataset, 1, 0),
                        self.batch_size,
                    ),
                    provider,
                    fragments_dir=fragments_dir,
                    expected=expected,
                    progress=progress,
                )
        return self._commit_fragments(fragments_dir, expected)

    def _commit_fragments(self, fragments_dir: str | Path, expected: int) -> Path:
        if expected == 0:
            path = DatasetWriter(
                self.output_dir,
                dataset_id=self._dataset_id,
                split=self.split,
                max_shard_samples=self.max_shard_samples,
            ).write(())
            cleanup_resume_dir(self.output_dir)
            return path
        path = commit_store_fragments(
            self.output_dir,
            fragments_dir,
            dataset_id=self._dataset_id,
            split=self.split,
            expected_sample_count=expected,
        )
        cleanup_resume_dir(self.output_dir)
        return path

    def _write_resumable_loader_batches(
        self,
        provider: MaterializerProvider,
        *,
        dataset_factory: DatasetFactory,
        fragments_dir: Path,
        expected: int,
        progress: _ProgressSink | None = None,
        worker_id: int = 0,
    ) -> None:
        self._write_resumable_indexed_batches(
            self._loader(dataset_factory=dataset_factory),
            provider,
            fragments_dir=fragments_dir,
            expected=expected,
            progress=progress,
            worker_id=worker_id,
        )

    def _loader(
        self,
        *,
        dataset_factory: DatasetFactory,
    ) -> DataLoader:
        return indexed_loader(
            dataset_factory,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            start_method=self.runtime.loader_start_method,
        )

    def _run_parallel_parts(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: tuple[str, ...],
        logs_dir: Path,
        worker_logs_dir: Path,
        fragments_dir: Path,
        expected: int,
        completed_count: int,
    ) -> None:
        context = multiprocessing_context(self.runtime.process_start_method)
        progress = context.Queue()
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", free_port())
        workers = [
            context.Process(
                target=_materialize_worker,
                args=(
                    _WorkerConfig(
                        output_dir=Path(self.output_dir),
                        split=self.split,
                        max_shard_samples=self.max_shard_samples,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        prefetch_factor=self.prefetch_factor,
                        write_workers=self.write_workers,
                        write_prefetch=self.write_prefetch,
                        mode=self._materializer_mode,
                        runtime=self.runtime,
                        fragments_dir=fragments_dir,
                        expected=expected,
                        logs_dir=logs_dir,
                        worker_logs_dir=worker_logs_dir,
                        device=device,
                        num_shards=len(devices),
                        shard_id=shard_id,
                        master_addr=master_addr,
                        master_port=master_port,
                    ),
                    dataset_factory,
                    provider_factory,
                    progress,
                ),
                name=f"anydataset-materialize-{shard_id}",
            )
            for shard_id, device in enumerate(devices)
        ]
        for worker in workers:
            worker.start()
        try:
            watch_workers(
                workers,
                progress,
                desc="materialize views",
                early_exit_message="View materialization worker exited early.",
                failure_prefix="View materialization worker",
                total=expected,
                count_stage="writer",
                initial=completed_count,
                stages=_PROGRESS_STAGES,
            )
        except Exception:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in workers:
                worker.join()
            raise
        for worker in workers:
            worker.join()

        failed = [worker for worker in workers if worker.exitcode != 0]
        if failed:
            details = ", ".join(
                f"{worker.name} exited {worker.exitcode}" for worker in failed
            )
            raise RuntimeError(f"View materialization workers failed: {details}.")

    @property
    def _materializer_mode(self) -> _MaterializerMode:
        return "view"

    def _sample_with_provider(
        self,
        sample: Sample,
        provider: MaterializerProvider,
    ) -> Sample:
        return with_view_provider(sample, cast(Provider, provider))

    def _indexed_samples(
        self,
        dataset: Any,
        provider: MaterializerProvider,
        *,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Sample]]:
        indexed = iter_indexed_shard(dataset, num_shards, shard_id)
        if self.batch_size == 1:
            for index, sample in indexed:
                yield index, self._sample_with_provider(sample, provider)
            return

        for batch in indexed_sample_batches(indexed, self.batch_size):
            indexes, samples = zip(*batch, strict=True)
            outputs = tuple(
                self._resilient_samples_with_batch_provider(samples, provider)
            )
            validate_batch_outputs(outputs, len(samples))
            yield from zip(indexes, outputs, strict=True)

    def _write_resumable_indexed_batches(
        self,
        batches: Iterable[Sequence[tuple[int, Sample]]],
        provider: MaterializerProvider,
        *,
        fragments_dir: Path,
        expected: int,
        progress: _ProgressSink | None = None,
        worker_id: int = 0,
    ) -> None:
        completed = set(
            validate_completed_indexes(
                completed_fragment_indexes(
                    fragments_dir,
                    dataset_id=self._dataset_id,
                    split=self.split,
                ),
                expected,
            )
        )
        with self._fragment_sink(progress=progress, worker_id=worker_id) as sink:
            read_start = time.perf_counter()
            for batch in batches:
                _put_stage_progress(
                    progress,
                    worker_id=worker_id,
                    stage="reader",
                    samples=len(batch),
                    elapsed=time.perf_counter() - read_start,
                )
                pending = pending_batch(batch, frozenset(completed))
                if not pending:
                    read_start = time.perf_counter()
                    continue
                provider_start = time.perf_counter()
                outputs = self._materialized_indexed_batch(pending, provider)
                _put_stage_progress(
                    progress,
                    worker_id=worker_id,
                    stage="provider",
                    samples=len(outputs),
                    elapsed=time.perf_counter() - provider_start,
                )
                indexes = tuple(sorted(index for index, _ in outputs))
                job = _FragmentWriteJob(
                    fragments_dir=fragments_dir,
                    dataset_id=self._dataset_id,
                    split=self.split,
                    max_shard_samples=self.max_shard_samples,
                    indexes=indexes,
                    samples=outputs,
                )
                sink.submit(job)
                completed.update(indexes)
                read_start = time.perf_counter()

    def _fragment_sink(
        self,
        *,
        progress: _ProgressSink | None = None,
        worker_id: int = 0,
    ) -> BackgroundWriteSink[_FragmentWriteJob]:
        return BackgroundWriteSink(
            _write_fragment,
            workers=self.write_workers,
            max_pending=self.write_prefetch,
            start_method=self.runtime.process_start_method,
            on_submit=lambda job, pending: _put_stage_progress(
                progress,
                worker_id=worker_id,
                stage="writer",
                pending=pending,
            ),
            on_complete=lambda job, pending, elapsed: _put_stage_progress(
                progress,
                worker_id=worker_id,
                stage="writer",
                samples=len(job.samples),
                elapsed=elapsed,
                pending=pending,
            ),
        )

    def _materialized_indexed_batch(
        self,
        batch: Sequence[tuple[int, Sample]],
        provider: MaterializerProvider,
    ) -> tuple[tuple[int, Sample], ...]:
        if self.batch_size == 1:
            return tuple(
                (index, self._sample_with_provider(sample, provider))
                for index, sample in batch
            )

        indexes, samples = zip(*batch, strict=True)
        outputs = tuple(self._resilient_samples_with_batch_provider(samples, provider))
        validate_batch_outputs(outputs, len(samples))
        return tuple(zip(indexes, outputs, strict=True))

    def _samples_with_batch_provider(
        self,
        samples: Sequence[Sample],
        provider: MaterializerProvider,
    ) -> Iterator[Sample]:
        return with_batch_view_provider(samples, cast(Provider, provider))

    def _resilient_samples_with_batch_provider(
        self,
        samples: Sequence[Sample],
        provider: MaterializerProvider,
    ) -> Iterator[Sample]:
        yield from with_resilient_batch_provider(
            samples,
            lambda batch: tuple(self._samples_with_batch_provider(batch, provider)),
        )


@dataclass
class ModalityMaterializer(ViewMaterializer):
    @property
    def _materializer_mode(self) -> _MaterializerMode:
        return "modality"

    def _sample_with_provider(
        self,
        sample: Sample,
        provider: MaterializerProvider,
    ) -> Sample:
        return with_modality_provider(sample, cast(ModalityProviderLike, provider))

    def _samples_with_batch_provider(
        self,
        samples: Sequence[Sample],
        provider: MaterializerProvider,
    ) -> Iterator[Sample]:
        return with_batch_modality_provider(
            samples, cast(ModalityProviderLike, provider)
        )


@dataclass(frozen=True)
class _FragmentWriteJob:
    fragments_dir: Path
    dataset_id: str
    split: str | None
    max_shard_samples: int
    indexes: tuple[int, ...]
    samples: tuple[tuple[int, Sample], ...]


def _write_fragment(job: _FragmentWriteJob) -> None:
    fragment_id = index_batch_id(job.indexes)
    DatasetFragmentWriter(
        job.fragments_dir / fragment_id,
        dataset_id=job.dataset_id,
        split=job.split,
        fragment_id=fragment_id,
        max_shard_samples=job.max_shard_samples,
    ).write(job.samples)


@dataclass(frozen=True)
class _WorkerConfig:
    output_dir: Path
    split: str | None
    max_shard_samples: int
    batch_size: int
    num_workers: int
    prefetch_factor: int | None
    write_workers: int
    write_prefetch: int | None
    mode: _MaterializerMode
    runtime: Runtime
    fragments_dir: Path
    expected: int
    logs_dir: Path
    worker_logs_dir: Path
    device: str
    num_shards: int
    shard_id: int
    master_addr: str
    master_port: str


def _init_worker_process_group(config: _WorkerConfig) -> bool:
    if config.num_shards <= 1:
        return False
    if not dist.is_available() or dist.is_initialized():
        return False
    dist.init_process_group(
        backend=_distributed_backend(config.device),
        init_method=f"tcp://{config.master_addr}:{config.master_port}",
        rank=config.shard_id,
        world_size=config.num_shards,
    )
    return True


def _distributed_backend(device: str) -> str:
    return (
        "nccl"
        if cuda_device(device) is not None and dist.is_nccl_available()
        else "gloo"
    )


def _materialize_worker(
    config: _WorkerConfig,
    dataset_factory: DatasetFactory,
    provider_factory: ProviderFactory,
    progress: multiprocessing.Queue,
) -> None:
    with use_run_logs_dir(config.logs_dir):
        logger = _worker_logger(config.worker_logs_dir, config.shard_id)
        logger.info(
            "starting shard %s/%s on %s",
            config.shard_id,
            config.num_shards,
            config.device,
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
        process_group_created = False
        try:
            if config.runtime.uses_local_device:
                set_torch_device(config.device)
                process_group_created = _init_worker_process_group(config)
            provider = provider_factory(config.device)
            materializer = _worker_materializer(config)
            materializer._write_resumable_loader_batches(
                provider,
                dataset_factory=dataset_factory,
                fragments_dir=config.fragments_dir,
                expected=config.expected,
                progress=progress,
                worker_id=config.shard_id,
            )
        except Exception:
            error = traceback.format_exc()
            logger.error("worker failed\n%s", error)
            put_progress(progress, Progress(config.shard_id, 0, True, error))
            raise
        finally:
            if process_group_created:
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
            restore_environment(env)
        logger.info("finished shard %s", config.shard_id)
        put_progress(progress, Progress(config.shard_id, 0, True, None))


def _worker_materializer(config: _WorkerConfig) -> ViewMaterializer:
    cls = ModalityMaterializer if config.mode == "modality" else ViewMaterializer
    return cls(
        config.output_dir,
        split=config.split,
        max_shard_samples=config.max_shard_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        write_workers=config.write_workers,
        write_prefetch=config.write_prefetch,
        runtime=config.runtime,
    )


def _worker_logger(logs_dir: Path, shard_id: int) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"anydataset.materializer.{os.getpid()}.{shard_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(
        logs_dir / f"part-{shard_id:05d}.log",
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(processName)s %(message)s")
    )
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def _put_stage_progress(
    progress: _ProgressSink | None,
    *,
    worker_id: int,
    stage: str,
    samples: int = 0,
    elapsed: float | None = None,
    pending: int | None = None,
) -> None:
    if progress is None:
        return
    put_progress(
        progress,
        Progress(
            worker_id,
            samples,
            False,
            None,
            stage=stage,
            elapsed=elapsed,
            pending=pending,
        ),
    )


def _dataset_id(output_dir: str | Path) -> str:
    return Path(output_dir).expanduser().name or "dataset"
