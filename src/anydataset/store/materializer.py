from __future__ import annotations

import logging
import multiprocessing
import os
import shutil
import traceback
import hashlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
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
    validate_spawn_value,
)
from .._progress import Progress, iter_with_progress, put_progress, watch_workers
from .._validation import non_negative_int, optional_positive_int, positive_int
from ..types.item import Sample
from ..view import Provider
from ._batch import (
    indexed_sample_batches,
    sample_batches,
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
    DatasetPartWriter,
    commit_store_fragments,
    commit_store_parts,
    completed_fragment_indexes,
)
from .writer import DEFAULT_MAX_SHARD_SAMPLES, DatasetWriter

type DatasetFactory = Callable[[], Any]
type ProviderFactory = Callable[[str], MaterializerProvider]
type _MaterializerMode = Literal["view", "modality"]


@dataclass
class ViewMaterializer:
    output_dir: str | Path
    split: str | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES
    batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: int | None = None

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

    @property
    def dataset_id(self) -> str:
        return _dataset_id(self.output_dir)

    def write(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: Devices = "auto",
        resume: bool = False,
    ) -> Path:
        resolved = resolve_devices(devices)
        if resume:
            return self._write_resumable(
                dataset_factory=dataset_factory,
                provider_factory=provider_factory,
                devices=resolved,
            )
        if len(resolved) == 1:
            device = resolved[0]
            set_torch_device(device)
            return self._write_single(
                dataset_factory=dataset_factory,
                provider=provider_factory(device),
                device=device,
            )
        return self._write_devices(
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

    def _write_devices(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: tuple[str, ...],
    ) -> Path:
        validate_spawn_value(
            "dataset_factory",
            dataset_factory,
            context="multi-device materialization",
        )
        validate_spawn_value(
            "provider_factory",
            provider_factory,
            context="multi-device materialization",
        )
        output_dir = _prepare_parallel_output_dir(Path(self.output_dir).expanduser())
        logs_dir = run_logs_dir()
        worker_logs_dir = logs_dir / "materializer"
        logs_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=f".{output_dir.name}-parts-",
            dir=str(output_dir.parent),
        ) as tmpdir:
            parts_dir = Path(tmpdir)
            self._run_parallel_parts(
                dataset_factory=dataset_factory,
                provider_factory=provider_factory,
                devices=devices,
                parts_dir=parts_dir,
                logs_dir=logs_dir,
                worker_logs_dir=worker_logs_dir,
            )
            return self._commit_parts(parts_dir)

    def _write_resumable_devices(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: tuple[str, ...],
    ) -> Path:
        validate_spawn_value(
            "dataset_factory",
            dataset_factory,
            context="multi-device materialization",
        )
        validate_spawn_value(
            "provider_factory",
            provider_factory,
            context="multi-device materialization",
        )
        expected = _dataset_sample_count(dataset_factory())
        fragments_dir = _prepare_resume_fragments_dir(Path(self.output_dir).expanduser())
        completed = completed_fragment_indexes(
            fragments_dir,
            dataset_id=self.dataset_id,
            split=self.split,
        )
        _validate_completed_indexes(completed, expected)
        if _complete(completed, expected):
            return self._commit_fragments(fragments_dir, expected)

        logs_dir = run_logs_dir()
        worker_logs_dir = logs_dir / "materializer"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self._run_parallel_parts(
            dataset_factory=dataset_factory,
            provider_factory=provider_factory,
            devices=devices,
            parts_dir=fragments_dir,
            logs_dir=logs_dir,
            worker_logs_dir=worker_logs_dir,
            resume=True,
            fragments_dir=fragments_dir,
        )
        return self._commit_fragments(fragments_dir, expected)

    def _write_single(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider: MaterializerProvider,
        device: str,
    ) -> Path:
        if self.num_workers > 0:
            return self._write_single_part(
                provider,
                device=device,
                dataset_factory=dataset_factory,
            )
        dataset = dataset_factory()
        return DatasetWriter(
            self.output_dir,
            dataset_id=self.dataset_id,
            split=self.split,
            max_shard_samples=self.max_shard_samples,
        ).write(self._samples(dataset, provider))

    def _write_resumable_single(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        device: str,
    ) -> Path:
        output_dir = Path(self.output_dir).expanduser()
        fragments_dir = _prepare_resume_fragments_dir(output_dir)
        dataset = dataset_factory()
        expected = _dataset_sample_count(dataset)
        completed = completed_fragment_indexes(
            fragments_dir,
            dataset_id=self.dataset_id,
            split=self.split,
        )
        _validate_completed_indexes(completed, expected)
        if _complete(completed, expected):
            return self._commit_fragments(fragments_dir, expected)

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
            )
        return self._commit_fragments(fragments_dir, expected)

    def _write_single_part(
        self,
        provider: MaterializerProvider,
        *,
        device: str,
        dataset_factory: DatasetFactory,
    ) -> Path:
        output_dir = Path(self.output_dir).expanduser()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        env = set_single_worker_environment(
            device,
            device_env="ANYDATASET_MATERIALIZE_DEVICE",
        )
        with TemporaryDirectory(
            prefix=f".{output_dir.name}-parts-",
            dir=str(output_dir.parent),
        ) as tmpdir:
            try:
                parts_dir = Path(tmpdir)
                DatasetPartWriter(
                    parts_dir / "part-00000",
                    dataset_id=self.dataset_id,
                    split=self.split,
                    shard_id=0,
                    num_shards=1,
                    max_shard_samples=self.max_shard_samples,
                ).write(
                    self._loader_indexed_samples(
                        provider,
                        dataset_factory=dataset_factory,
                    )
                )
                return self._commit_parts(parts_dir)
            finally:
                restore_environment(env)

    def _write_part(
        self,
        dataset: Any,
        provider: MaterializerProvider,
        *,
        parts_dir: str | Path,
        num_shards: int,
        shard_id: int,
    ) -> Path:
        return DatasetPartWriter(
            Path(parts_dir) / f"part-{shard_id:05d}",
            dataset_id=self.dataset_id,
            split=self.split,
            shard_id=shard_id,
            num_shards=num_shards,
            max_shard_samples=self.max_shard_samples,
        ).write(
            self._indexed_samples(
                dataset,
                provider,
                num_shards=num_shards,
                shard_id=shard_id,
            )
        )

    def _commit_parts(self, parts_dir: str | Path) -> Path:
        return commit_store_parts(
            self.output_dir,
            parts_dir,
            dataset_id=self.dataset_id,
            split=self.split,
        )

    def _commit_fragments(self, fragments_dir: str | Path, expected: int) -> Path:
        if expected == 0:
            path = DatasetWriter(
                self.output_dir,
                dataset_id=self.dataset_id,
                split=self.split,
                max_shard_samples=self.max_shard_samples,
            ).write(())
            _cleanup_resume_dir(Path(self.output_dir).expanduser())
            return path
        path = commit_store_fragments(
            self.output_dir,
            fragments_dir,
            dataset_id=self.dataset_id,
            split=self.split,
            expected_sample_count=expected,
        )
        _cleanup_resume_dir(Path(self.output_dir).expanduser())
        return path

    def _samples(self, dataset: Iterable[Sample], provider: MaterializerProvider):
        if self.batch_size == 1:
            for sample in dataset:
                yield self._sample_with_provider(sample, provider)
            return

        for batch in sample_batches(dataset, self.batch_size):
            yield from self._resilient_samples_with_batch_provider(batch, provider)

    def _loader_indexed_samples(
        self,
        provider: MaterializerProvider,
        *,
        dataset_factory: DatasetFactory,
    ) -> Iterator[tuple[int, Sample]]:
        for batch in self._loader(dataset_factory=dataset_factory):
            if self.batch_size == 1:
                for index, sample in batch:
                    yield index, self._sample_with_provider(sample, provider)
                continue

            indexes, samples = zip(*batch, strict=True)
            outputs = tuple(
                self._resilient_samples_with_batch_provider(samples, provider)
            )
            validate_batch_outputs(outputs, len(samples))
            yield from zip(indexes, outputs, strict=True)

    def _write_resumable_loader_batches(
        self,
        provider: MaterializerProvider,
        *,
        dataset_factory: DatasetFactory,
        fragments_dir: Path,
        progress: multiprocessing.Queue | None = None,
        worker_id: int = 0,
    ) -> None:
        self._write_resumable_indexed_batches(
            self._loader(dataset_factory=dataset_factory),
            provider,
            fragments_dir=fragments_dir,
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
        )

    def _run_parallel_parts(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: tuple[str, ...],
        parts_dir: Path,
        logs_dir: Path,
        worker_logs_dir: Path,
        resume: bool = False,
        fragments_dir: Path | None = None,
    ) -> None:
        context = multiprocessing_context()
        progress = context.Queue()
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", free_port())
        workers = [
            context.Process(
                target=_materialize_worker,
                args=(
                    _WorkerConfig(
                        output_dir=Path(self.output_dir),
                        dataset_id=self.dataset_id,
                        split=self.split,
                        max_shard_samples=self.max_shard_samples,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        prefetch_factor=self.prefetch_factor,
                        mode=self._materializer_mode,
                        parts_dir=parts_dir,
                        resume=resume,
                        fragments_dir=fragments_dir,
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
        progress: multiprocessing.Queue | None = None,
        worker_id: int = 0,
    ) -> None:
        completed = set(
            completed_fragment_indexes(
                fragments_dir,
                dataset_id=self.dataset_id,
                split=self.split,
            )
        )
        for batch in batches:
            pending = tuple(
                (index, sample) for index, sample in batch if index not in completed
            )
            if not pending:
                continue
            outputs = self._materialized_indexed_batch(pending, provider)
            indexes = tuple(index for index, _ in outputs)
            fragment_id = _fragment_id(indexes)
            DatasetFragmentWriter(
                fragments_dir / fragment_id,
                dataset_id=self.dataset_id,
                split=self.split,
                fragment_id=fragment_id,
                max_shard_samples=self.max_shard_samples,
            ).write(outputs)
            completed.update(indexes)
            if progress is not None:
                put_progress(progress, Progress(worker_id, len(outputs), False, None))

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
class _WorkerConfig:
    output_dir: Path
    dataset_id: str
    split: str | None
    max_shard_samples: int
    batch_size: int
    num_workers: int
    prefetch_factor: int | None
    mode: _MaterializerMode
    parts_dir: Path
    resume: bool
    fragments_dir: Path | None
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


def _prepare_parallel_output_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Target path exists and is not a directory: {output_dir}")
        entries = list(output_dir.iterdir())
        if entries:
            raise ValueError(f"Target directory must be empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    return output_dir


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
            set_torch_device(config.device)
            process_group_created = _init_worker_process_group(config)
            provider = provider_factory(config.device)
            materializer = _worker_materializer(config)
            if config.resume:
                if config.fragments_dir is None:
                    raise ValueError("fragments_dir is required for resume.")
                materializer._write_resumable_loader_batches(
                    provider,
                    dataset_factory=dataset_factory,
                    fragments_dir=config.fragments_dir,
                    progress=progress,
                    worker_id=config.shard_id,
                )
            else:
                DatasetPartWriter(
                    config.parts_dir / f"part-{config.shard_id:05d}",
                    dataset_id=config.dataset_id,
                    split=config.split,
                    shard_id=config.shard_id,
                    num_shards=config.num_shards,
                    max_shard_samples=config.max_shard_samples,
                ).write(
                    iter_with_progress(
                        materializer._loader_indexed_samples(
                            provider,
                            dataset_factory=dataset_factory,
                        ),
                        worker_id=config.shard_id,
                        progress=progress,
                    )
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


def _dataset_id(output_dir: str | Path) -> str:
    return Path(output_dir).expanduser().name or "dataset"


def _resume_root(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.resume"


def _resume_fragments_dir(output_dir: Path) -> Path:
    return _resume_root(output_dir) / "fragments"


def _prepare_resume_fragments_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Target path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(f"Target directory must be empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    fragments_dir = _resume_fragments_dir(output_dir)
    fragments_dir.mkdir(parents=True, exist_ok=True)
    return fragments_dir


def _cleanup_resume_dir(output_dir: Path) -> None:
    root = _resume_root(output_dir)
    if root.exists():
        shutil.rmtree(root)


def _dataset_sample_count(dataset: Any) -> int:
    try:
        count = len(dataset)
    except TypeError as exc:
        raise TypeError("resume requires a dataset with __len__().") from exc
    if not isinstance(count, int):
        raise TypeError("dataset __len__() must return an integer.")
    if count < 0:
        raise ValueError("dataset length must be non-negative.")
    return count


def _validate_completed_indexes(indexes: frozenset[int], expected: int) -> None:
    extras = sorted(index for index in indexes if index < 0 or index >= expected)
    if extras:
        raise ValueError(f"Completed fragment index is outside dataset: {extras[0]}.")


def _complete(indexes: frozenset[int], expected: int) -> bool:
    return len(indexes) == expected and indexes == frozenset(range(expected))


def _fragment_id(indexes: Sequence[int]) -> str:
    if not indexes:
        raise ValueError("fragment indexes must not be empty.")
    text = ",".join(str(index) for index in indexes)
    digest = hashlib.sha256(text.encode("ascii")).hexdigest()[:16]
    return f"batch-{indexes[0]:012d}-{indexes[-1]:012d}-{digest}"
