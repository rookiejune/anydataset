from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from torch.utils.data import DataLoader

from .._compat import strict_zip
from .._devices import Devices, resolve_devices
from .._logging import run_logs_dir
from .._parallel import (
    can_select_indexes,
    free_port,
    indexed_loader,
    iter_indexed_shard,
    map_style_indexed_loader,
    multiprocessing_context,
    restore_environment,
    set_single_worker_environment,
    set_torch_device,
    validate_process_parent,
    validate_process_value,
)
from .._progress import ProgressDashboard, watch_workers
from .._resume import (
    cleanup_resume_dir,
    dataset_sample_count,
    indexes_complete,
    log_resume_summary,
    missing_indexes,
    validate_completed_indexes,
)
from .._validation import non_negative_int, optional_positive_int, positive_int
from ..cache import FileLock
from ..runtime import Runtime
from ..types._sample import merge as merge_samples
from ..types._sample import select as select_sample
from ..types.item import Sample, Schema
from ..view import Provider
from ._batch import (
    indexed_sample_batches,
    validate_batch_outputs,
    with_batch_modality_provider,
    with_batch_view_provider,
    with_resilient_batch_provider,
)
from ._config import DEFAULT_MAX_SHARD_SAMPLES
from ._materializer_fragments import FragmentBatchWriter, ProgressSink
from ._materializer_identity import callable_id, metadata_value, optional_semantic_id
from ._materializer_resume import (
    materializer_lock_path,
    prepare_materializer_resume_dir,
)
from ._materializer_worker import WorkerConfig, materialize_worker
from ._modality import with_modality_provider
from ._types import MaterializerProvider, ModalityProviderLike
from ._view import with_view_provider
from ._part_commit import (
    commit_store_fragments,
    commit_store_parts,
    completed_fragment_indexes,
)
from .writer import DatasetWriter

DatasetFactory = Callable[[], Any]
ProviderFactory = Callable[[str], MaterializerProvider]
_MaterializerMode = Literal["view", "modality"]

_PROGRESS_STAGES = ("reader", "provider", "writer")
DEFAULT_COMMIT_SAMPLES = 32


@dataclass
class ViewMaterializer:
    output_dir: str | Path
    split: str | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES
    batch_size: int = 1
    commit_samples: int | None = None
    num_workers: int = 0
    prefetch_factor: int | None = None
    write_workers: int = 1
    write_prefetch: int | None = None
    runtime: Runtime = field(default_factory=Runtime)
    keep_schema: Schema | None = None
    input_id: str | None = None
    provider_id: str | None = None

    def __post_init__(self) -> None:
        self.max_shard_samples = positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )
        self.batch_size = positive_int("batch_size", self.batch_size)
        if self.commit_samples is None:
            self.commit_samples = max(self.batch_size, DEFAULT_COMMIT_SAMPLES)
        else:
            self.commit_samples = positive_int("commit_samples", self.commit_samples)
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
        self.input_id = optional_semantic_id("input_id", self.input_id)
        self.provider_id = optional_semantic_id("provider_id", self.provider_id)

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
        if len(resolved) > 1 or self.num_workers > 0:
            validate_process_parent(
                context=(
                    f"{type(self).__name__} with multiple devices or DataLoader "
                    "workers"
                )
            )
        with FileLock(materializer_lock_path(self.output_dir)):
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
        dataset = dataset_factory()
        expected = dataset_sample_count(dataset, context="resume")
        use_map_style_loader = can_select_indexes(dataset)
        fragments_dir = prepare_materializer_resume_dir(
            self.output_dir,
            self._resume_metadata(
                dataset,
                dataset_factory=dataset_factory,
                provider_factory=provider_factory,
                expected=expected,
                use_map_style_loader=use_map_style_loader,
            ),
        )
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

        missing = missing_indexes(completed, expected)
        log_resume_summary(
            "materializer",
            expected=expected,
            completed_count=len(completed),
            missing=missing,
            use_map_style_loader=use_map_style_loader,
        )
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
            use_map_style_loader=use_map_style_loader,
            completed_count=len(completed),
            missing_indexes=missing,
        )
        return self._commit_parts(fragments_dir / ".parts")

    def _write_resumable_single(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        device: str,
    ) -> Path:
        output_dir = Path(self.output_dir).expanduser()
        dataset = dataset_factory()
        expected = dataset_sample_count(dataset, context="resume")
        use_map_style_loader = can_select_indexes(dataset)
        fragments_dir = prepare_materializer_resume_dir(
            output_dir,
            self._resume_metadata(
                dataset,
                dataset_factory=dataset_factory,
                provider_factory=provider_factory,
                expected=expected,
                use_map_style_loader=use_map_style_loader,
            ),
        )
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

        missing = missing_indexes(completed, expected)
        log_resume_summary(
            "materializer",
            expected=expected,
            completed_count=len(completed),
            missing=missing,
            use_map_style_loader=use_map_style_loader,
        )
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
                        dataset=dataset,
                        sample_count=expected,
                        use_map_style_loader=use_map_style_loader,
                        sample_indexes=missing,
                        fragments_dir=fragments_dir,
                        expected=expected,
                        progress=progress,
                    )
                finally:
                    restore_environment(env)
            else:
                self._write_resumable_indexed_batches(
                    indexed_sample_batches(
                        _missing_indexed_samples(
                            dataset,
                            missing,
                            use_map_style_loader=use_map_style_loader,
                        ),
                        self.batch_size,
                    ),
                    provider,
                    fragments_dir=fragments_dir,
                    expected=expected,
                    progress=progress,
                )
        return self._commit_fragments(fragments_dir, expected)

    def _commit_fragments(
        self,
        fragments_dir: str | Path,
        expected: int,
    ) -> Path:
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

    def _commit_parts(self, parts_dir: str | Path) -> Path:
        path = commit_store_parts(
            self.output_dir,
            parts_dir,
            dataset_id=self._dataset_id,
            split=self.split,
        )
        cleanup_resume_dir(self.output_dir)
        return path

    def _write_resumable_loader_batches(
        self,
        provider: MaterializerProvider,
        *,
        dataset_factory: DatasetFactory,
        dataset: Any | None = None,
        sample_count: int | None = None,
        use_map_style_loader: bool | None = None,
        sample_indexes: Sequence[int] | None = None,
        fragments_dir: Path,
        expected: int,
        progress: ProgressSink | None = None,
        worker_id: int = 0,
    ) -> None:
        self._write_resumable_indexed_batches(
            self._loader(
                dataset_factory=dataset_factory,
                dataset=dataset,
                sample_count=sample_count,
                use_map_style_loader=use_map_style_loader,
                sample_indexes=sample_indexes,
            ),
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
        dataset: Any | None = None,
        sample_count: int | None = None,
        use_map_style_loader: bool | None = None,
        sample_indexes: Sequence[int] | None = None,
    ) -> DataLoader:
        if dataset is None:
            if use_map_style_loader is None or sample_count is None:
                dataset = dataset_factory()
        if use_map_style_loader is None:
            use_map_style_loader = can_select_indexes(dataset)
        if use_map_style_loader:
            if sample_count is None:
                sample_count = len(dataset)
            return map_style_indexed_loader(
                dataset_factory,
                sample_count=sample_count,
                sample_indexes=sample_indexes,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                start_method=self.runtime.reader_worker_start_method,
                dataset=dataset,
            )
        return indexed_loader(
            dataset_factory,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            start_method=self.runtime.reader_worker_start_method,
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
        use_map_style_loader: bool,
        completed_count: int,
        missing_indexes: Sequence[int],
    ) -> None:
        context = multiprocessing_context(self.runtime.process_start_method)
        progress = context.Queue()
        barrier = context.Barrier(len(devices))
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", free_port())
        workers = [
            context.Process(
                target=materialize_worker,
                args=(
                    WorkerConfig(
                        output_dir=Path(self.output_dir),
                        split=self.split,
                        max_shard_samples=self.max_shard_samples,
                        batch_size=self.batch_size,
                        commit_samples=self.commit_samples,
                        num_workers=self.num_workers,
                        prefetch_factor=self.prefetch_factor,
                        write_workers=self.write_workers,
                        write_prefetch=self.write_prefetch,
                        keep_schema=self.keep_schema,
                        mode=self._materializer_mode,
                        runtime=self.runtime,
                        use_map_style_loader=use_map_style_loader,
                        missing_indexes=missing_indexes,
                        fragments_dir=fragments_dir,
                        parts_dir=fragments_dir / ".parts",
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
                    barrier,
                ),
                name=f"anydataset-materialize-{shard_id}",
                daemon=False,
            )
            for shard_id, device in enumerate(devices)
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
                desc="materialize views",
                early_exit_message="View materialization worker exited early.",
                failure_prefix="View materialization worker",
                total=expected,
                count_stage="writer",
                initial=completed_count,
                stages=_PROGRESS_STAGES,
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
            raise RuntimeError(f"View materialization workers failed: {details}.")

    @property
    def _materializer_mode(self) -> _MaterializerMode:
        return "view"

    def _resume_metadata(
        self,
        dataset: Any,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        expected: int,
        use_map_style_loader: bool,
    ) -> dict[str, object]:
        return {
            "schema_version": 3,
            "materializer": {
                "mode": self._materializer_mode,
                "dataset_id": self._dataset_id,
                "split": self.split,
                "max_shard_samples": self.max_shard_samples,
                "batch_size": self.batch_size,
                "keep_schema": metadata_value(self.keep_schema),
            },
            "input": {
                "type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
                "factory": callable_id(dataset_factory),
                "semantic_id": self.input_id,
                "sample_count": expected,
                "use_map_style_loader": use_map_style_loader,
            },
            "provider": {
                "factory": callable_id(provider_factory),
                "semantic_id": self.provider_id,
            },
        }

    def _sample_with_provider(
        self,
        sample: Sample,
        provider: MaterializerProvider,
    ) -> Sample:
        return self._output_sample(
            sample,
            with_view_provider(sample, cast(Provider, provider)),
        )

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
            indexes, samples = strict_zip(*batch)
            outputs = tuple(
                self._resilient_samples_with_batch_provider(samples, provider)
            )
            validate_batch_outputs(outputs, len(samples))
            yield from strict_zip(indexes, outputs)

    def _write_resumable_indexed_batches(
        self,
        batches: Iterable[Sequence[tuple[int, Sample]]],
        provider: MaterializerProvider,
        *,
        fragments_dir: Path,
        expected: int,
        progress: ProgressSink | None = None,
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
        writer = FragmentBatchWriter(
            materializer=self,
            fragments_dir=fragments_dir,
            completed=completed,
            provider=provider,
            progress=progress,
            worker_id=worker_id,
        )
        writer.write(batches)

    def _samples_with_batch_provider(
        self,
        samples: Sequence[Sample],
        provider: MaterializerProvider,
    ) -> Iterator[Sample]:
        return self._output_samples(
            samples,
            with_batch_view_provider(samples, cast(Provider, provider)),
        )

    def _output_sample(self, source: Sample, output: Sample) -> Sample:
        if self.keep_schema is None:
            return output
        kept = _select_sample(source, self.keep_schema)
        return _merge_output_samples(kept, output)

    def _output_samples(
        self,
        sources: Sequence[Sample],
        outputs: Iterator[Sample],
    ) -> Iterator[Sample]:
        if self.keep_schema is None:
            yield from outputs
            return
        for source, output in strict_zip(sources, outputs):
            yield self._output_sample(source, output)

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
        return self._output_sample(
            sample,
            with_modality_provider(sample, cast(ModalityProviderLike, provider)),
        )

    def _samples_with_batch_provider(
        self,
        samples: Sequence[Sample],
        provider: MaterializerProvider,
    ) -> Iterator[Sample]:
        return self._output_samples(
            samples,
            with_batch_modality_provider(
                samples,
                cast(ModalityProviderLike, provider),
            ),
        )

def _missing_indexed_samples(
    dataset: Any,
    indexes: Sequence[int],
    *,
    use_map_style_loader: bool,
) -> Iterator[tuple[int, Sample]]:
    if use_map_style_loader:
        for index in indexes:
            yield index, dataset[index]
        return
    yield from iter_indexed_shard(dataset, 1, 0)


def _select_sample(sample: Sample, schema: Schema) -> Sample:
    return select_sample(sample, schema)


def _merge_output_samples(left: Sample, right: Sample) -> Sample:
    return merge_samples(left, right, context="Materialized sample")


def _dataset_id(output_dir: str | Path) -> str:
    return Path(output_dir).expanduser().name or "dataset"
