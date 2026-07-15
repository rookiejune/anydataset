from __future__ import annotations

import logging
import multiprocessing
import os
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import Any, Literal, Union, cast

from torch.utils.data import DataLoader

from .._compat import strict_zip
from .._devices import Devices, resolve_devices
from .._logging import run_logs_dir, use_run_logs_dir, write_warning
from .._parallel import (
    DeviceWorker,
    free_port,
    indexed_loader,
    iter_indexed_shard,
    map_style_indexed_loader,
    multiprocessing_context,
    restore_environment,
    set_single_worker_environment,
    set_torch_device,
    set_worker_environment,
    validate_process_value,
)
from .._progress import Progress, ProgressDashboard, put_progress, watch_workers
from .._resume import (
    append_completed_index_cache,
    cleanup_resume_dir,
    dataset_sample_count,
    index_batch_id,
    indexes_complete,
    log_resume_summary,
    missing_indexes,
    pending_batch,
    prepare_resume_dir,
    quarantine_resume_dir,
    resume_dir,
    validate_completed_indexes,
)
from .._validation import non_negative_int, optional_positive_int, positive_int
from .._write_pipeline import BackgroundWriteSink
from ..cache import FileLock
from ..dataset.abc import uses_default_indexed_shard
from ..runtime import Runtime
from ..types.item import Item, Sample, Schema, View
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
    commit_fragment_part,
    commit_store_fragments,
    commit_store_parts,
    completed_fragment_indexes,
    store_fragments,
)
from .jsonio import read_json, write_json
from .writer import DEFAULT_MAX_SHARD_SAMPLES, DatasetWriter

DatasetFactory = Callable[[], Any]
ProviderFactory = Callable[[str], MaterializerProvider]
_MaterializerMode = Literal["view", "modality"]
_ProgressSink = Union[multiprocessing.Queue, ProgressDashboard]

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
        with FileLock(_materializer_lock_path(self.output_dir)):
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
        use_map_style_loader = uses_default_indexed_shard(dataset)
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
        use_map_style_loader = uses_default_indexed_shard(dataset)
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
        progress: _ProgressSink | None = None,
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
            use_map_style_loader = uses_default_indexed_shard(dataset)
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
                target=_materialize_worker,
                args=(
                    _WorkerConfig(
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
                        missing_indexes=tuple(missing_indexes),
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
            "schema_version": 1,
            "materializer": {
                "mode": self._materializer_mode,
                "dataset_id": self._dataset_id,
                "split": self.split,
                "max_shard_samples": self.max_shard_samples,
                "batch_size": self.batch_size,
                "keep_schema": _metadata_value(self.keep_schema),
            },
            "input": {
                "type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
                "factory": _callable_id(dataset_factory),
                "sample_count": expected,
                "use_map_style_loader": use_map_style_loader,
            },
            "provider": {
                "factory": _callable_id(provider_factory),
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
        writer = _FragmentBatchWriter(
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


@dataclass
class _FragmentBatchWriter:
    materializer: ViewMaterializer
    fragments_dir: Path
    completed: set[int]
    provider: MaterializerProvider
    progress: _ProgressSink | None = None
    worker_id: int = 0

    def write(self, batches: Iterable[Sequence[tuple[int, Sample]]]) -> None:
        with self._sink() as sink:
            pending_outputs: list[tuple[int, Sample]] = []
            read_start = time.perf_counter()
            for batch in batches:
                self._record_read(batch, read_start)
                pending = pending_batch(batch, self.completed)
                if not pending:
                    read_start = time.perf_counter()
                    continue
                outputs = self._materialized_batch(pending)
                pending_outputs.extend(outputs)
                self._flush_ready(sink, pending_outputs)
                read_start = time.perf_counter()
            self._flush_remaining(sink, pending_outputs)

    def _record_read(
        self,
        batch: Sequence[tuple[int, Sample]],
        read_start: float,
    ) -> None:
        _put_stage_progress(
            self.progress,
            worker_id=self.worker_id,
            stage="reader",
            samples=len(batch),
            elapsed=time.perf_counter() - read_start,
        )

    def _materialized_batch(
        self,
        batch: Sequence[tuple[int, Sample]],
    ) -> tuple[tuple[int, Sample], ...]:
        provider_start = time.perf_counter()
        outputs = self._materialized_indexed_batch(batch)
        _put_stage_progress(
            self.progress,
            worker_id=self.worker_id,
            stage="provider",
            samples=len(outputs),
            elapsed=time.perf_counter() - provider_start,
        )
        return outputs

    def _materialized_indexed_batch(
        self,
        batch: Sequence[tuple[int, Sample]],
    ) -> tuple[tuple[int, Sample], ...]:
        if self.materializer.batch_size == 1:
            return tuple(
                (
                    index,
                    self.materializer._sample_with_provider(sample, self.provider),
                )
                for index, sample in batch
            )

        indexes, samples = strict_zip(*batch)
        outputs = tuple(
            self.materializer._resilient_samples_with_batch_provider(
                samples,
                self.provider,
            )
        )
        validate_batch_outputs(outputs, len(samples))
        return tuple(strict_zip(indexes, outputs))

    def _flush_ready(
        self,
        sink: BackgroundWriteSink[_FragmentWriteJob],
        pending_outputs: list[tuple[int, Sample]],
    ) -> None:
        while len(pending_outputs) >= self.materializer.commit_samples:
            self._submit(sink, pending_outputs[: self.materializer.commit_samples])
            del pending_outputs[: self.materializer.commit_samples]

    def _flush_remaining(
        self,
        sink: BackgroundWriteSink[_FragmentWriteJob],
        pending_outputs: list[tuple[int, Sample]],
    ) -> None:
        if pending_outputs:
            self._submit(sink, pending_outputs)

    def _submit(
        self,
        sink: BackgroundWriteSink[_FragmentWriteJob],
        samples: Sequence[tuple[int, Sample]],
    ) -> None:
        indexed = tuple(samples)
        indexes = tuple(sorted(index for index, _ in indexed))
        sink.submit(
            _FragmentWriteJob(
                fragments_dir=self.fragments_dir,
                dataset_id=self.materializer._dataset_id,
                split=self.materializer.split,
                max_shard_samples=self.materializer.max_shard_samples,
                indexes=indexes,
                samples=indexed,
            )
        )
        self.completed.update(indexes)

    def _sink(self) -> BackgroundWriteSink[_FragmentWriteJob]:
        return BackgroundWriteSink(
            _write_fragment,
            workers=self.materializer.write_workers,
            max_pending=self.materializer.write_prefetch,
            start_method=self.materializer.runtime.writer_worker_start_method,
            on_submit=lambda job, pending: _put_stage_progress(
                self.progress,
                worker_id=self.worker_id,
                stage="writer",
                pending=pending,
            ),
            on_complete=lambda job, pending, elapsed: _put_stage_progress(
                self.progress,
                worker_id=self.worker_id,
                stage="writer",
                samples=len(job.samples),
                elapsed=elapsed,
                pending=pending,
            ),
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
    append_completed_index_cache(job.fragments_dir, fragment_id, job.indexes)


def prepare_materializer_resume_dir(
    output_dir: str | Path,
    metadata: Mapping[str, object],
) -> Path:
    path = resume_dir(output_dir, "fragments")
    expected = dict(metadata)
    if path.exists() and _stored_resume_metadata(path) != expected:
        stale = quarantine_resume_dir(output_dir)
        write_warning(
            "materializer",
            "Quarantined incompatible resume directory "
            f"at {stale}; remove it after confirming it is no longer needed.",
        )
    path = prepare_resume_dir(output_dir, "fragments")
    write_json(path / "resume.json", expected)
    return path


def _stored_resume_metadata(path: Path) -> Mapping[str, object] | None:
    metadata_path = path / "resume.json"
    if not metadata_path.is_file():
        return None
    data = read_json(metadata_path)
    if not isinstance(data, Mapping):
        raise ValueError("Materializer resume metadata must be a mapping.")
    return data


def _metadata_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_metadata_value(item) for item in value]
    return repr(value)


def _callable_id(value: object) -> object:
    if isinstance(value, partial):
        return {
            "type": "functools.partial",
            "function": _callable_id(value.func),
            "args": _metadata_value(value.args),
            "keywords": _metadata_value(value.keywords or {}),
        }
    if isinstance(value, MethodType):
        return {
            "function": _callable_id(value.__func__),
            "owner": _callable_id(value.__self__),
        }
    if isinstance(value, (FunctionType, BuiltinFunctionType)):
        return f"{value.__module__}.{value.__qualname__}"

    type_id = f"{type(value).__module__}.{type(value).__qualname__}"
    if type(value).__repr__ is object.__repr__:
        return type_id
    return {"type": type_id, "value": repr(value)}


def _materializer_lock_path(output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser()
    return output.parent / f".{output.name}.materialize.lock"


@dataclass(frozen=True)
class _WorkerConfig:
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
    mode: _MaterializerMode
    runtime: Runtime
    use_map_style_loader: bool
    missing_indexes: tuple[int, ...]
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


def _materialize_worker(
    config: _WorkerConfig,
    dataset_factory: DatasetFactory,
    provider_factory: ProviderFactory,
    progress: multiprocessing.Queue,
    barrier: Any,
) -> None:
    with use_run_logs_dir(config.logs_dir):
        logger = _worker_logger(config.worker_logs_dir, config.shard_id)
        logger.info(
            "starting shard %s/%s on %s missing=%s map_style=%s",
            config.shard_id,
            config.num_shards,
            config.device,
            _shard_missing_count(config.missing_indexes, config.num_shards, config.shard_id),
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
            materializer = _worker_materializer(config)
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


def _worker_materializer(config: _WorkerConfig) -> ViewMaterializer:
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


def _worker_logger(logs_dir: Path, shard_id: int) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"anydataset.materializer.{os.getpid()}.{shard_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    path = logs_dir / f"part-{shard_id:05d}.log"
    handler = logging.FileHandler(
        path,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(processName)s %(message)s")
    )
    logger.handlers.clear()
    logger.addHandler(handler)
    if shard_id == 0:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(console)
    logger.info("worker log: %s", path)
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


def _shard_missing_count(indexes: Sequence[int], num_shards: int, shard_id: int) -> int:
    if shard_id >= len(indexes):
        return 0
    return (len(indexes) - 1 - shard_id) // num_shards + 1


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
    return {
        reference: sample[reference].select_by(requirement)
        for reference, requirement in schema.items()
    }


def _merge_output_samples(left: Sample, right: Sample) -> Sample:
    result = dict(left)
    for ref, item in right.items():
        current = result.get(ref)
        if current is None:
            result[ref] = item
            continue
        result[ref] = _merge_output_items(current, item, ref=ref)
    return result


def _merge_output_items(left: Item, right: Item, *, ref: object) -> Item:
    if type(left) is not type(right):
        raise TypeError(f"Materialized sample item {ref!r} has incompatible types.")
    view_conflicts = set(left.views) & set(right.views)
    if view_conflicts:
        view = _first_sorted_view(view_conflicts)
        raise ValueError(
            f"Materialized sample item {ref!r} view conflict for {view!r}."
        )

    meta = dict(left.meta)
    for key, value in right.meta.items():
        current = meta.get(key)
        if key in meta and not _values_equal(current, value):
            raise ValueError(
                f"Materialized sample item {ref!r} metadata conflict for {key!r}."
            )
        meta[key] = value

    return type(left)(
        views={**left.views, **right.views},
        meta=meta,
    )


def _first_sorted_view(views: set[View]) -> View:
    return sorted(views, key=lambda view: view.value)[0]


def _values_equal(left: Any, right: Any) -> bool:
    equal = left == right
    if isinstance(equal, bool):
        return equal
    try:
        return bool(equal)
    except (TypeError, ValueError, RuntimeError):
        return left is right


def _dataset_id(output_dir: str | Path) -> str:
    return Path(output_dir).expanduser().name or "dataset"
