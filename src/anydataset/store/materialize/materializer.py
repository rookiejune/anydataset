from __future__ import annotations

import os
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from types import TracebackType
from typing import Any, Literal, Optional, cast

from torch.utils.data import DataLoader

from ..._compat import strict_zip
from ..._runtime.devices import Devices, resolve_devices
from ..._runtime.logging import run_logs_dir, write_info, write_warning
from ..._runtime.parallel import (
    ProcessHandle,
    can_select_indexes,
    free_port,
    iter_shard,
    multiprocessing_context,
    restore_environment,
    sample_index_loader,
    set_single_worker_environment,
    set_torch_device,
    validate_process_parent,
    validate_process_value,
)
from ..._runtime.progress import (
    Progress,
    ProgressDashboard,
    watch_workers,
    write_progress_message,
)
from ..._runtime.resume import (
    dataset_sample_count,
    indexes_complete,
    log_resume_summary,
    missing_indexes,
)
from ..._validation import (
    non_negative_int,
    optional_positive_int,
    positive_int,
)
from ...cache import FileLock
from ...dataset.abc import MapStyleABC
from ...dataset.coverage import CoverageCoordinator, CoverageLease
from ...dataset.universe import DatasetUniverse
from ...dataset.view import SelectionView
from ...runtime import Runtime
from ...types._sample import combine as combine_samples
from ...types._sample import select as select_sample
from ...types.item import Modality, Role, Sample, Schema, View
from ...view import BatchSampleProvider, Provider, SampleProvider
from .batch import (
    sample_index_batches,
    validate_batch_outputs,
    with_batch_modality_provider,
    with_batch_view_provider,
    with_resilient_batch_provider,
)
from ..config import DEFAULT_MAX_SHARD_BYTES, DEFAULT_MAX_SHARD_SAMPLES
from .._identity import materialized_universe_id
from .fragments import (
    FragmentBatchConfig,
    FragmentBatchWriter,
    FragmentOutputSink,
    ProgressSink,
)
from .identity import (
    callable_id,
    metadata_digest,
    metadata_value,
    optional_semantic_id,
)
from .resume import (
    cleanup_materializer_resume_dir,
    materializer_fragments_dir,
    materializer_lock_path,
    prepare_materializer_resume_dir,
    stored_resume_metadata,
)
from .worker import WorkerConfig, materialize_worker
from .modality import with_modality_provider
from .types import (
    BatchModalityProviderLike,
    BatchViewProviderLike,
    MaterializerProvider,
    ModalityProviderLike,
    output_modality,
)
from .view import with_view_provider
from ..part.commit import (
    commit_store_fragments,
    commit_store_parts,
    compact_completed_fragment_indexes,
)
from ..part.sample_write import inherited_sample_id, sample_id, sample_id_prefix
from ..writer import DatasetWriter

DatasetFactory = Callable[[], Any]
ProviderFactory = Callable[[str], MaterializerProvider]
InputIdFactory = Callable[[Any], str]
_MaterializerMode = Literal["view", "modality", "sample"]

_PROGRESS_STAGES = ("reader", "provider", "writer")
_COMMIT_PROGRESS_STAGES = (
    "scan",
    "validate",
    "merge-runs",
    "merge-samples",
    "merge-views",
    "link-shards",
)
DEFAULT_COMMIT_SAMPLES = 1024


@dataclass(frozen=True)
class _UniverseDatasetFactory:
    """Recreate only the complete payload universe for worker execution."""

    factory: DatasetFactory

    def __call__(self) -> Any:
        dataset, _ = _materialization_input(self.factory())
        return dataset


@dataclass(frozen=True)
class MaterializationStatus:
    """Canonical or staging coverage for one materialization."""

    output_dir: Path
    expected: int
    completed: int
    finalized: bool = False

    @property
    def pending(self) -> int:
        return self.expected - self.completed


@dataclass
class ViewMaterializer:
    output_dir: str | Path
    split: str | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES
    max_shard_bytes: int | None = DEFAULT_MAX_SHARD_BYTES
    batch_size: int = 1
    commit_samples: int | None = None
    num_workers: int = 0
    prefetch_factor: int | None = None
    write_workers: int = 1
    write_prefetch: int | None = None
    runtime: Runtime = field(default_factory=Runtime)
    keep_schema: Schema | None = None
    input_id: str | None = None
    input_id_factory: InputIdFactory | None = field(default=None, repr=False)
    provider_id: str | None = None
    output: View | None = None
    schema: Schema | None = None
    staging_dir: str | Path | None = None
    dataset_id: str | None = None

    def __post_init__(self) -> None:
        self.max_shard_samples = positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )
        self.max_shard_bytes = optional_positive_int(
            "max_shard_bytes",
            self.max_shard_bytes,
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
        if self.input_id_factory is not None and not callable(self.input_id_factory):
            raise TypeError("input_id_factory must be callable or None.")
        if self.input_id is not None and self.input_id_factory is not None:
            raise ValueError("input_id and input_id_factory are mutually exclusive.")
        self.provider_id = optional_semantic_id("provider_id", self.provider_id)
        self.dataset_id = optional_semantic_id("dataset_id", self.dataset_id)
        self.staging_dir = _staging_dir(self.output_dir, self.staging_dir)
        self.output, self.schema = _output_contract(self.output, self.schema)

    @property
    def _dataset_id(self) -> str:
        if self.dataset_id is not None:
            return self.dataset_id
        return _dataset_id(self.output_dir)

    @property
    def _provenance(self) -> dict[str, str]:
        return {
            key: value
            for key, value in (
                ("input_id", self.input_id),
                ("provider_id", self.provider_id),
                ("output_id", self._output_id),
            )
            if value is not None
        }

    @property
    def _output_id(self) -> str | None:
        if self.output is None or self.schema is None:
            return None
        digest = metadata_digest(
            {
                "schema_version": 1,
                "output": self.output,
                "schema": self.schema,
            },
            set(),
        )
        return f"view-v1:{digest}"

    def _resolve_input_id(self, dataset: Any) -> None:
        if self.input_id is not None or self.input_id_factory is None:
            return
        self.input_id = optional_semantic_id(
            "input_id_factory output",
            self.input_id_factory(dataset),
        )
        if self.input_id is None:
            raise ValueError("input_id_factory must return a non-empty string.")

    def open(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        selection_factory: DatasetFactory | None = None,
        device: str = "cpu",
    ) -> MapStyleABC:
        """Open a complete canonical store or an online materializing view.

        The choice is dataset-wide. A compatible ready store with explicit
        ``input_id`` is opened without constructing the source dataset or
        provider. ``input_id_factory`` constructs only the input identity object
        needed to validate ready provenance. Otherwise every requested sample is
        generated from the source and only its persistence is queued in the
        background.
        """

        self._validate_online_contract()
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string.")
        if selection_factory is not None and not callable(selection_factory):
            raise TypeError("selection_factory must be callable or None.")
        ready = self._ready_dataset()
        if ready is not None:
            return self._ready_result(
                ready,
                dataset_factory=dataset_factory,
                selection_factory=selection_factory,
            )

        lock = FileLock(materializer_lock_path(self.output_dir))
        lock.__enter__()
        lock_owned = True
        dataset: Any | None = None
        source_view: SelectionView | None = None
        provider: MaterializerProvider | None = None
        sink: FragmentOutputSink | None = None
        try:
            ready = self._ready_dataset()
            if ready is not None:
                try:
                    lock.__exit__(None, None, None)
                except BaseException:
                    _close_resources_quietly(ready)
                    raise
                lock_owned = False
                return self._ready_result(
                    ready,
                    dataset_factory=dataset_factory,
                    selection_factory=selection_factory,
                )

            source_factory = (
                dataset_factory if selection_factory is None else selection_factory
            )
            dataset, source_view = _materialization_input(source_factory())
            _validate_publishable_input(dataset)
            if not can_select_indexes(dataset):
                raise TypeError(
                    "Online materializing views require a map-style source dataset."
                )
            self._resolve_input_id(_materialization_identity_input(dataset))
            expected = dataset_sample_count(dataset, context="online materialization")
            fragments_dir = prepare_materializer_resume_dir(
                self.output_dir,
                self._resume_metadata(
                    dataset,
                    dataset_factory=dataset_factory,
                    provider_factory=provider_factory,
                    expected=expected,
                    use_map_style_loader=True,
                    selection_agnostic=source_view is not None,
                ),
                staging_dir=self.staging_dir,
            )
            completed = compact_completed_fragment_indexes(
                fragments_dir,
                dataset_id=self._dataset_id,
                split=self.split,
                expected=expected,
            )
            provider = provider_factory(device)
            self._validate_provider_output(provider)
            sink = FragmentOutputSink(
                config=self._fragment_config(),
                fragments_dir=fragments_dir,
                completed=completed,
                expected=expected,
                sample_identity=dataset,
            )
            sink.__enter__()
            online = MaterializingViewDataset(
                _source=dataset,
                _provider=cast(Provider, provider),
                _materializer=self,
                _sink=sink,
                _lock=lock,
                _owner_pid=os.getpid(),
                _owns_source=source_view is None,
            )
            return _materialization_result(online, source_view)
        except BaseException:
            source = source_view if source_view is not None else dataset
            _abort_online_open(
                sink,
                provider,
                source,
                lock if lock_owned else None,
            )
            raise

    def status(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
    ) -> MaterializationStatus:
        """Inspect canonical or staging coverage without constructing a provider."""

        self._validate_online_contract()
        ready = self._ready_dataset()
        if ready is not None:
            try:
                if self.input_id is None:
                    self._resolve_ready_input_id(dataset_factory=dataset_factory)
                    self._validate_ready_input_id(ready)
                expected = len(ready)
            except BaseException:
                _close_resources_quietly(ready)
                raise
            _close_resource(ready)
            return MaterializationStatus(
                output_dir=Path(self.output_dir).expanduser(),
                expected=expected,
                completed=expected,
                finalized=True,
            )

        dataset, source_view = _materialization_input(dataset_factory())
        source = source_view if source_view is not None else dataset
        try:
            _validate_publishable_input(dataset)
            if not can_select_indexes(dataset):
                raise TypeError(
                    "Online materializing views require a map-style source dataset."
                )
            self._resolve_input_id(_materialization_identity_input(dataset))
            expected = dataset_sample_count(dataset, context="materialization status")
            metadata = self._resume_metadata(
                dataset,
                dataset_factory=dataset_factory,
                provider_factory=provider_factory,
                expected=expected,
                use_map_style_loader=True,
                selection_agnostic=source_view is not None,
            )
        except BaseException:
            _close_resources_quietly(source)
            raise
        _close_resource(source)

        fragments_dir = materializer_fragments_dir(
            self.output_dir,
            staging_dir=self.staging_dir,
        )
        if not fragments_dir.exists() or (
            fragments_dir.is_dir() and not any(fragments_dir.iterdir())
        ):
            completed: Collection[int] = ()
        else:
            if not fragments_dir.is_dir():
                raise ValueError(
                    f"Materializer staging path is not a directory: {fragments_dir}"
                )
            if stored_resume_metadata(fragments_dir) != metadata:
                raise ValueError(
                    "Materializer identity does not match the staging state."
                )
            completed = compact_completed_fragment_indexes(
                fragments_dir,
                dataset_id=self._dataset_id,
                split=self.split,
                expected=expected,
            )
        return self._status(expected, completed)

    def _ready_result(
        self,
        ready: MapStyleABC,
        *,
        dataset_factory: DatasetFactory,
        selection_factory: DatasetFactory | None,
    ) -> MapStyleABC:
        if self.input_id is not None:
            return _ready_materialization_result(ready, selection_factory)
        source_view: SelectionView | None = None
        try:
            source_view = self._resolve_ready_input_id(
                dataset_factory=dataset_factory,
                selection_factory=selection_factory,
            )
            self._validate_ready_input_id(ready)
            if source_view is None:
                return ready
            return source_view.rebase(DatasetUniverse(ready))
        except BaseException:
            _close_resources_quietly(source_view, ready)
            raise

    def _resolve_ready_input_id(
        self,
        *,
        dataset_factory: DatasetFactory,
        selection_factory: DatasetFactory | None = None,
    ) -> SelectionView | None:
        if self.input_id is not None:
            return None
        source_factory = (
            dataset_factory if selection_factory is None else selection_factory
        )
        source: Any | None = None
        try:
            source = source_factory()
            dataset, source_view = _materialization_input(source)
            _validate_publishable_input(dataset)
            if not can_select_indexes(dataset):
                raise TypeError(
                    "Online materializing views require a map-style source dataset."
                )
            self._resolve_input_id(_materialization_identity_input(dataset))
            if selection_factory is None:
                identity_source = source
                source = None
                _close_resource(identity_source)
                return None
            if source_view is None:
                raise TypeError(
                    "selection_factory must return a SelectionView or DatasetUniverse."
                )
            return source_view
        except BaseException:
            _close_resources_quietly(source)
            raise

    def _validate_ready_input_id(self, ready: MapStyleABC) -> None:
        if self.input_id is None:
            raise RuntimeError("ready input identity was not resolved.")
        manifest = cast(Any, ready).manifest
        stored_input_id = manifest.provenance.get("input_id")
        if stored_input_id != self.input_id:
            raise ValueError(
                "Canonical store input_id does not match the materializer."
            )

    def _validate_online_contract(self) -> None:
        if self._materializer_mode not in {"view", "modality", "sample"}:
            raise TypeError("Unsupported online materializer mode.")
        if self.provider_id is None:
            raise ValueError("Online materialization requires provider_id.")
        if self.output is None or self.schema is None:
            raise ValueError("Online materialization requires output and schema.")
        if self.input_id is None and self.input_id_factory is None:
            raise ValueError(
                "Online materialization requires input_id or input_id_factory."
            )

    def _ready_dataset(self) -> MapStyleABC | None:
        from ..paths import dataset_ready_path
        from ..reader import StoreDataset, read_store_dataset

        root = Path(self.output_dir).expanduser()
        if not root.exists():
            return None
        if not root.is_dir():
            raise ValueError(f"Canonical store path is not a directory: {root}")
        if not dataset_ready_path(root).is_file():
            if any(root.iterdir()):
                raise ValueError(f"Canonical store exists but is not ready: {root}")
            return None

        dataset: StoreDataset | None = None
        try:
            dataset = read_store_dataset(
                root,
                views=_schema_views(cast(Schema, self.schema)),
                legacy_policy="reject",
            )
            manifest = dataset.manifest
            if manifest.dataset_id != self._dataset_id:
                raise ValueError(
                    "Canonical store dataset_id does not match the materializer."
                )
            if manifest.split != self.split:
                raise ValueError(
                    "Canonical store split does not match the materializer."
                )
            provenance = manifest.provenance
            if provenance.get("provider_id") != self.provider_id:
                raise ValueError(
                    "Canonical store provider_id does not match the materializer."
                )
            if provenance.get("output_id") != self._output_id:
                raise ValueError(
                    "Canonical store output contract does not match the materializer."
                )
            stored_input_id = provenance.get("input_id")
            if stored_input_id is None:
                raise ValueError("Canonical store is missing input_id provenance.")
            if self.input_id is not None and stored_input_id != self.input_id:
                raise ValueError(
                    "Canonical store input_id does not match the materializer."
                )
            return dataset
        except BaseException:
            if dataset is not None:
                _close_resources_quietly(dataset)
            raise

    def write(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: Devices = "auto",
        sample_indexes: Sequence[int] | None = None,
        max_new_samples: int | None = None,
        finalize: bool = True,
    ) -> Path | MaterializationStatus:
        if not isinstance(finalize, bool):
            raise TypeError("finalize must be a boolean.")
        if sample_indexes is not None and max_new_samples is not None:
            raise ValueError(
                "sample_indexes and max_new_samples are mutually exclusive."
            )
        if finalize and (sample_indexes is not None or max_new_samples is not None):
            raise ValueError(
                "sample_indexes and max_new_samples require finalize=False."
            )
        if max_new_samples is not None:
            max_new_samples = positive_int("max_new_samples", max_new_samples)
        resolved = resolve_devices(devices)
        if len(resolved) > 1 or self.num_workers > 0:
            validate_process_parent(
                context=(
                    f"{type(self).__name__} with multiple devices or DataLoader workers"
                )
            )
        with FileLock(materializer_lock_path(self.output_dir)):
            return self._write_resumable(
                dataset_factory=dataset_factory,
                provider_factory=provider_factory,
                devices=resolved,
                sample_indexes=sample_indexes,
                max_new_samples=max_new_samples,
                finalize=finalize,
            )

    def snapshot(
        self,
        output_dir: str | Path,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
    ) -> Path:
        """Publish the currently completed dense prefix without closing the run."""

        target = Path(output_dir).expanduser()
        source = Path(self.output_dir).expanduser()
        if target.resolve() == source.resolve():
            raise ValueError(
                "snapshot output_dir must differ from materializer output_dir."
            )
        with FileLock(materializer_lock_path(source)):
            fragments_dir = materializer_fragments_dir(
                source,
                staging_dir=self.staging_dir,
            )
            dataset, source_view = _materialization_input(dataset_factory())
            try:
                _validate_publishable_input(dataset)
                self._resolve_input_id(_materialization_identity_input(dataset))
                expected = dataset_sample_count(dataset, context="snapshot")
                expected_metadata = self._resume_metadata(
                    dataset,
                    dataset_factory=dataset_factory,
                    provider_factory=provider_factory,
                    expected=expected,
                    use_map_style_loader=can_select_indexes(dataset),
                    selection_agnostic=source_view is not None,
                )
                if stored_resume_metadata(fragments_dir) != expected_metadata:
                    raise ValueError(
                        "Snapshot materializer identity does not match the resume state."
                    )
                completed = compact_completed_fragment_indexes(
                    fragments_dir,
                    dataset_id=self._dataset_id,
                    split=self.split,
                    expected=expected,
                )
                if not completed:
                    raise ValueError(
                        "No completed materialization samples to snapshot."
                    )
                if completed[0] != 0 or completed[-1] != len(completed) - 1:
                    raise ValueError(
                        "A materialization snapshot requires a dense completed prefix "
                        "starting at sample index 0."
                    )
                return commit_store_fragments(
                    target,
                    fragments_dir,
                    dataset_id=self._dataset_id,
                    split=self.split,
                    expected_sample_count=len(completed),
                    max_shard_samples=self.max_shard_samples,
                    max_shard_bytes=self.max_shard_bytes,
                    provenance=self._provenance,
                )
            finally:
                _close_resource(dataset)

    def _write_resumable(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: tuple[str, ...],
        sample_indexes: Sequence[int] | None,
        max_new_samples: int | None,
        finalize: bool,
    ) -> Path | MaterializationStatus:
        if len(devices) == 1:
            device = devices[0]
            if self.runtime.uses_local_device:
                set_torch_device(device)
            return self._write_resumable_single(
                dataset_factory=dataset_factory,
                provider_factory=provider_factory,
                device=device,
                sample_indexes=sample_indexes,
                max_new_samples=max_new_samples,
                finalize=finalize,
            )
        return self._write_resumable_devices(
            dataset_factory=dataset_factory,
            provider_factory=provider_factory,
            devices=devices,
            sample_indexes=sample_indexes,
            max_new_samples=max_new_samples,
            finalize=finalize,
        )

    def _write_resumable_devices(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        devices: tuple[str, ...],
        sample_indexes: Sequence[int] | None,
        max_new_samples: int | None,
        finalize: bool,
    ) -> Path | MaterializationStatus:
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
        dataset, source_view = _materialization_input(dataset_factory())
        execution_factory: DatasetFactory = (
            _UniverseDatasetFactory(dataset_factory)
            if source_view is not None
            else dataset_factory
        )
        _validate_publishable_input(dataset)
        self._resolve_input_id(_materialization_identity_input(dataset))
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
                selection_agnostic=source_view is not None,
            ),
            staging_dir=self.staging_dir,
        )
        completed = compact_completed_fragment_indexes(
            fragments_dir,
            dataset_id=self._dataset_id,
            split=self.split,
            expected=expected,
        )
        missing = self._work_indexes(
            completed,
            expected,
            sample_indexes=sample_indexes,
            max_new_samples=max_new_samples,
            use_map_style_loader=use_map_style_loader,
        )
        if not missing:
            return self._finish_resumable(
                fragments_dir,
                expected,
                finalize=finalize,
            )
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
            dataset_factory=execution_factory,
            provider_factory=provider_factory,
            devices=devices,
            logs_dir=logs_dir,
            worker_logs_dir=worker_logs_dir,
            fragments_dir=fragments_dir,
            expected=expected,
            use_map_style_loader=use_map_style_loader,
            completed_indexes=completed,
            completed_count=len(completed),
            missing_indexes=missing,
            finalize=finalize,
        )
        if not finalize:
            completed = compact_completed_fragment_indexes(
                fragments_dir,
                dataset_id=self._dataset_id,
                split=self.split,
                expected=expected,
            )
            status = self._status(expected, completed)
            self._log_status(status)
            return status
        return self._finish_resumable(
            fragments_dir,
            expected,
            finalize=True,
            parts=True,
        )

    def _write_resumable_single(
        self,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        device: str,
        sample_indexes: Sequence[int] | None,
        max_new_samples: int | None,
        finalize: bool,
    ) -> Path | MaterializationStatus:
        output_dir = Path(self.output_dir).expanduser()
        dataset, source_view = _materialization_input(dataset_factory())
        execution_factory: DatasetFactory = (
            _UniverseDatasetFactory(dataset_factory)
            if source_view is not None
            else dataset_factory
        )
        _validate_publishable_input(dataset)
        self._resolve_input_id(_materialization_identity_input(dataset))
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
                selection_agnostic=source_view is not None,
            ),
            staging_dir=self.staging_dir,
        )
        completed = compact_completed_fragment_indexes(
            fragments_dir,
            dataset_id=self._dataset_id,
            split=self.split,
            expected=expected,
        )
        missing = self._work_indexes(
            completed,
            expected,
            sample_indexes=sample_indexes,
            max_new_samples=max_new_samples,
            use_map_style_loader=use_map_style_loader,
        )
        if not missing:
            return self._finish_resumable(
                fragments_dir,
                expected,
                finalize=finalize,
            )
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
            self._validate_provider_output(provider)
            if self.num_workers > 0:
                env = set_single_worker_environment(
                    device,
                    device_env="ANYDATASET_MATERIALIZE_DEVICE",
                )
                try:
                    self._write_resumable_loader_batches(
                        provider,
                        dataset_factory=execution_factory,
                        dataset=dataset,
                        sample_count=expected,
                        use_map_style_loader=use_map_style_loader,
                        completed_indexes=completed,
                        sample_indexes=missing,
                        fragments_dir=fragments_dir,
                        expected=expected,
                        progress=progress,
                    )
                finally:
                    restore_environment(env)
            else:
                self._write_resumable_indexed_batches(
                    sample_index_batches(
                        _missing_sample_records(
                            dataset,
                            missing,
                            use_map_style_loader=use_map_style_loader,
                        ),
                        self.batch_size,
                    ),
                    provider,
                    fragments_dir=fragments_dir,
                    expected=expected,
                    completed_indexes=completed,
                    sample_identity=dataset,
                    progress=progress,
                )
        return self._finish_resumable(
            fragments_dir,
            expected,
            finalize=finalize,
        )

    def _work_indexes(
        self,
        completed: Collection[int],
        expected: int,
        *,
        sample_indexes: Sequence[int] | None,
        max_new_samples: int | None,
        use_map_style_loader: bool,
    ) -> Sequence[int]:
        if sample_indexes is not None:
            if not use_map_style_loader:
                raise TypeError("sample_indexes requires a map-style dataset.")
            selected = _validate_sample_indexes(sample_indexes, expected)
            return tuple(index for index in selected if index not in completed)
        missing = missing_indexes(completed, expected)
        if max_new_samples is not None:
            if not use_map_style_loader:
                raise TypeError("max_new_samples requires a map-style dataset.")
            return missing[:max_new_samples]
        return missing

    def _finish_resumable(
        self,
        fragments_dir: Path,
        expected: int,
        *,
        finalize: bool,
        parts: bool = False,
    ) -> Path | MaterializationStatus:
        completed = compact_completed_fragment_indexes(
            fragments_dir,
            dataset_id=self._dataset_id,
            split=self.split,
            expected=expected,
        )
        if not indexes_complete(completed, expected):
            if finalize:
                raise RuntimeError(
                    "Materialization is incomplete; call write(finalize=False) "
                    "for staged work or provide the remaining samples. "
                    f"{expected - len(completed)} samples remain."
                )
            status = self._status(expected, completed)
            self._log_status(status)
            return status
        if not finalize:
            status = self._status(expected, completed)
            self._log_status(status)
            return status
        if parts:
            path = self._commit_parts(fragments_dir / ".parts")
        else:
            path = self._commit_fragments(fragments_dir, expected)
        self._log_published(path, expected=expected, parts=parts)
        return path

    def _status(
        self,
        expected: int,
        completed: Collection[int],
    ) -> MaterializationStatus:
        return MaterializationStatus(
            output_dir=Path(self.output_dir).expanduser(),
            expected=expected,
            completed=len(completed),
        )

    def _log_status(self, status: MaterializationStatus) -> None:
        write_info(
            "materializer",
            "materialization staged: "
            f"output_dir={status.output_dir!s} expected={status.expected} "
            f"completed={status.completed} pending={status.pending}",
            event="materializer_staged",
            fields={
                "materializer": type(self).__name__,
                "output_dir": status.output_dir,
                "split": self.split,
                "dataset_id": self._dataset_id,
                "expected": status.expected,
                "completed": status.completed,
                "pending": status.pending,
                "finalized": status.finalized,
                "batch_size": self.batch_size,
                "commit_samples": self.commit_samples,
                "max_shard_bytes": self.max_shard_bytes,
                "num_workers": self.num_workers,
                "write_workers": self.write_workers,
            },
        )

    def _log_published(self, path: Path, *, expected: int, parts: bool) -> None:
        write_info(
            "materializer",
            "published materialized store: "
            f"path={path!s} expected={expected} parts={parts}",
            event="materializer_published",
            fields={
                "materializer": type(self).__name__,
                "path": path,
                "output_dir": Path(self.output_dir).expanduser(),
                "split": self.split,
                "dataset_id": self._dataset_id,
                "expected": expected,
                "parts": parts,
                "batch_size": self.batch_size,
                "commit_samples": self.commit_samples,
                "max_shard_bytes": self.max_shard_bytes,
                "num_workers": self.num_workers,
                "write_workers": self.write_workers,
            },
        )

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
                max_shard_bytes=self.max_shard_bytes,
                provenance=self._provenance,
            ).write(())
            cleanup_materializer_resume_dir(
                self.output_dir,
                staging_dir=self.staging_dir,
            )
            return path
        with ProgressDashboard(
            desc="commit materialized views",
            total=expected,
            count_stage="merge-samples",
            stages=_COMMIT_PROGRESS_STAGES,
        ) as progress:
            path = commit_store_fragments(
                self.output_dir,
                fragments_dir,
                dataset_id=self._dataset_id,
                split=self.split,
                expected_sample_count=expected,
                max_shard_samples=self.max_shard_samples,
                max_shard_bytes=self.max_shard_bytes,
                provenance=self._provenance,
                progress=_commit_progress(progress),
            )
        cleanup_materializer_resume_dir(
            self.output_dir,
            staging_dir=self.staging_dir,
        )
        return path

    def _commit_parts(self, parts_dir: str | Path) -> Path:
        with ProgressDashboard(
            desc="commit materialized views",
            total=None,
            count_stage="merge-samples",
            stages=_COMMIT_PROGRESS_STAGES,
        ) as progress:
            path = commit_store_parts(
                self.output_dir,
                parts_dir,
                dataset_id=self._dataset_id,
                split=self.split,
                provenance=self._provenance,
                progress=_commit_progress(progress),
            )
        cleanup_materializer_resume_dir(
            self.output_dir,
            staging_dir=self.staging_dir,
        )
        return path

    def _write_resumable_loader_batches(
        self,
        provider: MaterializerProvider,
        *,
        dataset_factory: DatasetFactory,
        dataset: Any | None = None,
        sample_count: int | None = None,
        use_map_style_loader: bool | None = None,
        completed_indexes: Collection[int] | None = None,
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
            completed_indexes=completed_indexes,
            sample_identity=dataset,
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
        return sample_index_loader(
            dataset_factory,
            dataset=dataset,
            sample_count=sample_count,
            sample_indexes=sample_indexes,
            use_map_style_loader=use_map_style_loader,
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
        completed_indexes: Sequence[int] | None = None,
        finalize: bool = True,
    ) -> None:
        commit_samples = self.commit_samples
        if commit_samples is None:
            raise RuntimeError("materializer commit_samples was not initialized.")
        context = multiprocessing_context(self.runtime.process_start_method)
        progress = context.Queue()
        barrier = context.Barrier(len(devices))
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", free_port())
        worker_completed_indexes = () if use_map_style_loader else completed_indexes
        workers = [
            context.Process(
                target=materialize_worker,
                args=(
                    WorkerConfig(
                        output_dir=Path(self.output_dir),
                        dataset_id=self._dataset_id,
                        split=self.split,
                        provenance=self._provenance,
                        max_shard_samples=self.max_shard_samples,
                        max_shard_bytes=self.max_shard_bytes,
                        batch_size=self.batch_size,
                        commit_samples=commit_samples,
                        num_workers=self.num_workers,
                        prefetch_factor=self.prefetch_factor,
                        write_workers=self.write_workers,
                        write_prefetch=self.write_prefetch,
                        keep_schema=self.keep_schema,
                        output=self.output,
                        schema=self.schema,
                        roles=self._materializer_roles,
                        mode=self._materializer_mode,
                        runtime=self.runtime,
                        use_map_style_loader=use_map_style_loader,
                        completed_indexes=worker_completed_indexes,
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
                        finalize=finalize,
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
        started: list[ProcessHandle] = []
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

    @property
    def _materializer_roles(self) -> frozenset[Role] | None:
        return None

    def _resume_metadata(
        self,
        dataset: Any,
        *,
        dataset_factory: DatasetFactory,
        provider_factory: ProviderFactory,
        expected: int,
        use_map_style_loader: bool,
        selection_agnostic: bool = False,
    ) -> dict[str, object]:
        if selection_agnostic and self.input_id is None:
            raise ValueError(
                "Materializing a selected dataset requires a concrete input_id so "
                "the physical transform identity can exclude selection state."
            )
        identity_input = _materialization_identity_input(dataset)
        return {
            "schema_version": 7,
            "materializer": {
                "mode": self._materializer_mode,
                "dataset_id": self._dataset_id,
                "split": self.split,
                "max_shard_samples": self.max_shard_samples,
                "max_shard_bytes": self.max_shard_bytes,
                "keep_schema": metadata_value(self.keep_schema),
                "output": metadata_value(self.output),
                "schema": metadata_value(self.schema),
                "output_id": self._output_id,
                "roles": metadata_value(self._materializer_roles),
            },
            "input": {
                "type": (
                    f"{type(identity_input).__module__}."
                    f"{type(identity_input).__qualname__}"
                ),
                "factory": _resume_factory_identity(dataset_factory, self.input_id),
                "semantic_id": self.input_id,
                "sample_count": expected,
                "use_map_style_loader": use_map_style_loader,
                "store": _store_input_metadata(identity_input),
            },
            "provider": {
                "factory": _resume_factory_identity(
                    provider_factory,
                    self.provider_id,
                ),
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

    def _write_resumable_indexed_batches(
        self,
        batches: Iterable[Sequence[tuple[int, Sample]]],
        provider: MaterializerProvider,
        *,
        fragments_dir: Path,
        expected: int,
        completed_indexes: Collection[int] | None = None,
        sample_identity: object | None = None,
        progress: ProgressSink | None = None,
        worker_id: int = 0,
    ) -> None:
        completed = completed_indexes
        if completed is None:
            completed = compact_completed_fragment_indexes(
                fragments_dir,
                dataset_id=self._dataset_id,
                split=self.split,
                expected=expected,
            )
        writer = FragmentBatchWriter(
            strategy=_FragmentStrategy(self),
            config=self._fragment_config(),
            fragments_dir=fragments_dir,
            completed=completed,
            provider=provider,
            sample_identity=sample_identity,
            progress=progress,
            worker_id=worker_id,
        )
        writer.write(batches)

    def _fragment_config(self) -> FragmentBatchConfig:
        commit_samples = self.commit_samples
        if commit_samples is None:
            raise RuntimeError("materializer commit_samples was not initialized.")
        return FragmentBatchConfig(
            dataset_id=self._dataset_id,
            split=self.split,
            provenance=self._provenance,
            max_shard_samples=self.max_shard_samples,
            max_shard_bytes=self.max_shard_bytes,
            commit_samples=commit_samples,
            write_workers=self.write_workers,
            write_prefetch=self.write_prefetch,
            writer_start_method=self.runtime.writer_worker_start_method,
        )

    def _samples_with_batch_provider(
        self,
        samples: Sequence[Sample],
        provider: MaterializerProvider,
    ) -> Iterator[Sample]:
        return self._output_samples(
            samples,
            with_batch_view_provider(samples, cast(BatchViewProviderLike, provider)),
        )

    def _output_sample(self, source: Sample, output: Sample) -> Sample:
        combined = output
        if self.keep_schema is not None:
            kept = _select_sample(source, self.keep_schema)
            combined = _combine_output_sample(kept, output)
        if self.schema is not None:
            return _select_sample(combined, self.schema)
        return combined

    def _output_samples(
        self,
        sources: Sequence[Sample],
        outputs: Iterator[Sample],
    ) -> Iterator[Sample]:
        if self.keep_schema is None and self.schema is None:
            yield from outputs
            return
        for source, output in strict_zip(sources, outputs):
            yield self._output_sample(source, output)

    def _resilient_samples_with_batch_provider(
        self,
        samples: Sequence[Sample],
        provider: MaterializerProvider,
        *,
        worker_id: int = 0,
    ) -> Iterator[Sample]:
        def on_oom(batch_size: int, left_size: int, right_size: int) -> None:
            write_progress_message(
                "materialize views",
                "provider OOM: "
                f"worker={worker_id} provider={type(provider).__name__} "
                f"batch_size={batch_size}; retrying as {left_size}+{right_size} "
                "after cache cleanup",
            )
            write_warning(
                "materializer",
                "provider OOM split: "
                f"worker={worker_id} provider={type(provider).__name__} "
                f"batch_size={batch_size} retry={left_size}+{right_size}",
                event="materializer_provider_oom_split",
                fields={
                    "worker": worker_id,
                    "provider": type(provider).__name__,
                    "batch_size": batch_size,
                    "left_size": left_size,
                    "right_size": right_size,
                },
            )

        yield from with_resilient_batch_provider(
            samples,
            lambda batch: tuple(self._samples_with_batch_provider(batch, provider)),
            on_oom=on_oom,
        )

    def _uses_batch_provider(self, provider: MaterializerProvider) -> bool:
        return self.batch_size > 1 or bool(getattr(provider, "batch_only", False))

    def _validate_provider_output(self, provider: MaterializerProvider) -> None:
        if self.output is None:
            return
        actual = getattr(provider, "output", None)
        if actual != self.output:
            raise ValueError(
                "Materializer provider output does not match the configured "
                f"output: expected {self.output!r}, got {actual!r}."
            )


@dataclass
class MaterializingViewDataset(MapStyleABC):
    """Single-process online view with full-universe background coverage.

    Foreground access claims the requested indexes first when possible. A
    background sweep independently covers every remaining source index, so
    selection and sampler behavior cannot reduce transform materialization.
    Completed staging indexes suppress duplicate persistence but are recomputed
    when a caller needs their value because partial staging is never readable.
    The dataset owns its source, provider, background sink and materializer lock
    until ``close()``; closing waits for dense background coverage.
    """

    _source: Any = field(repr=False)
    _provider: Provider = field(repr=False)
    _materializer: ViewMaterializer = field(repr=False)
    _sink: FragmentOutputSink = field(repr=False)
    _lock: FileLock = field(repr=False)
    _owner_pid: int = field(repr=False)
    _owns_source: bool = field(default=True, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _coverage: CoverageCoordinator = field(init=False, repr=False)
    _operation_lock: Any = field(init=False, repr=False)
    _start_lock: Any = field(init=False, repr=False)
    _sweep_thread: Thread = field(init=False, repr=False)
    _sweep_started: bool = field(default=False, init=False, repr=False)
    _sweep_error: BaseException | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._coverage = CoverageCoordinator(
            len(self._source),
            completed=self._sink.completed,
        )
        self._operation_lock = Lock()
        self._start_lock = Lock()
        self._sweep_thread = Thread(
            target=self._run_full_sweep,
            name=f"anydataset-materialize-{self._materializer._dataset_id}",
            daemon=True,
        )

    def __len__(self) -> int:
        self._ensure_accessible()
        return len(self._source)

    def __getitem__(self, index: int) -> Sample:
        self._ensure_accessible()
        normalized = self._index(index)
        lease = self._coverage.claim(normalized, require_value=True)
        if lease.owner:
            try:
                self._materialize_owned((lease,))
            finally:
                self._start_sweep()
        else:
            self._start_sweep()
        return cast(Sample, lease.wait())

    def __getitems__(self, indexes: Sequence[int]) -> list[Sample]:
        self._ensure_accessible()
        normalized = tuple(self._index(index) for index in indexes)
        if not normalized:
            return []
        batch = self._coverage.claim_batch(normalized, require_value=True)
        if batch.owners:
            try:
                self._materialize_owned(batch.owners)
            finally:
                self._start_sweep()
        else:
            self._start_sweep()
        return list(cast(tuple[Sample, ...], batch.wait()))

    def cost_row(self, index: int) -> Any:
        self._ensure_accessible()
        normalized = self._index(index)
        cost_row = getattr(self._source, "cost_row", None)
        if callable(cost_row):
            return cost_row(normalized)
        return self._source[normalized]

    def sample_id(self, index: int) -> str:
        normalized = self._index(index)
        source_sample_id = inherited_sample_id(self._source, normalized)
        if source_sample_id is not None:
            return source_sample_id
        return sample_id(
            sample_id_prefix(self._materializer._dataset_id),
            normalized,
        )

    def global_index(self, index: int) -> int:
        normalized = self._index(index)
        global_index = getattr(self._source, "global_index", None)
        if not callable(global_index):
            return normalized
        value = global_index(normalized)
        if type(value) is not int:
            raise TypeError("global_index() must return an integer.")
        return value

    def universe_id(self) -> str:
        schema = self._materializer.schema
        if schema is None:
            raise RuntimeError("online materialization requires a complete schema.")
        value = materialized_universe_id(
            self._materializer._dataset_id,
            self._materializer.split,
            self._materializer._provenance,
            len(self._source),
            _schema_views(schema),
        )
        if value is None:
            raise RuntimeError(
                "online materialization provenance is incomplete for universe identity."
            )
        return value

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        self._ensure_accessible()
        source_shuffle = cast(
            Optional[Callable[..., Iterable[Sequence[int]]]],
            getattr(self._source, "_shuffle", None),
        )
        if callable(source_shuffle):
            yield from source_shuffle(
                shuffle=shuffle,
                seed=seed,
                epoch=epoch,
                num_replicas=num_replicas,
                rank=rank,
            )
            return
        yield from super()._shuffle(
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            num_replicas=num_replicas,
            rank=rank,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def flush(self) -> None:
        self._ensure_accessible()
        self._start_sweep()
        with self._operation_lock:
            self._sink.flush()
        self._raise_sweep_error()

    @property
    def coverage_complete(self) -> bool:
        return self._coverage.complete

    @property
    def completed_count(self) -> int:
        return self._coverage.completed_count

    def wait(self) -> None:
        """Wait until the background transform and staging writes are complete."""

        self._ensure_owner()
        self._start_sweep()
        try:
            self._coverage.wait_complete()
        finally:
            self._sweep_thread.join()
        self._raise_sweep_error()

    def close(self) -> None:
        self._ensure_owner()
        if self._closed:
            self._sink.close()
            return

        error: BaseException | None = None
        try:
            self.wait()
        except BaseException as exc:
            error = exc
        try:
            self._coverage.close()
        except BaseException as exc:
            if error is None:
                error = exc
        try:
            self._sink.close()
        except BaseException as exc:
            if error is None:
                error = exc
        resources = (
            (self._provider, self._source)
            if self._owns_source
            else (self._provider,)
        )
        for resource in resources:
            try:
                _close_resource(resource)
            except BaseException as exc:
                if error is None:
                    error = exc
        try:
            self._lock.__exit__(None, None, None)
        except BaseException as exc:
            if error is None:
                error = exc
        self._closed = True
        if error is not None:
            raise error

    def __enter__(self) -> MaterializingViewDataset:
        self._ensure_accessible()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        raise RuntimeError(
            "MaterializingViewDataset cannot be pickled; use DataLoader(num_workers=0)."
        )

    def _index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer.")
        length = len(self._source)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("materializing view dataset index out of range.")
        return index

    def _run_full_sweep(self) -> None:
        pending: list[CoverageLease] = []
        try:
            for lease in self._coverage.full_sweep():
                pending.append(lease)
                if len(pending) >= self._materializer.batch_size:
                    self._materialize_owned(tuple(pending))
                    pending.clear()
            if pending:
                self._materialize_owned(tuple(pending))
            self._coverage.wait_complete()
            with self._operation_lock:
                self._sink.flush()
        except BaseException as exc:
            self._sweep_error = exc
            self._coverage.abort(exc)

    def _start_sweep(self) -> None:
        with self._start_lock:
            if self._sweep_started:
                return
            self._sweep_started = True
            self._sweep_thread.start()

    def _materialize_owned(
        self,
        leases: Sequence[CoverageLease],
    ) -> tuple[Sample, ...]:
        if not leases:
            return ()
        indexes = tuple(lease.index for lease in leases)
        try:
            with self._operation_lock:
                sources = self._source_samples(indexes)
                batch_transform = getattr(self._provider, "batch_transform_fn", True)
                if batch_transform is not None and callable(
                    getattr(self._provider, "call_batch", None)
                ):
                    outputs = tuple(
                        self._materializer._resilient_samples_with_batch_provider(
                            sources,
                            self._provider,
                        )
                    )
                else:
                    outputs = tuple(
                        self._materializer._sample_with_provider(
                            sample,
                            self._provider,
                        )
                        for sample in sources
                    )
                validate_batch_outputs(outputs, len(sources))
                self._sink.submit(tuple(strict_zip(indexes, outputs)))
            for lease, output in strict_zip(leases, outputs):
                lease.complete(output)
            return outputs
        except BaseException as exc:
            for lease in leases:
                try:
                    lease.fail(exc)
                except BaseException:
                    pass
            self._coverage.abort(exc)
            raise

    def _source_samples(self, indexes: Sequence[int]) -> tuple[Sample, ...]:
        getitems = getattr(self._source, "__getitems__", None)
        if callable(getitems):
            samples = tuple(cast(Iterable[Sample], getitems(indexes)))
            validate_batch_outputs(samples, len(indexes))
            return samples
        return tuple(self._source[index] for index in indexes)

    def _raise_sweep_error(self) -> None:
        if self._sweep_error is not None:
            raise self._sweep_error

    def _ensure_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "MaterializingViewDataset cannot be used from a forked process; "
                "use DataLoader(num_workers=0)."
            )

    def _ensure_accessible(self) -> None:
        self._ensure_owner()
        if self._closed:
            raise RuntimeError("MaterializingViewDataset is closed.")
        self._raise_sweep_error()


@dataclass(frozen=True)
class _FragmentStrategy:
    materializer: ViewMaterializer

    def uses_batch_provider(self, provider: MaterializerProvider) -> bool:
        return self.materializer._uses_batch_provider(provider)

    def materialize_sample(
        self,
        sample: Sample,
        provider: MaterializerProvider,
    ) -> Sample:
        return self.materializer._sample_with_provider(sample, provider)

    def materialize_batch(
        self,
        samples: Sequence[Sample],
        provider: MaterializerProvider,
        *,
        worker_id: int,
    ) -> Sequence[Sample]:
        return tuple(
            self.materializer._resilient_samples_with_batch_provider(
                samples,
                provider,
                worker_id=worker_id,
            )
        )


@dataclass
class SampleMaterializer(ViewMaterializer):
    """Resumably materialize complete output samples from input samples.

    Unlike ViewMaterializer and ModalityMaterializer, this class delegates the
    whole sample transform to the provider. It is intended for dependent
    pipelines where a publishable unit must include multiple generated fields
    from the same source sample.
    """

    @property
    def _materializer_mode(self) -> _MaterializerMode:
        return "sample"

    def _sample_with_provider(
        self,
        sample: Sample,
        provider: MaterializerProvider,
    ) -> Sample:
        return self._output_sample(
            sample,
            _validated_sample_provider_output(
                cast(SampleProvider, provider)(sample),
            ),
        )

    def _samples_with_batch_provider(
        self,
        samples: Sequence[Sample],
        provider: MaterializerProvider,
    ) -> Iterator[Sample]:
        try:
            call_batch = cast(BatchSampleProvider, provider).call_batch
        except AttributeError as exc:
            raise TypeError(
                "batch_size > 1 requires sample provider.call_batch()."
            ) from exc
        outputs = tuple(
            _validated_sample_provider_output(output) for output in call_batch(samples)
        )
        validate_batch_outputs(outputs, len(samples))
        yield from self._output_samples(samples, iter(outputs))


@dataclass
class ModalityMaterializer(ViewMaterializer):
    roles: frozenset[Role] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.roles is not None:
            if not isinstance(self.roles, (set, frozenset, tuple, list)):
                raise TypeError("roles must be a collection of Role values or None.")
            normalized = frozenset(self.roles)
            if not normalized:
                raise ValueError("roles must not be empty.")
            if any(not isinstance(role, Role) for role in normalized):
                raise TypeError("roles must contain at least one Role value.")
            self.roles = normalized

    @property
    def _materializer_roles(self) -> frozenset[Role] | None:
        return self.roles

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
            with_modality_provider(
                sample,
                cast(ModalityProviderLike, provider),
                roles=self.roles,
            ),
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
                cast(BatchModalityProviderLike, provider),
                selected_roles=self.roles,
            ),
        )


@dataclass(frozen=True)
class MaterializerWorker:
    """Explicit adapter used by materializer subprocess entry points."""

    materializer: ViewMaterializer

    @property
    def dataset_id(self) -> str:
        return self.materializer._dataset_id

    def write_batches(
        self,
        provider: MaterializerProvider,
        *,
        dataset_factory: DatasetFactory,
        sample_count: int,
        use_map_style_loader: bool,
        completed_indexes: Sequence[int] | None,
        sample_indexes: Sequence[int],
        fragments_dir: Path,
        expected: int,
        progress: ProgressSink,
        worker_id: int,
    ) -> None:
        self.materializer._validate_provider_output(provider)
        dataset, _ = _materialization_input(dataset_factory())
        try:
            self.materializer._write_resumable_loader_batches(
                provider,
                dataset_factory=dataset_factory,
                dataset=dataset,
                sample_count=sample_count,
                use_map_style_loader=use_map_style_loader,
                completed_indexes=completed_indexes,
                sample_indexes=sample_indexes,
                fragments_dir=fragments_dir,
                expected=expected,
                progress=progress,
                worker_id=worker_id,
            )
        finally:
            _close_resource(dataset)


def _materialization_input(
    source: Any,
) -> tuple[Any, SelectionView | None]:
    """Separate complete provider input from the caller-facing selection."""

    if isinstance(source, SelectionView):
        return source.universe, source
    if isinstance(source, DatasetUniverse):
        return source, SelectionView(source)
    return source, None


def _materialization_identity_input(dataset: Any) -> Any:
    if isinstance(dataset, DatasetUniverse):
        return dataset.dataset
    return dataset


def _materialization_result(
    dataset: MaterializingViewDataset,
    source_view: SelectionView | None,
) -> MapStyleABC:
    if source_view is None:
        return dataset
    output_universe = DatasetUniverse(dataset)
    return source_view.rebase(output_universe)


def _ready_materialization_result(
    dataset: MapStyleABC,
    selection_factory: DatasetFactory | None,
) -> MapStyleABC:
    if selection_factory is None:
        return dataset
    source: Any | None = None
    try:
        source = selection_factory()
        _, source_view = _materialization_input(source)
        if source_view is None:
            raise TypeError(
                "selection_factory must return a SelectionView or DatasetUniverse."
            )
        return source_view.rebase(DatasetUniverse(dataset))
    except BaseException:
        _close_resources_quietly(source, dataset)
        raise


def _validated_sample_provider_output(output: object) -> Sample:
    if not isinstance(output, Mapping):
        raise TypeError("sample provider output must be a sample mapping.")
    return cast(Sample, output)


def _missing_sample_records(
    dataset: Any,
    indexes: Sequence[int],
    *,
    use_map_style_loader: bool,
) -> Iterator[tuple[int, Sample]]:
    if use_map_style_loader:
        for index in indexes:
            yield index, dataset[index]
        return
    yield from iter_shard(dataset, 1, 0)


def _validate_publishable_input(dataset: Any) -> None:
    from ...dataset.abc import AnyDataset
    from ...types import Source
    from ..manifest.schema import STORE_SCHEMA_VERSION
    from ..reader import StoreDataset, read_store_manifest

    dataset = _materialization_identity_input(dataset)
    if isinstance(dataset, StoreDataset):
        if dataset.manifest.schema_version != STORE_SCHEMA_VERSION:
            raise ValueError(
                f"Store schema_version {dataset.manifest.schema_version} is legacy "
                "and lacks provenance. Rematerialize or migrate the store before "
                "using it as materializer input."
            )
        return
    if isinstance(dataset, AnyDataset) and dataset.spec.source == Source.STORE:
        read_store_manifest(dataset.spec.path, legacy_policy="reject")


def _store_input_metadata(dataset: Any) -> dict[str, object] | None:
    from ...dataset.abc import AnyDataset
    from ...types import Source
    from ..reader import StoreDataset

    dataset = _materialization_identity_input(dataset)
    store: StoreDataset | None
    if isinstance(dataset, StoreDataset):
        store = dataset
    elif isinstance(dataset, AnyDataset) and dataset.spec.source == Source.STORE:
        prepared = dataset.dataset
        if not isinstance(prepared, StoreDataset):
            raise TypeError("store AnyDataset must prepare to StoreDataset.")
        store = prepared
    else:
        store = None
    if store is None:
        return None
    return {
        "schema_version": store.manifest.schema_version,
        "provenance": dict(store.manifest.provenance),
    }


def _select_sample(sample: Sample, schema: Schema) -> Sample:
    return select_sample(sample, schema)


def _combine_output_sample(left: Sample, right: Sample) -> Sample:
    return combine_samples(left, right, context="Materialized sample")


def _schema_views(schema: Schema) -> tuple[tuple[Role, Modality, View], ...]:
    views = (
        (role, modality, view)
        for (role, modality), requirement in schema.items()
        for view in requirement.views
    )
    return tuple(
        sorted(
            views,
            key=lambda item: (item[0].value, item[1].value, item[2].value),
        )
    )


def _resume_factory_identity(
    factory: object,
    semantic_id: str | None,
) -> object:
    if semantic_id is None:
        return callable_id(factory)
    return {
        "kind": "semantic",
        "id": semantic_id,
    }


def _close_resource(resource: object | None) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def _close_resources_quietly(*resources: object | None) -> None:
    """Attempt every cleanup while preserving the active setup failure."""

    for resource in resources:
        try:
            _close_resource(resource)
        except BaseException:
            pass


def _abort_online_open(
    sink: FragmentOutputSink | None,
    provider: object | None,
    source: object | None,
    lock: FileLock | None,
) -> None:
    """Release partial online resources while preserving the setup failure."""

    if sink is not None:
        try:
            sink.abort()
        except BaseException:
            pass
    _close_resources_quietly(provider, source)
    if lock is not None:
        try:
            lock.__exit__(None, None, None)
        except BaseException:
            pass


def _commit_progress(dashboard: ProgressDashboard) -> Callable[[str, int], None]:
    def put(stage: str, count: int) -> None:
        dashboard.put(Progress(0, count, False, None, stage=stage))

    return put


def _dataset_id(output_dir: str | Path) -> str:
    return Path(output_dir).expanduser().name or "dataset"


def _staging_dir(
    output_dir: str | Path,
    staging_dir: str | Path | None,
) -> Path | None:
    if staging_dir is None:
        return None
    output = Path(output_dir).expanduser().resolve()
    staging = Path(staging_dir).expanduser().resolve()
    if output == staging or output in staging.parents or staging in output.parents:
        raise ValueError(
            "staging_dir and output_dir must be separate, non-nested paths."
        )
    return staging


def _output_contract(
    output: View | None,
    schema: Schema | None,
) -> tuple[View | None, Schema | None]:
    if (output is None) != (schema is None):
        raise ValueError("output and schema must be configured together.")
    if output is None or schema is None:
        return None, None
    modality = output_modality(output)
    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a mapping or None.")
    normalized = dict(schema)
    if not normalized:
        raise ValueError("schema must not be empty.")
    for reference, requirement in normalized.items():
        if (
            not isinstance(reference, tuple)
            or len(reference) != 2
            or not isinstance(reference[0], Role)
            or not isinstance(reference[1], Modality)
        ):
            raise TypeError("schema keys must be (Role, Modality) tuples.")
        if not isinstance(requirement, reference[1].requirement_type()):
            raise TypeError(
                "schema requirement type must match its reference modality."
            )
    if not any(
        reference[1] is modality and output in requirement.views
        for reference, requirement in normalized.items()
    ):
        raise ValueError("schema must include the configured output view.")
    return output, normalized


def _validate_sample_indexes(
    indexes: Sequence[int],
    expected: int,
) -> tuple[int, ...]:
    if isinstance(indexes, (str, bytes, bytearray)) or not isinstance(
        indexes,
        Sequence,
    ):
        raise TypeError("sample_indexes must be a sequence of integers.")
    result: list[int] = []
    previous = -1
    for position, index in enumerate(indexes):
        if type(index) is not int:
            raise TypeError(f"sample_indexes[{position}] must be an integer.")
        if index <= previous:
            raise ValueError("sample_indexes must be strictly increasing.")
        if index < 0 or index >= expected:
            raise ValueError(
                f"sample_indexes[{position}] is outside the dataset: {index}."
            )
        result.append(index)
        previous = index
    return tuple(result)
