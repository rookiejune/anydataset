from __future__ import annotations

import time
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..._compat import strict_zip
from ..._runtime.parallel import StartMethod
from ..._runtime.progress import Progress, ProgressWriter, put_progress
from ..._runtime.resume import index_batch_id
from ..._runtime.write_pipeline import BackgroundWriteSink
from ..._validation import non_negative_int
from ...types.item import Sample
from .batch import validate_batch_outputs
from .types import MaterializerProvider
from ..part.writer import DatasetFragmentWriter, SampleRecord, sample_record


ProgressSink = ProgressWriter[Progress]


class FragmentStrategy(Protocol):
    def uses_batch_provider(self, provider: MaterializerProvider) -> bool: ...

    def materialize_sample(
        self,
        sample: Sample,
        provider: MaterializerProvider,
    ) -> Sample: ...

    def materialize_batch(
        self,
        samples: Sequence[Sample],
        provider: MaterializerProvider,
        *,
        worker_id: int,
    ) -> Sequence[Sample]: ...


@dataclass(frozen=True)
class FragmentBatchConfig:
    dataset_id: str
    split: str | None
    provenance: Mapping[str, str]
    max_shard_samples: int
    max_shard_bytes: int | None
    commit_samples: int
    write_workers: int
    write_prefetch: int | None
    writer_start_method: StartMethod


@dataclass
class FragmentOutputSink:
    """Persist already materialized samples into resumable immutable fragments."""

    config: FragmentBatchConfig
    fragments_dir: Path
    completed: Collection[int]
    progress: ProgressSink | None = None
    worker_id: int = 0
    expected: int | None = None
    sample_identity: object | None = field(default=None, repr=False)
    _pending_outputs: list[SampleRecord] = field(
        init=False,
        repr=False,
        default_factory=list,
    )
    _reserved_indexes: set[int] = field(
        init=False,
        repr=False,
        default_factory=set,
    )
    _background: BackgroundWriteSink[FragmentWriteJob] = field(
        init=False,
        repr=False,
    )
    _closed: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self.fragments_dir = Path(self.fragments_dir)
        if self.expected is not None:
            self.expected = non_negative_int("expected", self.expected)
        self._background = BackgroundWriteSink(
            write_fragment,
            workers=self.config.write_workers,
            max_pending=self.config.write_prefetch,
            start_method=self.config.writer_start_method,
            on_submit=lambda job, pending: put_stage_progress(
                self.progress,
                worker_id=self.worker_id,
                stage="writer",
                pending=pending,
            ),
            on_complete=self._on_write_complete,
        )

    def __enter__(self) -> FragmentOutputSink:
        self._background.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        self.abort()

    def submit(self, outputs: Sequence[tuple[int, Sample]]) -> None:
        self._background.raise_if_failed()
        if self._closed:
            raise RuntimeError("fragment output sink is already closed.")
        for index, _sample in outputs:
            self._validate_index(index)
        for index, sample in outputs:
            if index in self.completed or index in self._reserved_indexes:
                continue
            self._reserved_indexes.add(index)
            self._pending_outputs.append(
                sample_record(self.sample_identity, index, sample)
            )
        while len(self._pending_outputs) >= self.config.commit_samples:
            self._submit_pending(self.config.commit_samples)

    def flush(self) -> None:
        if self._closed:
            self._background.flush()
            return
        error: BaseException | None = None
        try:
            if self._pending_outputs:
                self._submit_pending(len(self._pending_outputs))
        except BaseException as exc:
            error = exc
        try:
            self._background.flush()
        except BaseException:
            raise
        if error is not None:
            raise error

    def close(self) -> None:
        if self._closed:
            self._background.close()
            return
        try:
            if self._pending_outputs:
                self._submit_pending(len(self._pending_outputs))
        finally:
            try:
                self._background.close()
            finally:
                self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self._pending_outputs.clear()
        self._background.abort()
        self._closed = True

    def _submit_pending(self, count: int) -> None:
        samples = tuple(
            sorted(self._pending_outputs[:count], key=lambda item: item[0])
        )
        indexes = tuple(record[0] for record in samples)
        job = FragmentWriteJob(
            fragments_dir=self.fragments_dir,
            dataset_id=self.config.dataset_id,
            split=self.config.split,
            provenance=self.config.provenance,
            max_shard_samples=self.config.max_shard_samples,
            max_shard_bytes=self.config.max_shard_bytes,
            indexes=indexes,
            samples=samples,
        )
        self._background.submit(job)
        del self._pending_outputs[:count]

    def _validate_index(self, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Materialized sample indexes must be integers.")
        if index < 0 or (self.expected is not None and index >= self.expected):
            raise ValueError(f"Materialized sample index is outside dataset: {index}.")

    def _on_write_complete(
        self,
        job: FragmentWriteJob,
        pending: int,
        elapsed: float,
    ) -> None:
        put_stage_progress(
            self.progress,
            worker_id=self.worker_id,
            stage="writer",
            samples=len(job.samples),
            elapsed=elapsed,
            pending=pending,
        )


@dataclass
class FragmentBatchWriter:
    strategy: FragmentStrategy
    config: FragmentBatchConfig
    fragments_dir: Path
    completed: Collection[int]
    provider: MaterializerProvider
    sample_identity: object | None = field(default=None, repr=False)
    progress: ProgressSink | None = None
    worker_id: int = 0

    def __post_init__(self) -> None:
        self._submitted_indexes: set[int] = set()

    def write(self, batches: Iterable[Sequence[tuple[int, Sample]]]) -> None:
        with self._output_sink() as sink:
            read_start = time.perf_counter()
            for batch in batches:
                self._record_read(batch, read_start)
                pending = self._pending_batch(batch)
                if not pending:
                    read_start = time.perf_counter()
                    continue
                outputs = self._materialized_batch(pending)
                sink.submit(outputs)
                read_start = time.perf_counter()

    def _pending_batch(
        self,
        batch: Sequence[tuple[int, Sample]],
    ) -> tuple[tuple[int, Sample], ...]:
        pending: list[tuple[int, Sample]] = []
        previous: int | None = None
        for index, sample in batch:
            if previous is not None:
                if index <= previous:
                    raise ValueError(
                        "Materializer sample indexes must be in increasing order "
                        "within each batch."
                    )
            previous = index
            if index in self.completed:
                continue
            if index in self._submitted_indexes:
                raise ValueError(
                    "Materializer sample indexes must be unique within a run."
                )
            pending.append((index, sample))
        self._submitted_indexes.update(index for index, _sample in pending)
        return tuple(pending)

    def _record_read(
        self,
        batch: Sequence[tuple[int, Sample]],
        read_start: float,
    ) -> None:
        put_stage_progress(
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
        put_stage_progress(
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
        if not self.strategy.uses_batch_provider(self.provider):
            return tuple(
                (
                    index,
                    self.strategy.materialize_sample(sample, self.provider),
                )
                for index, sample in batch
            )

        indexes = tuple(index for index, _sample in batch)
        samples = tuple(sample for _index, sample in batch)
        outputs = tuple(
            self.strategy.materialize_batch(
                samples,
                self.provider,
                worker_id=self.worker_id,
            )
        )
        validate_batch_outputs(outputs, len(samples))
        return tuple(strict_zip(indexes, outputs))

    def _output_sink(self) -> FragmentOutputSink:
        return FragmentOutputSink(
            config=self.config,
            fragments_dir=self.fragments_dir,
            completed=self.completed,
            sample_identity=self.sample_identity,
            progress=self.progress,
            worker_id=self.worker_id,
        )


@dataclass(frozen=True)
class FragmentWriteJob:
    fragments_dir: Path
    dataset_id: str
    split: str | None
    provenance: Mapping[str, str]
    max_shard_samples: int
    max_shard_bytes: int | None
    indexes: tuple[int, ...]
    samples: tuple[SampleRecord, ...]


def write_fragment(job: FragmentWriteJob) -> None:
    fragment_id = index_batch_id(job.indexes)
    DatasetFragmentWriter(
        job.fragments_dir / fragment_id,
        dataset_id=job.dataset_id,
        split=job.split,
        fragment_id=fragment_id,
        provenance=job.provenance,
        max_shard_samples=job.max_shard_samples,
        max_shard_bytes=job.max_shard_bytes,
    ).write(job.samples)


def put_stage_progress(
    progress: ProgressSink | None,
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
