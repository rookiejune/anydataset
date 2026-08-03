from __future__ import annotations

import multiprocessing
import time
from collections import deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Deque, Protocol, Union

from ..._compat import strict_zip
from ..._runtime.parallel import StartMethod
from ..._runtime.progress import Progress, ProgressDashboard, put_progress
from ..._runtime.resume import append_completed_index_cache, index_batch_id
from ..._runtime.write_pipeline import BackgroundWriteSink
from ...types.item import Sample
from .batch import validate_batch_outputs
from .types import MaterializerProvider
from ..part.writer import DatasetFragmentWriter


ProgressSink = Union[multiprocessing.Queue, ProgressDashboard]


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
    commit_samples: int
    write_workers: int
    write_prefetch: int | None
    writer_start_method: StartMethod


@dataclass
class FragmentBatchWriter:
    strategy: FragmentStrategy
    config: FragmentBatchConfig
    fragments_dir: Path
    completed: Collection[int]
    provider: MaterializerProvider
    progress: ProgressSink | None = None
    worker_id: int = 0

    def __post_init__(self) -> None:
        self._submitted_indexes: set[int] = set()

    def write(self, batches: Iterable[Sequence[tuple[int, Sample]]]) -> None:
        with self._sink() as sink:
            pending_outputs: Deque[tuple[int, Sample]] = deque()
            read_start = time.perf_counter()
            for batch in batches:
                self._record_read(batch, read_start)
                pending = self._pending_batch(batch)
                if not pending:
                    read_start = time.perf_counter()
                    continue
                outputs = self._materialized_batch(pending)
                pending_outputs.extend(outputs)
                self._flush_ready(sink, pending_outputs)
                read_start = time.perf_counter()
            self._flush_remaining(sink, pending_outputs)

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

    def _flush_ready(
        self,
        sink: BackgroundWriteSink[FragmentWriteJob],
        pending_outputs: Deque[tuple[int, Sample]],
    ) -> None:
        commit_samples = self.config.commit_samples
        while len(pending_outputs) >= commit_samples:
            samples = tuple(islice(pending_outputs, commit_samples))
            self._submit(sink, samples)
            for _ in range(commit_samples):
                pending_outputs.popleft()

    def _flush_remaining(
        self,
        sink: BackgroundWriteSink[FragmentWriteJob],
        pending_outputs: Deque[tuple[int, Sample]],
    ) -> None:
        if pending_outputs:
            self._submit(sink, pending_outputs)

    def _submit(
        self,
        sink: BackgroundWriteSink[FragmentWriteJob],
        samples: Sequence[tuple[int, Sample]],
    ) -> None:
        indexed = tuple(samples)
        indexes = tuple(sorted(index for index, _ in indexed))
        job = FragmentWriteJob(
            fragments_dir=self.fragments_dir,
            dataset_id=self.config.dataset_id,
            split=self.config.split,
            provenance=self.config.provenance,
            max_shard_samples=self.config.max_shard_samples,
            indexes=indexes,
            samples=indexed,
        )
        sink.submit(job)

    def _sink(self) -> BackgroundWriteSink[FragmentWriteJob]:
        return BackgroundWriteSink(
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


@dataclass(frozen=True)
class FragmentWriteJob:
    fragments_dir: Path
    dataset_id: str
    split: str | None
    provenance: Mapping[str, str]
    max_shard_samples: int
    indexes: tuple[int, ...]
    samples: tuple[tuple[int, Sample], ...]


def write_fragment(job: FragmentWriteJob) -> None:
    fragment_id = index_batch_id(job.indexes)
    DatasetFragmentWriter(
        job.fragments_dir / fragment_id,
        dataset_id=job.dataset_id,
        split=job.split,
        fragment_id=fragment_id,
        provenance=job.provenance,
        max_shard_samples=job.max_shard_samples,
    ).write(job.samples)
    append_completed_index_cache(job.fragments_dir, fragment_id, job.indexes)


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
