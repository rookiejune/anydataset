from __future__ import annotations

import multiprocessing
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Union

from .._compat import strict_zip
from .._progress import Progress, ProgressDashboard, put_progress
from .._resume import append_completed_index_cache, index_batch_id, pending_batch
from .._write_pipeline import BackgroundWriteSink
from ..types.item import Sample
from ._batch import validate_batch_outputs
from ._types import MaterializerProvider
from ._part_writer import DatasetFragmentWriter

if TYPE_CHECKING:
    from .materializer import ViewMaterializer


ProgressSink = Union[multiprocessing.Queue, ProgressDashboard]


@dataclass
class FragmentBatchWriter:
    materializer: ViewMaterializer
    fragments_dir: Path
    completed: set[int]
    provider: MaterializerProvider
    progress: ProgressSink | None = None
    worker_id: int = 0

    def __post_init__(self) -> None:
        self._inflight: set[int] = set()

    def write(self, batches: Iterable[Sequence[tuple[int, Sample]]]) -> None:
        with self._sink() as sink:
            pending_outputs: Deque[tuple[int, Sample]] = deque()
            read_start = time.perf_counter()
            for batch in batches:
                self._record_read(batch, read_start)
                pending = pending_batch(batch, self.completed | self._inflight)
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
        if not self.materializer._uses_batch_provider(self.provider):
            return tuple(
                (
                    index,
                    self.materializer._sample_with_provider(sample, self.provider),
                )
                for index, sample in batch
            )

        indexes = tuple(index for index, _sample in batch)
        samples = tuple(sample for _index, sample in batch)
        outputs = tuple(
            self.materializer._resilient_samples_with_batch_provider(
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
        commit_samples = self.materializer.commit_samples
        if commit_samples is None:
            raise RuntimeError("materializer commit_samples was not initialized.")
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
        sink.submit(
            FragmentWriteJob(
                fragments_dir=self.fragments_dir,
                dataset_id=self.materializer._dataset_id,
                split=self.materializer.split,
                max_shard_samples=self.materializer.max_shard_samples,
                indexes=indexes,
                samples=indexed,
            )
        )
        self._inflight.update(indexes)

    def _sink(self) -> BackgroundWriteSink[FragmentWriteJob]:
        return BackgroundWriteSink(
            write_fragment,
            workers=self.materializer.write_workers,
            max_pending=self.materializer.write_prefetch,
            start_method=self.materializer.runtime.writer_worker_start_method,
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
        self.completed.update(job.indexes)
        self._inflight.difference_update(job.indexes)
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
