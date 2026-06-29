from __future__ import annotations

"""Progress reporting for dataset-wide worker scans.

The module only counts completed iterations and worker lifecycle events. It does
not own sample indices, filter labels, store layout, or materializer semantics.
"""

import multiprocessing
import time
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Empty

_PROGRESS_INTERVAL = 1.0


@dataclass(frozen=True)
class Progress:
    worker_id: int
    samples: int
    done: bool
    error: str | None


def iter_with_progress[ItemT](
    items: Iterator[ItemT],
    *,
    worker_id: int,
    progress: multiprocessing.Queue,
) -> Iterator[ItemT]:
    pending = 0
    last_flush = time.monotonic()
    try:
        for item in items:
            yield item
            pending += 1
            now = time.monotonic()
            if now - last_flush >= _PROGRESS_INTERVAL:
                put_progress(progress, Progress(worker_id, pending, False, None))
                pending = 0
                last_flush = now
    finally:
        if pending:
            put_progress(progress, Progress(worker_id, pending, False, None))


def watch_workers(
    workers: list[multiprocessing.Process],
    progress: multiprocessing.Queue,
    *,
    desc: str,
    early_exit_message: str,
    failure_prefix: str,
    total: int | None = None,
) -> None:
    done = 0
    with _progress_bar(desc=desc, total=total) as bar:
        while done < len(workers):
            try:
                message = progress.get(timeout=0.2)
            except Empty:
                if _dead_worker(workers):
                    raise RuntimeError(early_exit_message)
                continue
            if not isinstance(message, Progress):
                continue
            if message.samples:
                bar.update(message.samples)
            if message.done:
                done += 1
                if message.error is not None:
                    raise RuntimeError(
                        f"{failure_prefix} {message.worker_id} failed.\n"
                        f"{message.error}"
                    )


def put_progress(progress: multiprocessing.Queue, message: Progress) -> None:
    progress.put(message)


def _dead_worker(workers: list[multiprocessing.Process]) -> bool:
    return any(worker.exitcode not in (None, 0) for worker in workers)


def _progress_bar(*, desc: str, total: int | None):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NullProgressBar()
    return tqdm(total=total, unit="sample", desc=desc)


class _NullProgressBar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def update(self, count: int) -> None:
        return None
