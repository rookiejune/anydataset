"""Progress reporting for dataset-wide worker scans.

The module only counts completed iterations and worker lifecycle events. It does
not own sample indices, filter labels, store layout, or materializer semantics.
"""

from __future__ import annotations

import multiprocessing
import sys
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
    stage: str = "samples"
    elapsed: float | None = None
    pending: int | None = None


@dataclass
class _StageStats:
    samples: int = 0
    elapsed: float = 0.0
    last_elapsed: float | None = None
    pending: int | None = None
    first_update: float | None = None
    last_update: float | None = None

    def update(self, message: Progress) -> None:
        now = time.monotonic()
        if self.first_update is None:
            self.first_update = now
        self.last_update = now
        self.samples += message.samples
        if message.elapsed is not None:
            self.elapsed += message.elapsed
            self.last_elapsed = message.elapsed
        if message.pending is not None:
            self.pending = message.pending


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
    count_stage: str | None = None,
    initial: int = 0,
    stages: tuple[str, ...] = (),
) -> None:
    done = 0
    with ProgressDashboard(
        desc=desc,
        total=total,
        count_stage=count_stage,
        initial=initial,
        stages=stages,
    ) as dashboard:
        while done < len(workers):
            try:
                message = progress.get(timeout=0.2)
            except Empty:
                if _dead_worker(workers):
                    raise RuntimeError(early_exit_message)
                continue
            if not isinstance(message, Progress):
                continue
            dashboard.put(message)
            if message.done:
                done += 1
                if message.error is not None:
                    raise RuntimeError(
                        f"{failure_prefix} {message.worker_id} failed.\n"
                        f"{message.error}"
                    )


def put_progress(progress: multiprocessing.Queue, message: Progress) -> None:
    progress.put(message)


class ProgressDashboard:
    def __init__(
        self,
        *,
        desc: str,
        total: int | None,
        count_stage: str | None = None,
        initial: int = 0,
        stages: tuple[str, ...] = (),
    ) -> None:
        self.desc = desc
        self.total = total
        self.count_stage = count_stage
        self.initial = initial
        self.stages = stages
        self._stats: dict[str, _StageStats] = {}
        self._bar = None
        self._stage_bars: dict[str, object] = {}

    def __enter__(self):
        self._bar = _progress_bar(desc=self.desc, total=self.total, position=0)
        self._bar.__enter__()
        if self.initial:
            self._bar.update(self.initial)
        for position, stage in enumerate(self.stages, start=1):
            bar = _progress_bar(
                desc=f"{stage:>8}",
                total=self._stage_total(),
                position=position,
                leave=False,
            )
            bar.__enter__()
            self._stage_bars[stage] = bar
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for bar in reversed(tuple(self._stage_bars.values())):
            bar.__exit__(exc_type, exc_value, traceback)
        self._stage_bars.clear()
        if self._bar is not None:
            self._bar.__exit__(exc_type, exc_value, traceback)
            self._bar = None

    def put(self, message: Progress) -> None:
        if not isinstance(message, Progress):
            return
        if message.samples or message.elapsed is not None or message.pending is not None:
            stats = self._stats.setdefault(message.stage, _StageStats())
            stats.update(message)
        if message.samples and self._counts_bar(message):
            self._update_bar(message.samples)
        if self.stages:
            self._update_stage_bar(message)
            self._set_postfix(_format_stage_postfix(self._stats, self.stages))

    def _counts_bar(self, message: Progress) -> bool:
        if self.count_stage is None:
            return True
        return message.stage == self.count_stage

    def _update_bar(self, samples: int) -> None:
        if self._bar is not None:
            self._bar.update(samples)

    def _set_postfix(self, value: str) -> None:
        if self._bar is not None:
            self._bar.set_postfix_str(value)

    def _update_stage_bar(self, message: Progress) -> None:
        bar = self._stage_bars.get(message.stage)
        if bar is None:
            return
        if message.samples:
            bar.update(message.samples)
        stats = self._stats.get(message.stage)
        if stats is not None:
            bar.set_postfix_str(_format_stage_stats(stats))

    def _stage_total(self) -> int | None:
        if self.total is None:
            return None
        return max(0, self.total - self.initial)


def _dead_worker(workers: list[multiprocessing.Process]) -> bool:
    return any(worker.exitcode not in (None, 0) for worker in workers)


def _progress_bar(
    *,
    desc: str,
    total: int | None,
    position: int = 0,
    leave: bool = True,
):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NullProgressBar()
    return tqdm(
        total=total,
        unit="sample",
        desc=desc,
        position=position,
        leave=leave,
        disable=not sys.stderr.isatty(),
    )


def _format_stage_postfix(
    stats: dict[str, _StageStats],
    stages: tuple[str, ...],
) -> str:
    parts: list[str] = []
    for stage in stages:
        value = stats.get(stage)
        if value is None:
            parts.append(f"{stage}=0")
            continue
        segment = f"{stage}={value.samples}"
        rate = _stage_rate(value)
        if rate is not None:
            segment += f" {rate:.1f}/s"
        if value.pending is not None:
            segment += f" pending={value.pending}"
        if value.last_elapsed is not None:
            segment += f" last={value.last_elapsed:.2f}s"
        parts.append(segment)
    return " | ".join(parts)


def _format_stage_stats(stats: _StageStats) -> str:
    parts: list[str] = []
    rate = _stage_rate(stats)
    if rate is not None:
        parts.append(f"{rate:.1f}/s")
    if stats.pending is not None:
        parts.append(f"pending={stats.pending}")
    if stats.last_elapsed is not None:
        parts.append(f"last={stats.last_elapsed:.2f}s")
    return " ".join(parts)


def _stage_rate(stats: _StageStats) -> float | None:
    if stats.samples <= 0:
        return None
    if stats.elapsed > 0:
        return stats.samples / stats.elapsed
    if stats.first_update is None or stats.last_update is None:
        return None
    elapsed = stats.last_update - stats.first_update
    if elapsed <= 0:
        return None
    return stats.samples / elapsed


class _NullProgressBar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def update(self, count: int) -> None:
        return None

    def set_postfix_str(self, value: str) -> None:
        return None
