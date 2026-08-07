"""Progress reporting for dataset-wide worker scans.

The module only counts completed iterations and worker lifecycle events. It does
not own sample indices, filter labels, store layout, or materializer semantics.
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from queue import Empty
from types import TracebackType
from typing import Any, Protocol, TypeVar, cast

from .parallel import ProcessHandle

_PROGRESS_INTERVAL = 1.0
_NON_INTERACTIVE_PROGRESS_INTERVAL = 10.0
ItemT = TypeVar("ItemT")
MessageT = TypeVar("MessageT", contravariant=True)


class _ProgressBar(Protocol):
    def __enter__(self) -> Any: ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any: ...

    def update(self, count: int) -> None: ...

    def set_postfix_str(self, value: str, *, refresh: bool = True) -> None: ...


class ProgressWriter(Protocol[MessageT]):
    def put(self, message: MessageT, /) -> Any: ...


@dataclass(frozen=True)
class Progress:
    worker_id: int
    samples: int
    done: bool
    error: str | None
    stage: str = "samples"
    elapsed: float | None = None
    pending: int | None = None
    details: Mapping[str, object] | None = None


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


def iter_with_progress(
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
    workers: Sequence[ProcessHandle],
    progress: multiprocessing.Queue,
    *,
    desc: str,
    early_exit_message: str,
    failure_prefix: str,
    total: int | None = None,
    count_stage: str | None = None,
    initial: int = 0,
    stages: tuple[str, ...] = (),
) -> tuple[Mapping[str, object], ...]:
    done = 0
    summaries: list[Mapping[str, object]] = []
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
                if message.details is not None:
                    summaries.append(dict(message.details))
                if message.error is not None:
                    raise RuntimeError(
                        f"{failure_prefix} {message.worker_id} failed.\n{message.error}"
                    )
    return tuple(summaries)


def put_progress(progress: ProgressWriter[Progress], message: Progress) -> None:
    progress.put(message)


def write_progress_message(desc: str, message: str) -> None:
    line = f"{desc}: {message}"
    if not sys.stdout.isatty():
        print(line, file=sys.stdout, flush=True)
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print(line, file=sys.stdout, flush=True)
        return
    tqdm.write(line, file=sys.stdout)


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
        self._run_samples = 0
        self._bar: _ProgressBar | None = None
        self._stage_bars: dict[str, _ProgressBar] = {}

    def __enter__(self) -> ProgressDashboard:
        self._bar = _progress_bar(
            desc=self.desc,
            total=self.total,
            initial=self.initial,
            postfix=_format_run_progress(
                initial=self.initial,
                run_samples=self._run_samples,
                total=self.total,
            ),
            position=0,
        )
        self._bar.__enter__()
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for stage in reversed(self.stages):
            bar = self._stage_bars.get(stage)
            if bar is not None:
                bar.__exit__(exc_type, exc_value, traceback)
        self._stage_bars.clear()
        if self._bar is not None:
            self._bar.__exit__(exc_type, exc_value, traceback)
            self._bar = None

    def put(self, message: Progress) -> None:
        if not isinstance(message, Progress):
            return
        if (
            message.samples
            or message.elapsed is not None
            or message.pending is not None
        ):
            stats = self._stats.setdefault(message.stage, _StageStats())
            stats.update(message)
        counts_bar = bool(message.samples and self._counts_bar(message))
        if counts_bar:
            self._run_samples += message.samples
        if self.stages or self.initial:
            self._update_stage_bar(message)
            self._set_postfix(
                _format_progress_postfix(
                    initial=self.initial,
                    run_samples=self._run_samples,
                    total=self.total,
                    stats=self._stats,
                    stages=self.stages,
                ),
                refresh=not counts_bar,
            )
        if counts_bar:
            self._update_bar(message.samples)

    def _counts_bar(self, message: Progress) -> bool:
        if self.count_stage is None:
            return True
        return message.stage == self.count_stage

    def _update_bar(self, samples: int) -> None:
        if self._bar is not None:
            self._bar.update(samples)

    def _set_postfix(self, value: str, *, refresh: bool) -> None:
        if self._bar is not None:
            self._bar.set_postfix_str(value, refresh=refresh)

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


def _dead_worker(workers: Sequence[ProcessHandle]) -> bool:
    return any(worker.exitcode not in (None, 0) for worker in workers)


def _progress_bar(
    *,
    desc: str,
    total: int | None,
    initial: int = 0,
    postfix: str = "",
    position: int = 0,
    leave: bool = True,
) -> _ProgressBar:
    if not sys.stdout.isatty():
        if position > 0:
            return _NullProgressBar()
        return _LogProgressBar(
            desc=desc,
            total=total,
            initial=initial,
            postfix=postfix,
        )
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NullProgressBar()
    bar = tqdm(
        total=total,
        initial=initial,
        unit="sample",
        desc=desc,
        position=position,
        leave=leave,
        file=sys.stdout,
    )
    if postfix:
        bar.set_postfix_str(postfix, refresh=False)
    return cast(
        _ProgressBar,
        bar,
    )


def _format_progress_postfix(
    *,
    initial: int,
    run_samples: int,
    total: int | None,
    stats: dict[str, _StageStats],
    stages: tuple[str, ...],
) -> str:
    parts = [
        _format_run_progress(
            initial=initial,
            run_samples=run_samples,
            total=total,
        ),
        _format_stage_postfix(stats, stages),
    ]
    return " | ".join(part for part in parts if part)


def _format_run_progress(
    *,
    initial: int,
    run_samples: int,
    total: int | None,
) -> str:
    if initial <= 0:
        return ""
    run = str(run_samples)
    if total is not None:
        run += f"/{max(0, total - initial)}"
    return f"resumed={initial} | run={run}"


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

    def set_postfix_str(self, value: str, *, refresh: bool = True) -> None:
        return None


class _LogProgressBar:
    def __init__(
        self,
        *,
        desc: str,
        total: int | None,
        initial: int = 0,
        postfix: str = "",
    ) -> None:
        self.desc = desc
        self.total = total
        self.initial = initial
        self.count = initial
        self.postfix = postfix
        self.started_at = time.monotonic()
        self.last_printed_at: float | None = None

    def __enter__(self):
        self._print(force=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._print(force=True)

    def update(self, count: int) -> None:
        self.count += count
        self._print()

    def set_postfix_str(self, value: str, *, refresh: bool = True) -> None:
        self.postfix = value
        if refresh:
            self._print()

    def _print(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and self.last_printed_at is not None
            and now - self.last_printed_at < _NON_INTERACTIVE_PROGRESS_INTERVAL
        ):
            return
        elapsed = max(0.0, now - self.started_at)
        run_samples = max(0, self.count - self.initial)
        rate = run_samples / elapsed if elapsed > 0 else 0.0
        progress = f"{self.count} sample"
        if self.total is not None:
            percent = 100.0 if self.total == 0 else 100.0 * self.count / self.total
            progress += f"/{self.total} ({percent:.1f}%)"
        timing = f"{elapsed:.0f}s, {rate:.1f} sample/s"
        if self.total is not None and rate > 0 and self.count < self.total:
            timing += f", ETA {(self.total - self.count) / rate:.0f}s"
        line = f"{self.desc}: {progress} [{timing}]"
        if self.postfix:
            line += f" {self.postfix}"
        print(line, file=sys.stdout, flush=True)
        self.last_printed_at = now
