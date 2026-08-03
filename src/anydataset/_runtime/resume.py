"""Shared helpers for indexed resume workflows.

This module owns hidden resume directories, dataset length checks, index
coverage checks, and deterministic batch ids. It does not read or write filter
partitions, metrics, store parts, or store fragments; callers keep those
domain-specific formats at their own layer.
"""

from __future__ import annotations

import errno
import hashlib
import heapq
import os
import shutil
import time
from array import array
from bisect import bisect_right
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TypeVar, overload

from .logging import write_info, write_warning
from .._validation import validate_path_segment

T = TypeVar("T")

_CLEANUP_RESUME_ATTEMPTS = 5
_CLEANUP_RESUME_RETRY_DELAY_SECONDS = 0.2
_CLEANUP_RESUME_RETRY_ERRNOS = {
    errno.ENOTEMPTY,
    errno.EBUSY,
}
if hasattr(errno, "ESTALE"):
    _CLEANUP_RESUME_RETRY_ERRNOS.add(errno.ESTALE)

def resume_root(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir).expanduser()
    return output_dir.parent / f".{output_dir.name}.resume"


def resume_dir(output_dir: str | Path, name: str) -> Path:
    validate_path_segment("resume dir name", name)
    return resume_root(output_dir) / name


def prepare_resume_dir(output_dir: str | Path, name: str) -> Path:
    output_dir = Path(output_dir).expanduser()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Target path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(f"Target directory must be empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    path = resume_dir(output_dir, name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_resume_dir(output_dir: str | Path) -> None:
    root = resume_root(output_dir)
    if not root.exists():
        return

    last_error: OSError | None = None
    for attempt in range(_CLEANUP_RESUME_ATTEMPTS):
        try:
            shutil.rmtree(root)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno not in _CLEANUP_RESUME_RETRY_ERRNOS:
                raise
            last_error = exc
            if attempt + 1 < _CLEANUP_RESUME_ATTEMPTS:
                time.sleep(_CLEANUP_RESUME_RETRY_DELAY_SECONDS)

    try:
        stale = quarantine_resume_dir(output_dir)
    except OSError as exc:
        if last_error is not None:
            raise last_error from exc
        raise
    if stale is not None:
        write_warning(
            "resume",
            "quarantined resume dir after cleanup failure: "
            f"root={root!s} stale={stale!s} error={last_error!r}",
            event="resume_cleanup_quarantined",
            fields={
                "root": root,
                "stale": stale,
                "error": repr(last_error),
            },
        )


def quarantine_resume_dir(output_dir: str | Path) -> Path | None:
    root = resume_root(output_dir)
    if not root.exists():
        return None

    suffix = f"{time.time_ns()}-{os.getpid()}"
    stale = root.with_name(f"{root.name}.stale-{suffix}")
    root.replace(stale)
    return stale


def dataset_sample_count(dataset: Any, *, context: str) -> int:
    try:
        count = len(dataset)
    except TypeError as exc:
        raise TypeError(f"{context} requires a dataset with __len__().") from exc
    if not isinstance(count, int):
        raise TypeError("dataset __len__() must return an integer.")
    if count < 0:
        raise ValueError("dataset length must be non-negative.")
    return count


def validate_completed_indexes(indexes: Iterable[int], expected: int) -> frozenset[int]:
    _validate_expected(expected)
    if isinstance(indexes, frozenset):
        completed = indexes
    else:
        completed = frozenset(_validated_completed_indexes(indexes))
    extra: int | None = None
    for index in completed:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Completed fragment indexes must be integers.")
        if index < 0 or index >= expected:
            extra = index if extra is None else min(extra, index)
    if extra is not None:
        raise ValueError(f"Completed fragment index is outside dataset: {extra}.")
    return completed


class CompactIndexes(Sequence[int]):
    """Sorted immutable indexes backed by a packed signed 64-bit array."""

    def __init__(self, expected: int, indexes: array) -> None:
        _validate_expected(expected)
        if indexes.typecode != "q":
            raise TypeError("compact indexes must use signed 64-bit storage.")
        self.expected = expected
        self._indexes = indexes

    def __len__(self) -> int:
        return len(self._indexes)

    def __iter__(self) -> Iterator[int]:
        return iter(self._indexes)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return tuple(self._indexes[index])
        return int(self._indexes[index])

    def __contains__(self, index: object) -> bool:
        if isinstance(index, bool) or not isinstance(index, int):
            return False
        position = bisect_right(self._indexes, index)
        return position > 0 and self._indexes[position - 1] == index


def indexes_complete(indexes: Collection[int], expected: int) -> bool:
    return len(indexes) == expected


def missing_indexes(completed: Collection[int], expected: int) -> Sequence[int]:
    if isinstance(completed, CompactIndexes):
        _validate_expected(expected)
        if completed.expected != expected:
            raise ValueError("Completed indexes expected sample count does not match.")
        validated: Collection[int] = completed
    else:
        validated = validate_completed_indexes(completed, expected)
    missing_count = expected - len(validated)
    if not validated:
        return range(expected)
    if isinstance(validated, CompactIndexes):
        missing = ComplementIndexes(expected, validated)
        return tuple(missing) if missing_count <= len(validated) else missing
    if missing_count <= len(validated):
        return tuple(index for index in range(expected) if index not in validated)
    return ComplementIndexes(expected, array("q", sorted(validated)))


@dataclass(frozen=True)
class ComplementIndexes(Sequence[int]):
    expected: int
    completed: Sequence[int]

    def __post_init__(self) -> None:
        _validate_expected(self.expected)
        previous: int | None = None
        for index in self.completed:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("completed indexes must be integers.")
            if index < 0 or index >= self.expected:
                raise ValueError("completed index is outside expected range.")
            if previous is not None and index <= previous:
                raise ValueError("completed indexes must be strictly increasing.")
            previous = index

    def __len__(self) -> int:
        return self.expected - len(self.completed)

    def __iter__(self) -> Iterator[int]:
        completed = iter(self.completed)
        current = next(completed, None)
        for index in range(self.expected):
            if index == current:
                current = next(completed, None)
                continue
            yield index

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            positions = range(len(self))[index]
            return tuple(self[position] for position in positions)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("missing index out of range.")

        low = index
        high = index + len(self.completed)
        while low < high:
            candidate = (low + high) // 2
            missing_through_candidate = (
                candidate
                + 1
                - bisect_right(
                    self.completed,
                    candidate,
                )
            )
            if missing_through_candidate > index:
                high = candidate
            else:
                low = candidate + 1
        return low


def pending_batch(
    batch: Iterable[tuple[int, T]],
    completed: Collection[int],
) -> tuple[tuple[int, T], ...]:
    return tuple((index, value) for index, value in batch if index not in completed)


def index_batch_id(indexes: Sequence[int], *, prefix: str = "batch") -> str:
    validate_path_segment("batch id prefix", prefix)
    if not indexes:
        raise ValueError("batch indexes must not be empty.")
    text = ",".join(str(index) for index in indexes)
    digest = hashlib.sha256(text.encode("ascii")).hexdigest()[:16]
    return f"{prefix}-{indexes[0]:012d}-{indexes[-1]:012d}-{digest}"


def format_index_ranges(indexes: Sequence[int], *, limit: int = 8) -> str:
    def format_range(start: int, end: int | None) -> str:
        if end is None or end == start:
            return str(start)
        return f"{start}-{end}"

    ranges: list[str] = []
    position = 0
    while position < len(indexes) and len(ranges) < limit:
        start = indexes[position]
        delta = start - position
        low = position + 1
        high = len(indexes)
        while low < high:
            middle = (low + high) // 2
            if indexes[middle] - middle == delta:
                low = middle + 1
            else:
                high = middle
        end_position = low - 1
        ranges.append(format_range(start, indexes[end_position]))
        position = low
    if position < len(indexes):
        ranges.append("...")
    return ",".join(ranges)


def log_resume_summary(
    source: str,
    *,
    expected: int,
    completed_count: int,
    missing: Sequence[int],
    use_map_style_loader: bool,
) -> None:
    ranges = format_index_ranges(missing)
    write_info(
        source,
        "resume "
        f"expected={expected} completed={completed_count} "
        f"missing={len(missing)} map_style={use_map_style_loader} "
        f"ranges={ranges}",
        event=f"{source}_resume",
        fields={
            "expected": expected,
            "completed": completed_count,
            "missing": len(missing),
            "map_style": use_map_style_loader,
            "ranges": ranges,
        },
    )


def compact_completed_index_entries(
    entries: Iterable[tuple[str, Sequence[int]]],
    *,
    expected: int,
) -> CompactIndexes:
    """Compact authoritative fragment index rows into one validated sequence."""

    _validate_expected(expected)
    arrays: list[array[int]] = []
    for _fragment_id, indexes in entries:
        packed = array("q")
        last_fragment_index: int | None = None
        for value in indexes:
            index = _completed_index(value)
            if index < 0 or index >= expected:
                raise ValueError(
                    f"Completed fragment index is outside dataset: {index}."
                )
            if last_fragment_index is not None and index <= last_fragment_index:
                raise ValueError(
                    "Completed fragment indexes must be strictly increasing."
                )
            packed.append(index)
            last_fragment_index = index
        if packed:
            arrays.append(packed)

    ordered = array("q")
    last_index: int | None = None
    for index in heapq.merge(*arrays):
        if index == last_index:
            raise ValueError(f"Duplicate resume index {index}.")
        ordered.append(index)
        last_index = index
    return CompactIndexes(expected, ordered)


def _completed_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Completed fragment indexes must be integers.")
    return value


def _validate_expected(expected: int) -> None:
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise TypeError("expected sample count must be an integer.")
    if expected < 0:
        raise ValueError("expected sample count must be non-negative.")


def _validated_completed_indexes(indexes: Iterable[int]) -> Iterator[int]:
    for index in indexes:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Completed fragment indexes must be integers.")
        yield index
