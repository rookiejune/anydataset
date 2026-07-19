"""Shared helpers for indexed resume workflows.

This module owns hidden resume directories, dataset length checks, index
coverage checks, and deterministic batch ids. It does not read or write filter
partitions, metrics, store parts, or store fragments; callers keep those
domain-specific formats at their own layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from bisect import bisect_right
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TypeVar

from ._logging import write_info

T = TypeVar("T")

_COMPLETED_INDEX_CACHE = ".completed-indexes.jsonl"


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
    if root.exists():
        shutil.rmtree(root)


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


def indexes_complete(indexes: frozenset[int], expected: int) -> bool:
    return len(indexes) == expected


def missing_indexes(completed: frozenset[int], expected: int) -> Sequence[int]:
    validate_completed_indexes(completed, expected)
    missing_count = expected - len(completed)
    if not completed:
        return range(expected)
    if missing_count <= len(completed):
        return tuple(index for index in range(expected) if index not in completed)
    return ComplementIndexes(expected, tuple(sorted(completed)))


@dataclass(frozen=True)
class ComplementIndexes(Sequence[int]):
    expected: int
    completed: tuple[int, ...]

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
            missing_through_candidate = candidate + 1 - bisect_right(
                self.completed,
                candidate,
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
    write_info(
        source,
        "resume "
        f"expected={expected} completed={completed_count} "
        f"missing={len(missing)} map_style={use_map_style_loader} "
        f"ranges={format_index_ranges(missing)}",
    )


def cached_completed_indexes(
    root: str | Path,
    fragment_ids: Iterable[str],
) -> frozenset[int] | None:
    path = Path(root) / _COMPLETED_INDEX_CACHE
    if not path.is_file():
        return None
    expected = frozenset(fragment_ids)
    entries = _read_completed_index_entries(path)
    if frozenset(entries) != expected:
        return None
    indexes: set[int] = set()
    for fragment_indexes in entries.values():
        for index in fragment_indexes:
            if index in indexes:
                raise ValueError(f"Duplicate resume index {index}.")
            indexes.add(index)
    return frozenset(indexes)


def write_completed_index_cache(
    root: str | Path,
    entries: Iterable[tuple[str, Sequence[int]]],
) -> None:
    path = Path(root) / _COMPLETED_INDEX_CACHE
    rows = tuple(
        _completed_index_row(fragment_id, indexes) for fragment_id, indexes in entries
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with tmp.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def append_completed_index_cache(
    root: str | Path,
    fragment_id: str,
    indexes: Sequence[int],
) -> None:
    path = Path(root) / _COMPLETED_INDEX_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        json.dumps(
            _completed_index_row(fragment_id, indexes),
            separators=(",", ":"),
        )
        + "\n"
    )
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def validate_path_segment(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value in {"", ".", ".."}:
        raise ValueError(f"{name} must be a non-empty path segment.")
    if "/" in value:
        raise ValueError(f"{name} cannot contain '/'.")


def _completed_index_row(fragment_id: str, indexes: Sequence[int]) -> dict[str, object]:
    validate_path_segment("fragment id", fragment_id)
    ordered = tuple(int(index) for index in indexes)
    if len(set(ordered)) != len(ordered):
        raise ValueError("completed indexes must be unique.")
    return {
        "schema_version": 1,
        "fragment_id": fragment_id,
        "indexes": list(ordered),
    }


def _read_completed_index_entries(path: Path) -> dict[str, tuple[int, ...]]:
    entries: dict[str, tuple[int, ...]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("schema_version") != 1:
                raise ValueError("Completed index cache schema_version mismatch.")
            fragment_id = str(data["fragment_id"])
            raw = data.get("indexes")
            if not isinstance(raw, list):
                raise ValueError("Completed index cache indexes must be a list.")
            indexes = tuple(_completed_index(value) for value in raw)
            previous = entries.get(fragment_id)
            if previous is not None and previous != indexes:
                raise ValueError(
                    f"Completed index cache has duplicate fragment {fragment_id}."
                )
            entries[fragment_id] = indexes
    return entries


def _completed_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Completed index cache entries must be integers.")
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
