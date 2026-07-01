from __future__ import annotations

"""Shared helpers for indexed resume workflows.

This module owns hidden resume directories, dataset length checks, index
coverage checks, and deterministic batch ids. It does not read or write filter
partitions, metrics, store parts, or store fragments; callers keep those
domain-specific formats at their own layer.
"""

import hashlib
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


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
    completed = frozenset(indexes)
    extras = sorted(index for index in completed if index < 0 or index >= expected)
    if extras:
        raise ValueError(f"Completed fragment index is outside dataset: {extras[0]}.")
    return completed


def indexes_complete(indexes: frozenset[int], expected: int) -> bool:
    return len(indexes) == expected


def pending_batch(
    batch: Iterable[tuple[int, T]],
    completed: frozenset[int],
) -> tuple[tuple[int, T], ...]:
    return tuple((index, value) for index, value in batch if index not in completed)


def index_batch_id(indexes: Sequence[int], *, prefix: str = "batch") -> str:
    validate_path_segment("batch id prefix", prefix)
    if not indexes:
        raise ValueError("batch indexes must not be empty.")
    text = ",".join(str(index) for index in indexes)
    digest = hashlib.sha256(text.encode("ascii")).hexdigest()[:16]
    return f"{prefix}-{indexes[0]:012d}-{indexes[-1]:012d}-{digest}"


def validate_path_segment(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value in {"", ".", ".."}:
        raise ValueError(f"{name} must be a non-empty path segment.")
    if "/" in value:
        raise ValueError(f"{name} cannot contain '/'.")
