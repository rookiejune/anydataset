"""Immutable filter cache publication and reader lifecycle.

The current pointer selects one complete generation. Shared OS locks pin live
readers, while cleanup requires an exclusive lock before deleting an old
generation.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, TypeVar

from .._io.atomic import replace_dir
from ..cache import FileLock
from ..store.jsonio import read_json, write_json

ValueT = TypeVar("ValueT")

_CURRENT_FILE = "current.json"
_GENERATIONS_DIR = "generations"
_LEASE_FILE = ".lease"
_POINTER_SCHEMA_VERSION = 1
_LOCK_TIMEOUT = 3600.0
_LOCK_POLL = 0.2


class GenerationUnavailable(RuntimeError):
    pass


class GenerationLease:
    __slots__ = ("_file", "_path")

    def __init__(self, path: Path, file: IO[str]) -> None:
        self._path = Path(path)
        self._file: IO[str] | None = file

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        file = self._file
        if file is None:
            return
        self._file = None
        file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True)
class FilterGeneration:
    path: Path
    lease: GenerationLease


def create_filter_generation(
    path: Path,
    write: Callable[[Path], ValueT],
) -> FilterGeneration:
    root = filter_cache_root(path)
    generation = root / _GENERATIONS_DIR / uuid.uuid4().hex

    def write_generation(tmp: Path) -> ValueT:
        result = write(tmp)
        (tmp / _LEASE_FILE).write_text("filter generation lease\n", encoding="utf-8")
        return result

    replace_dir(generation, write_generation)
    write_json(
        root / _CURRENT_FILE,
        {
            "schema_version": _POINTER_SCHEMA_VERSION,
            "generation": generation.name,
        },
    )
    return lease_filter_generation(generation)


def current_filter_generation(path: Path) -> Path:
    root = filter_cache_root(path)
    value = read_json(root / _CURRENT_FILE)
    if not isinstance(value, Mapping):
        raise ValueError("Filter current generation pointer must be a mapping.")
    if value.get("schema_version") != _POINTER_SCHEMA_VERSION:
        raise ValueError("Filter current generation pointer schema_version mismatch.")
    generation = value.get("generation")
    if not isinstance(generation, str) or not _valid_generation_id(generation):
        raise ValueError("Filter current generation id is invalid.")
    return root / _GENERATIONS_DIR / generation


def lease_current_filter_generation(path: Path) -> FilterGeneration:
    root = filter_cache_root(path)
    while True:
        generation = current_filter_generation(root)
        try:
            return lease_filter_generation(generation)
        except GenerationUnavailable:
            try:
                current = current_filter_generation(root)
            except FileNotFoundError:
                raise
            if current == generation:
                raise


def lease_filter_generation(path: Path) -> FilterGeneration:
    generation = Path(path)
    lease_path = generation / _LEASE_FILE
    try:
        file = lease_path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise GenerationUnavailable(
            f"Filter cache generation is no longer available: {generation}."
        ) from exc
    try:
        fcntl.flock(file.fileno(), fcntl.LOCK_SH)
        opened = os.fstat(file.fileno())
        try:
            current = lease_path.stat()
        except FileNotFoundError as exc:
            raise GenerationUnavailable(
                f"Filter cache generation is no longer available: {generation}."
            ) from exc
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise GenerationUnavailable(
                f"Filter cache generation changed while leasing: {generation}."
            )
        if not generation.is_dir() or generation.is_symlink():
            raise GenerationUnavailable(
                f"Filter cache generation is no longer available: {generation}."
            )
    except Exception:
        file.close()
        raise
    lease = GenerationLease(generation, file)
    return FilterGeneration(path=generation, lease=lease)


def cleanup_filter_generations(path: str | Path) -> tuple[Path, ...]:
    """Remove unleased non-current generations and return their paths."""
    root = filter_cache_root(Path(path))
    with FileLock(
        filter_generation_lock_path(root),
        wait_timeout=_LOCK_TIMEOUT,
        poll_interval=_LOCK_POLL,
    ):
        return cleanup_filter_generations_locked(root)


def cleanup_filter_generations_locked(path: Path) -> tuple[Path, ...]:
    root = filter_cache_root(path)
    current = current_filter_generation(root)
    generations = root / _GENERATIONS_DIR
    if not generations.is_dir():
        return ()

    removed = []
    for generation in sorted(generations.iterdir()):
        if generation == current or not _valid_generation_id(generation.name):
            continue
        if _remove_unleased_generation(generation, current=current):
            removed.append(generation)
    return tuple(removed)


def filter_cache_root(path: Path) -> Path:
    path = Path(path)
    if path.parent.name == _GENERATIONS_DIR and _valid_generation_id(path.name):
        return path.parent.parent
    return path


def filter_generation_lock_path(path: Path) -> Path:
    root = filter_cache_root(path)
    return root.with_name(f".{root.name}.lock")


def _remove_unleased_generation(path: Path, *, current: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    lease_path = path / _LEASE_FILE
    try:
        file = lease_path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        opened = os.fstat(file.fileno())
        try:
            stored = lease_path.stat()
        except FileNotFoundError:
            return False
        if (opened.st_dev, opened.st_ino) != (stored.st_dev, stored.st_ino):
            return False
        if path == current:
            return False
        shutil.rmtree(path)
        return True
    finally:
        file.close()


def _valid_generation_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )
