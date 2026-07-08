"""Small helpers for writing count-bounded shard files.

The helper owns only shard buffering and manifest file bookkeeping. Callers keep
the actual file format and top-level manifest schema.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Generic, TypeVar


ItemT = TypeVar("ItemT")
BufferT = TypeVar("BufferT")


class BufferedShardWriter(Generic[ItemT, BufferT]):
    def __init__(
        self,
        root: Path,
        *,
        max_shard_items: int | None,
        new_buffer: Callable[[], BufferT],
        extend: Callable[[BufferT, Sequence[ItemT]], None],
        size: Callable[[BufferT], int],
        shard_path: Callable[[int], Path],
        write_buffer: Callable[[Path, BufferT], None],
    ) -> None:
        self._root = root
        self._max_shard_items = max_shard_items
        self._new_buffer = new_buffer
        self._extend = extend
        self._size = size
        self._shard_path = shard_path
        self._write_buffer = write_buffer
        self._buffer = new_buffer()
        self._files: list[dict[str, object]] = []
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def files(self) -> tuple[dict[str, object], ...]:
        return tuple(self._files)

    def write(self, items: Sequence[ItemT]) -> None:
        if not items:
            return
        if self._max_shard_items is None:
            self._extend(self._buffer, items)
            return
        offset = 0
        while offset < len(items):
            capacity = self._max_shard_items - self._size(self._buffer)
            next_offset = min(offset + capacity, len(items))
            self._extend(self._buffer, items[offset:next_offset])
            offset = next_offset
            if self._size(self._buffer) >= self._max_shard_items:
                self.flush()

    def close(self, *, flush_empty: bool = False) -> None:
        if self._size(self._buffer) > 0 or (flush_empty and not self._files):
            self.flush()

    def abort(self) -> None:
        self._buffer = self._new_buffer()

    def flush(self) -> None:
        shard_index = len(self._files)
        relpath = self._shard_path(shard_index)
        count = self._size(self._buffer)
        self._write_buffer(self._root / relpath, self._buffer)
        self._files.append(
            {
                "file": relpath.as_posix(),
                "count": count,
            }
        )
        self._count += count
        self._buffer = self._new_buffer()
