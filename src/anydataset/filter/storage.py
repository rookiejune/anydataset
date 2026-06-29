from __future__ import annotations

import heapq
import json
import math
from array import array
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..store.jsonio import read_json, write_json
from .rules import label_file_id
from .types import JsonValue, _FilterMetricsRow, _Index


def read_partitions(path: Path) -> dict[str, _Index]:
    manifest = read_json(path / "partitions.json")
    return {
        str(item["label"]): _FileIndex(
            tuple(
                _IndexFile(path / file["file"], int(file["count"]))
                for file in item["files"]
            ),
            int(item["count"]),
        )
        for item in manifest["partitions"]
    }


def partition_files(manifest: Mapping[str, Any]) -> Iterable[str]:
    for item in manifest["partitions"]:
        files = item["files"]
        if not isinstance(files, list):
            raise TypeError("partition files must be a list.")
        for file in files:
            yield str(file["file"])


def metrics_ready(path: Path) -> bool:
    manifest_path = path / "metrics.json"
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        return False
    files = list(metrics_files(manifest))
    file_count = sum(int(file["count"]) for file in manifest["files"])
    if file_count != int(manifest["count"]):
        return False
    return all((path / relpath).is_file() for relpath in files)


def read_metrics(path: Path) -> Iterable[Mapping[str, Any]]:
    manifest = read_json(path / "metrics.json")
    for relpath in metrics_files(manifest):
        for row in _read_metric_rows(path / relpath):
            yield row


def metrics_files(manifest: Mapping[str, Any]) -> Iterable[str]:
    files = manifest["files"]
    if not isinstance(files, list):
        raise TypeError("metrics files must be a list.")
    for file in files:
        yield str(file["file"])


class PartitionWriter:
    __slots__ = ("_entries", "_max_shard_samples", "_path", "_states")

    def __init__(self, path: Path, *, max_shard_samples: int | None) -> None:
        self._path = path
        self._max_shard_samples = max_shard_samples
        self._states: dict[str, _PartitionWriteState] = {}
        self._entries: list[dict[str, Any]] = []

    def write_partitions(self, partitions: Mapping[str, Sequence[int]]) -> None:
        for label, indices in partitions.items():
            if len(indices) == 0:
                continue
            state = self._state(label)
            state.write(indices)

    def close(self) -> None:
        self._entries = [state.close() for state in self._states.values()]
        write_json(
            self._path / "partitions.json",
            {
                "partitions": self._entries,
            },
        )

    def abort(self) -> None:
        for state in self._states.values():
            state.abort()

    def _state(self, label: str) -> _PartitionWriteState:
        state = self._states.get(label)
        if state is None:
            state = _PartitionWriteState(
                self._path,
                label,
                max_shard_samples=self._max_shard_samples,
            )
            self._states[label] = state
        return state


class MetricsWriter:
    __slots__ = ("_buffer", "_count", "_files", "_max_shard_samples", "_path")

    def __init__(self, path: Path, *, max_shard_samples: int | None) -> None:
        self._path = path
        self._max_shard_samples = max_shard_samples
        self._buffer: list[_FilterMetricsRow] = []
        self._files: list[dict[str, Any]] = []
        self._count = 0

    def write_rows(self, rows: Sequence[_FilterMetricsRow]) -> None:
        if len(rows) == 0:
            return
        if self._max_shard_samples is None:
            self._buffer.extend(rows)
            return
        offset = 0
        while offset < len(rows):
            capacity = self._max_shard_samples - len(self._buffer)
            next_offset = min(offset + capacity, len(rows))
            self._buffer.extend(rows[offset:next_offset])
            offset = next_offset
            if len(self._buffer) >= self._max_shard_samples:
                self._flush()

    def close(self) -> None:
        if self._buffer:
            self._flush()
        write_json(
            self._path / "metrics.json",
            {
                "schema_version": 1,
                "count": self._count,
                "files": self._files,
            },
        )

    def abort(self) -> None:
        self._buffer.clear()

    def _flush(self) -> None:
        shard_index = len(self._files)
        relpath = Path("shards") / f"part-{shard_index:06d}.parquet"
        count = len(self._buffer)
        _write_metrics(self._path / relpath, self._buffer)
        self._files.append(
            {
                "file": relpath.as_posix(),
                "count": count,
            }
        )
        self._count += count
        self._buffer = []


def merged_index(indexes: Sequence[_Index]) -> _Index:
    if len(indexes) == 0:
        return ()
    if len(indexes) == 1:
        return indexes[0]
    output = array("q")
    heads = [0] * len(indexes)
    heap: list[tuple[int, int]] = []
    for label_index, index in enumerate(indexes):
        if len(index) > 0:
            heap.append((int(index[0]), label_index))
    heapq.heapify(heap)
    while heap:
        value, label_index = heapq.heappop(heap)
        output.append(value)
        heads[label_index] += 1
        head = heads[label_index]
        index = indexes[label_index]
        if head >= len(index):
            continue
        heapq.heappush(heap, (int(index[head]), label_index))
    return output


def validate_metrics(metrics: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    if not isinstance(metrics, Mapping):
        raise TypeError("filter decision metrics must be a mapping.")
    output: dict[str, JsonValue] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError("filter decision metrics keys must be strings.")
        output[key] = _validate_json_value(value)
    return output


@dataclass(frozen=True)
class _IndexFile:
    path: Path
    count: int


class _FileIndex(Sequence[int]):
    __slots__ = ("_count", "_files", "_offsets", "_shards")

    def __init__(self, files: Sequence[_IndexFile], count: int) -> None:
        self._files = tuple(files)
        self._count = count
        offsets = [0]
        for file in self._files:
            offsets.append(offsets[-1] + file.count)
        if offsets[-1] != count:
            raise ValueError("partition manifest count does not match shard counts.")
        self._offsets = tuple(offsets)
        self._shards: list[array[int] | None] = [None] * len(self._files)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("filter index out of range.")
        shard_index = bisect_right(self._offsets, index) - 1
        shard = self._shard(shard_index)
        return int(shard[index - self._offsets[shard_index]])

    def __iter__(self):
        for shard_index in range(len(self._files)):
            for index in self._shard(shard_index):
                yield int(index)

    def _shard(self, index: int) -> array[int]:
        shard = self._shards[index]
        if shard is None:
            shard = _read_indices(self._files[index].path)
            self._shards[index] = shard
        return shard


class _PartitionWriteState:
    __slots__ = ("_buffer", "_count", "_files", "_label", "_max_shard_samples", "_path")

    def __init__(
        self,
        path: Path,
        label: str,
        *,
        max_shard_samples: int | None,
    ) -> None:
        self._path = path
        self._label = label
        self._max_shard_samples = max_shard_samples
        self._buffer = array("q")
        self._files: list[dict[str, Any]] = []
        self._count = 0

    def write(self, indices: Sequence[int]) -> None:
        if self._max_shard_samples is None:
            self._buffer.extend(indices)
            return
        offset = 0
        while offset < len(indices):
            capacity = self._max_shard_samples - len(self._buffer)
            next_offset = min(offset + capacity, len(indices))
            self._buffer.extend(indices[offset:next_offset])
            offset = next_offset
            if len(self._buffer) >= self._max_shard_samples:
                self._flush()

    def close(self) -> dict[str, Any]:
        if self._buffer or not self._files:
            self._flush()
        return {
            "label": self._label,
            "count": self._count,
            "files": self._files,
        }

    def abort(self) -> None:
        self._buffer.clear()

    def _flush(self) -> None:
        shard_index = len(self._files)
        relpath = (
            Path("partitions")
            / label_file_id(self._label)
            / f"part-{shard_index:06d}.parquet"
        )
        _write_indices(self._path / relpath, self._buffer)
        count = len(self._buffer)
        self._files.append(
            {
                "file": relpath.as_posix(),
                "count": count,
            }
        )
        self._count += count
        self._buffer = array("q")


def _read_indices(path: Path) -> array[int]:
    _, pq = _pyarrow()
    table = pq.read_table(path, columns=["index"])
    return array("q", (int(index) for index in table.column("index").to_pylist()))


def _write_indices(path: Path, indices: Sequence[int]) -> None:
    pa, pq = _pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [pa.array(indices, type=pa.int64())],
        schema=pa.schema([("index", pa.int64())]),
    )
    pq.write_table(table, path)


def _read_metric_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    _, pq = _pyarrow()
    table = pq.read_table(path, columns=["index", "label", "metrics"])
    indices = table.column("index").to_pylist()
    labels = table.column("label").to_pylist()
    metrics = table.column("metrics").to_pylist()
    for index, label, payload in zip(indices, labels, metrics, strict=True):
        yield {
            "index": int(index),
            "label": str(label),
            "metrics": json.loads(str(payload)),
        }


def _write_metrics(path: Path, rows: Sequence[_FilterMetricsRow]) -> None:
    pa, pq = _pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [
            pa.array((row.index for row in rows), type=pa.int64()),
            pa.array((row.label for row in rows), type=pa.string()),
            pa.array((_metrics_json(row.metrics) for row in rows), type=pa.string()),
        ],
        schema=pa.schema(
            [
                ("index", pa.int64()),
                ("label", pa.string()),
                ("metrics", pa.string()),
            ]
        ),
    )
    pq.write_table(table, path)


def _validate_json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("filter decision metrics must not contain NaN or infinity.")
        return value
    if isinstance(value, list):
        return [_validate_json_value(item) for item in value]
    if isinstance(value, Mapping):
        output: dict[str, JsonValue] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("filter decision metrics keys must be strings.")
            output[key] = _validate_json_value(child)
        return output
    raise TypeError("filter decision metrics must be JSON-serializable.")


def _metrics_json(metrics: Mapping[str, JsonValue]) -> str:
    try:
        return json.dumps(
            metrics,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("filter decision metrics must be JSON-serializable.") from exc


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Cached filters require pyarrow.") from exc
    return pa, pq
