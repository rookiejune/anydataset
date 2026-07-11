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

from .._io.parquet import read_rows, write_columns
from .._io.shard import BufferedShardWriter
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
        for row in read_metric_rows(path / relpath):
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
            if not indices:
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
    __slots__ = ("_path", "_shards")

    def __init__(self, path: Path, *, max_shard_samples: int | None) -> None:
        self._path = path
        self._shards = BufferedShardWriter[_FilterMetricsRow, list[_FilterMetricsRow]](
            path,
            max_shard_items=max_shard_samples,
            new_buffer=list,
            extend=list.extend,
            size=len,
            shard_path=_metric_shard_path,
            write_buffer=write_metric_rows,
        )

    def write_rows(self, rows: Sequence[_FilterMetricsRow]) -> None:
        self._shards.write(rows)

    def close(self) -> None:
        self._shards.close()
        write_json(
            self._path / "metrics.json",
            {
                "schema_version": 1,
                "count": self._shards.count,
                "files": list(self._shards.files),
            },
        )

    def abort(self) -> None:
        self._shards.abort()


def merged_index(indexes: Sequence[_Index]) -> _Index:
    if not indexes:
        return ()
    if len(indexes) == 1:
        return indexes[0]
    output = array("q")
    heads = [0] * len(indexes)
    heap: list[tuple[int, int]] = []
    for label_index, index in enumerate(indexes):
        if index:
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
            start, stop, step = index.indices(len(self))
            return tuple(self._iter_slice(start, stop, step))
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

    def _iter_slice(self, start: int, stop: int, step: int) -> Iterable[int]:
        if step <= 0:
            for index in range(start, stop, step):
                yield self[index]
            return
        if start >= stop:
            return

        shard_index = bisect_right(self._offsets, start) - 1
        position = start
        while position < stop and shard_index < len(self._files):
            shard_offset = self._offsets[shard_index]
            shard_stop = min(stop, self._offsets[shard_index + 1])
            shard = self._shard(shard_index)
            local_start = position - shard_offset
            local_stop = shard_stop - shard_offset
            for value in shard[local_start:local_stop:step]:
                yield int(value)
            position += ((shard_stop - position + step - 1) // step) * step
            shard_index += 1

    def _shard(self, index: int) -> array[int]:
        shard = self._shards[index]
        if shard is None:
            shard = read_index_rows(self._files[index].path)
            self._shards[index] = shard
        return shard


class _PartitionWriteState:
    __slots__ = ("_label", "_shards")

    def __init__(
        self,
        path: Path,
        label: str,
        *,
        max_shard_samples: int | None,
    ) -> None:
        self._label = label
        self._shards = BufferedShardWriter[int, array[int]](
            path,
            max_shard_items=max_shard_samples,
            new_buffer=_index_buffer,
            extend=_extend_index_buffer,
            size=len,
            shard_path=lambda index: _partition_shard_path(label, index),
            write_buffer=write_index_rows,
        )

    def write(self, indices: Sequence[int]) -> None:
        self._shards.write(indices)

    def close(self) -> dict[str, Any]:
        self._shards.close(flush_empty=True)
        return {
            "label": self._label,
            "count": self._shards.count,
            "files": list(self._shards.files),
        }

    def abort(self) -> None:
        self._shards.abort()


def _metric_shard_path(shard_index: int) -> Path:
    return Path("shards") / f"part-{shard_index:06d}.parquet"


def _partition_shard_path(label: str, shard_index: int) -> Path:
    return Path("partitions") / label_file_id(label) / f"part-{shard_index:06d}.parquet"


def _index_buffer() -> array[int]:
    return array("q")


def _extend_index_buffer(buffer: array[int], indices: Sequence[int]) -> None:
    buffer.extend(indices)


def read_index_rows(path: Path) -> array[int]:
    return array("q", (int(row["index"]) for row in read_rows(path, columns=["index"])))


def write_index_rows(path: Path, indices: Sequence[int]) -> None:
    write_columns(
        path,
        {"index": indices},
        (("index", "int64"),),
    )


def read_metric_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    for row in read_rows(path, columns=["index", "label", "metrics"]):
        yield {
            "index": int(row["index"]),
            "label": str(row["label"]),
            "metrics": json.loads(str(row["metrics"])),
        }


def write_metric_rows(path: Path, rows: Sequence[_FilterMetricsRow]) -> None:
    write_columns(
        path,
        {
            "index": (row.index for row in rows),
            "label": (row.label for row in rows),
            "metrics": (_metrics_json(row.metrics) for row in rows),
        },
        (
            ("index", "int64"),
            ("label", "string"),
            ("metrics", "string"),
        ),
    )


def _validate_json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
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
