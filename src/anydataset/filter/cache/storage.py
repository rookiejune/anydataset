from __future__ import annotations

import heapq
import json
from array import array
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from threading import Lock
from typing import Any, cast, overload

from ..._io.parquet import read_rows, write_columns
from ..._io.shard import BufferedShardWriter
from ...store.jsonio import read_json, write_json
from ..rules import label_file_id
from ..types import JsonValue, _FilterMetricsRow, _Index, validate_metrics

_METRICS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _ManifestFile:
    path: str
    count: int


@dataclass(frozen=True)
class _Partition:
    label: str
    count: int
    files: tuple[_ManifestFile, ...]


@dataclass(frozen=True)
class PartitionManifest:
    partitions: tuple[_Partition, ...]

    @property
    def count(self) -> int:
        return sum(partition.count for partition in self.partitions)

    @property
    def files(self) -> tuple[_ManifestFile, ...]:
        return tuple(file for partition in self.partitions for file in partition.files)


@dataclass(frozen=True)
class MetricsManifest:
    count: int
    files: tuple[_ManifestFile, ...]


def load_partition_manifest(path: Path) -> PartitionManifest:
    data = _manifest_mapping(read_json(path), "Filter partition manifest")
    _manifest_fields(data, ("partitions",), "Filter partition manifest")
    raw_partitions = _manifest_list(
        data["partitions"],
        "Filter partition manifest partitions",
    )
    partitions: list[_Partition] = []
    labels: set[str] = set()
    file_paths: set[str] = set()
    for index, raw_partition in enumerate(raw_partitions):
        context = f"Filter partition manifest partition {index}"
        partition = _manifest_mapping(raw_partition, context)
        _manifest_fields(partition, ("label", "count", "files"), context)
        label = _manifest_string(partition["label"], f"{context} label")
        if label in labels:
            raise ValueError(f"Duplicate filter partition label: {label!r}.")
        labels.add(label)
        count = _manifest_count(partition["count"], f"{context} count")
        raw_files = _manifest_list(partition["files"], f"{context} files")
        files = tuple(
            _manifest_file(raw_file, f"{context} file {file_index}")
            for file_index, raw_file in enumerate(raw_files)
        )
        for file in files:
            if file.path in file_paths:
                raise ValueError(
                    f"Duplicate filter partition file reference: {file.path!r}."
                )
            file_paths.add(file.path)
        if sum(file.count for file in files) != count:
            raise ValueError(
                "Filter partition manifest count does not match shard counts."
            )
        partitions.append(_Partition(label=label, count=count, files=files))
    return PartitionManifest(tuple(partitions))


def load_metrics_manifest(path: Path) -> MetricsManifest:
    context = "Filter metrics manifest"
    data = _manifest_mapping(read_json(path), context)
    _manifest_fields(data, ("schema_version", "count", "files"), context)
    version = data["schema_version"]
    if type(version) is not int or version != _METRICS_SCHEMA_VERSION:
        raise ValueError("Filter metrics manifest schema_version mismatch.")
    count = _manifest_count(data["count"], f"{context} count")
    raw_files = _manifest_list(data["files"], f"{context} files")
    files = tuple(
        _manifest_file(raw_file, f"{context} file {index}")
        for index, raw_file in enumerate(raw_files)
    )
    file_paths: set[str] = set()
    for file in files:
        if file.path in file_paths:
            raise ValueError(f"Duplicate filter metrics file reference: {file.path!r}.")
        file_paths.add(file.path)
    if sum(file.count for file in files) != count:
        raise ValueError("Filter metrics manifest count does not match shard counts.")
    return MetricsManifest(count=count, files=files)


def _manifest_mapping(
    value: object,
    context: str,
) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")
    return cast(Mapping[object, object], value)


def _manifest_fields(
    data: Mapping[object, object],
    expected: tuple[str, ...],
    context: str,
) -> None:
    for field in expected:
        if field not in data:
            raise ValueError(f"{context} is missing required field {field!r}.")
    for field in data:
        if field not in expected:
            raise ValueError(f"{context} has unsupported field {field!r}.")


def _manifest_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list.")
    return cast(list[object], value)


def _manifest_string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string.")
    return value


def _manifest_relative_path(value: object, context: str) -> str:
    path = _manifest_string(value, context)
    parts = path.replace("\\", "/").split("/")
    if (
        not path
        or Path(path).is_absolute()
        or PureWindowsPath(path).is_absolute()
        or PureWindowsPath(path).drive
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"{context} must be a normalized relative path.")
    return path


def _manifest_count(value: object, context: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer.")
    if value < 0:
        raise ValueError(f"{context} must be non-negative.")
    return value


def _manifest_file(value: object, context: str) -> _ManifestFile:
    data = _manifest_mapping(value, context)
    _manifest_fields(data, ("file", "count"), context)
    return _ManifestFile(
        path=_manifest_relative_path(data["file"], f"{context} path"),
        count=_manifest_count(data["count"], f"{context} count"),
    )


def read_partitions(path: Path) -> dict[str, _Index]:
    path = Path(path)
    try:
        before = _partition_snapshot(path)
        manifest = load_partition_manifest(path / "partitions.json")
        partitions: dict[str, _Index] = {}
        for partition in manifest.partitions:
            partitions[partition.label] = _FileIndex(
                tuple(
                    _IndexFile(
                        path / file.path,
                        file.count,
                        _fingerprint(path / file.path),
                    )
                    for file in partition.files
                ),
                partition.count,
            )
        after = _partition_snapshot(path)
    except FileNotFoundError as exc:
        raise _snapshot_changed(path) from exc
    if before != after:
        raise _snapshot_changed(path)
    return partitions


def metrics_ready(path: Path, *, expected_count: int) -> bool:
    manifest_path = path / "metrics.json"
    if not manifest_path.is_file():
        return False
    manifest = load_metrics_manifest(manifest_path)
    if manifest.count != expected_count:
        return False
    return all((path / file.path).is_file() for file in manifest.files)


def read_metrics(path: Path) -> Iterable[Mapping[str, Any]]:
    manifest = load_metrics_manifest(path / "metrics.json")
    for file in manifest.files:
        for row in read_metric_rows(path / file.path):
            yield row


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
        self._shards = BufferedShardWriter(
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
                "schema_version": _METRICS_SCHEMA_VERSION,
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
    return _MergedIndex(indexes)


@dataclass(frozen=True)
class _FileFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _IndexFile:
    path: Path
    count: int
    fingerprint: _FileFingerprint


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

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

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
            file = self._files[index]
            try:
                before = _fingerprint(file.path)
                if before != file.fingerprint:
                    raise _snapshot_changed(file.path)
                shard = read_index_rows(file.path)
                after = _fingerprint(file.path)
            except FileNotFoundError as exc:
                raise _snapshot_changed(file.path) from exc
            if before != after or after != file.fingerprint:
                raise _snapshot_changed(file.path)
            if len(shard) != file.count:
                raise ValueError(
                    "Filter partition shard row count does not match its manifest: "
                    f"{file.path}."
                )
            self._shards[index] = shard
        return shard


class _MergedIndex(Sequence[int]):
    __slots__ = ("_count", "_error", "_indexes", "_iterator", "_lock", "_values")

    def __init__(self, indexes: Sequence[_Index]) -> None:
        self._indexes = tuple(indexes)
        self._count = sum(len(index) for index in self._indexes)
        self._values = array("q")
        self._iterator: Iterator[int] | None = None
        self._error: Exception | None = None
        self._lock = Lock()

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return tuple(self[position] for position in range(start, stop, step))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("filter index out of range.")
        self._ensure(index + 1)
        return int(self._values[index])

    def __iter__(self) -> Iterator[int]:
        for index in range(len(self)):
            yield self[index]

    def _ensure(self, count: int) -> None:
        if len(self._values) >= count:
            return
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._iterator is None:
                self._iterator = iter(heapq.merge(*self._indexes))
            try:
                while len(self._values) < count:
                    self._values.append(next(self._iterator))
            except StopIteration as exc:
                error = ValueError(
                    "Merged filter partitions ended before their manifest counts."
                )
                self._error = error
                raise error from exc
            except Exception as exc:
                self._error = exc
                raise


def _partition_snapshot(path: Path) -> tuple[_FileFingerprint, _FileFingerprint]:
    return _fingerprint(path / "partitions.json"), _fingerprint(path / ".ready")


def _fingerprint(path: Path) -> _FileFingerprint:
    stat = path.stat()
    return _FileFingerprint(
        device=stat.st_dev,
        inode=stat.st_ino,
        size=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


def _snapshot_changed(path: Path) -> RuntimeError:
    return RuntimeError(f"Filter cache snapshot changed while reading {path}.")


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
        self._shards = BufferedShardWriter(
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
            "metrics": validate_metrics(json.loads(str(row["metrics"]))),
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


def _metrics_json(metrics: Mapping[str, JsonValue]) -> str:
    return json.dumps(
        validate_metrics(metrics),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
