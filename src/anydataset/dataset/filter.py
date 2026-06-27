from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import threading
from array import array
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from torch.utils.data import Dataset

from .._sharding import validate_shard
from ..cache import CacheManifest, FileLock
from ..store.jsonio import read_json, write_json
from ..types import Spec
from ..types.item import (
    Modality,
    Role,
    Sample,
    Schema,
)
from .abc import AnyDataset

type FilterLabel = bool | str | Enum
type FilterPredicate = Callable[[Sample], FilterLabel]
type _Index = Sequence[int]

_DEFAULT_MAX_SHARD_SAMPLES = 1_000_000
_DEFAULT_COMMIT_SAMPLES = 100_000
_WORKER_DATASET: AnyDataset | None = None
_WORKER_SCHEMA: Schema | None = None
_WORKER_PREDICATE: FilterPredicate | None = None


class FilterRule:
    __slots__ = ("_name", "_predicate", "_schema", "_schema_identity")

    def __init__(
        self,
        name: str,
        schema: Schema,
        predicate: FilterPredicate,
    ) -> None:
        _validate_string("name", name)
        if not isinstance(schema, Mapping):
            raise TypeError("schema must be a mapping.")
        if not callable(predicate):
            raise TypeError("predicate must be callable.")
        schema_copy = dict(schema)
        schema_identity = _schema_identity(schema_copy)
        _json_text({"schema": schema_identity})
        self._name = name
        self._schema = MappingProxyType(schema_copy)
        self._schema_identity = schema_identity
        self._predicate = predicate

    def __repr__(self) -> str:
        return (
            "FilterRule("
            f"name={self.name!r}, "
            f"schema={self.schema!r}, "
            f"predicate={self.predicate!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FilterRule):
            return NotImplemented
        return (
            self.name,
            self._schema_identity,
            self.predicate,
        ) == (
            other.name,
            other._schema_identity,
            other.predicate,
        )

    def __hash__(self) -> int:
        return hash((self.name, _json_text({"schema": self._schema_identity}), self.predicate))

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> MappingProxyType[tuple[Role, Modality], Any]:
        return self._schema

    @property
    def predicate(self) -> FilterPredicate:
        return self._predicate

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "schema": self._schema_identity,
        }

    @property
    def id(self) -> str:
        return _stable_hash(self.identity)

    def apply(
        self,
        dataset: AnyDataset,
        *,
        num_workers: int = 1,
        commit_samples: int = _DEFAULT_COMMIT_SAMPLES,
        max_shard_samples: int | None = _DEFAULT_MAX_SHARD_SAMPLES,
    ) -> FilterResult:
        return _apply_filter(
            dataset,
            self,
            num_workers=num_workers,
            commit_samples=commit_samples,
            max_shard_samples=max_shard_samples,
        )


class FilterResult:
    __slots__ = ("_base", "_cache_path", "_counts", "_indexes", "_labels", "_rule")

    def __init__(
        self,
        base: AnyDataset,
        partitions: Mapping[str, _Index],
        rule: FilterRule,
        cache_path: Path,
    ) -> None:
        if not isinstance(base, AnyDataset):
            raise TypeError("base must be an AnyDataset.")
        if not isinstance(rule, FilterRule):
            raise TypeError("rule must be a FilterRule.")
        normalized = {label: _index_sequence(indices) for label, indices in partitions.items()}
        self._base = base
        self._indexes = MappingProxyType(normalized)
        self._labels = tuple(normalized)
        self._counts = MappingProxyType(
            {label: len(indices) for label, indices in normalized.items()}
        )
        self._rule = rule
        self._cache_path = Path(cache_path)

    def __repr__(self) -> str:
        return (
            "FilterResult("
            f"base={self.base!r}, "
            f"labels={self.labels!r}, "
            f"rule={self.rule!r}, "
            f"cache_path={self.cache_path!r}"
            ")"
        )

    @property
    def base(self) -> AnyDataset:
        return self._base

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def counts(self) -> MappingProxyType[str, int]:
        return self._counts

    @property
    def partitions(self) -> MappingProxyType[str, tuple[int, ...]]:
        return MappingProxyType(
            {label: tuple(indices) for label, indices in self._indexes.items()}
        )

    @property
    def rule(self) -> FilterRule:
        return self._rule

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    @property
    def spec(self) -> Spec:
        return self.base.spec

    def select(self, *labels: FilterLabel) -> FilteredDataset:
        if not labels:
            raise ValueError("select requires at least one label.")
        normalized = _unique_labels(_label(label) for label in labels)
        indexes = tuple(self._indexes[label] for label in normalized if label in self._indexes)
        return FilteredDataset(
            self.base,
            _merged_index(indexes),
            self.rule,
            self.cache_path,
            labels=normalized,
        )


class FilteredDataset(Dataset):
    __slots__ = ("_base", "_cache_path", "_indices", "_labels", "_rule")

    def __init__(
        self,
        base: AnyDataset,
        indices: Sequence[int],
        rule: FilterRule,
        cache_path: Path,
        *,
        labels: Sequence[str],
    ) -> None:
        if not isinstance(base, AnyDataset):
            raise TypeError("base must be an AnyDataset.")
        if not isinstance(rule, FilterRule):
            raise TypeError("rule must be a FilterRule.")
        self._base = base
        self._indices = indices
        self._labels = tuple(labels)
        self._rule = rule
        self._cache_path = Path(cache_path)

    def __repr__(self) -> str:
        return (
            "FilteredDataset("
            f"base={self.base!r}, "
            f"indices={self.indices!r}, "
            f"labels={self.labels!r}, "
            f"rule={self.rule!r}, "
            f"cache_path={self.cache_path!r}"
            ")"
        )

    @property
    def base(self) -> AnyDataset:
        return self._base

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(self._indices)

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def rule(self) -> FilterRule:
        return self._rule

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    @property
    def spec(self) -> Spec:
        return self.base.spec

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Sample:
        return self.base[self._indices[index]]

    def iter_shard(self, num_shards: int, shard_id: int):
        validate_shard(num_shards, shard_id)
        for index in range(shard_id, len(self), num_shards):
            yield self[index]


def _apply_filter(
    dataset: AnyDataset,
    rule: FilterRule,
    *,
    num_workers: int,
    commit_samples: int,
    max_shard_samples: int | None,
) -> FilterResult:
    if not isinstance(dataset, AnyDataset):
        raise TypeError("dataset must be an AnyDataset.")
    if not isinstance(rule, FilterRule):
        raise TypeError("rule must be a FilterRule.")
    num_workers = _positive_int("num_workers", num_workers)
    commit_samples = _positive_int("commit_samples", commit_samples)
    max_shard_samples = _optional_positive_int(
        "max_shard_samples",
        max_shard_samples,
    )

    cache = dataset.cache_manager.prepare(dataset.spec)
    base_count = len(dataset)
    expected = _metadata(dataset.spec, base_count, rule)
    cache_path = _filter_path(cache, rule)

    if _is_ready(cache_path, expected):
        return FilterResult(dataset, _read_partitions(cache_path), rule, cache_path)

    lock_path = _lock_path(cache, rule)
    with FileLock(lock_path):
        if _is_ready(cache_path, expected):
            return FilterResult(dataset, _read_partitions(cache_path), rule, cache_path)
        _write_cache(
            cache_path,
            expected,
            dataset,
            rule,
            num_workers=num_workers,
            commit_samples=commit_samples,
            max_shard_samples=max_shard_samples,
        )
        return FilterResult(dataset, _read_partitions(cache_path), rule, cache_path)


def _metadata(spec: Spec, base_count: int, rule: FilterRule) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "base": {
            "spec_id": spec.id,
            "sample_count": base_count,
        },
        "rule": dict(rule.identity),
    }


def _filter_path(cache: CacheManifest, rule: FilterRule) -> Path:
    return cache.cache_path / "filters" / rule.id


def _lock_path(cache: CacheManifest, rule: FilterRule) -> Path:
    return cache.cache_path / "filters" / f".{rule.id}.lock"


def _is_ready(path: Path, expected: Mapping[str, Any]) -> bool:
    if not (path / ".ready").is_file():
        return False
    metadata_path = path / "rule.json"
    if not metadata_path.is_file():
        return False
    manifest_path = path / "partitions.json"
    if not manifest_path.is_file():
        return False
    if read_json(metadata_path) != expected:
        return False
    manifest = read_json(manifest_path)
    return all((path / relpath).is_file() for relpath in _partition_files(manifest))


def _collect_range(
    dataset: AnyDataset,
    schema: Schema,
    predicate: FilterPredicate,
    start: int,
    stop: int,
) -> dict[str, array[int]]:
    partitions: dict[str, array[int]] = {}
    for index in range(start, stop):
        sample = AnyDataset.resolve_sample(dataset[index], schema)
        label = _label(predicate(sample))
        if label not in partitions:
            partitions[label] = array("q")
        partitions[label].append(index)
    return partitions


def _write_partitions(
    path: Path,
    dataset: AnyDataset,
    rule: FilterRule,
    *,
    num_workers: int,
    commit_samples: int,
    max_shard_samples: int | None,
) -> None:
    writer = _PartitionWriter(path, max_shard_samples=max_shard_samples)
    try:
        if num_workers == 1 or len(dataset) == 0:
            for partitions in _collect_ranges(
                dataset,
                rule.schema,
                rule.predicate,
                commit_samples,
            ):
                writer.write_partitions(partitions)
        else:
            for partitions in _collect_ranges_parallel(
                dataset,
                rule,
                num_workers,
                commit_samples,
            ):
                writer.write_partitions(partitions)
        writer.close()
    except Exception:
        writer.abort()
        raise


def _collect_ranges(
    dataset: AnyDataset,
    schema: Schema,
    predicate: FilterPredicate,
    commit_samples: int,
) -> Iterable[Mapping[str, Sequence[int]]]:
    for start, stop in _range_chunks(len(dataset), commit_samples):
        yield _collect_range(dataset, schema, predicate, start, stop)


def _collect_ranges_parallel(
    dataset: AnyDataset,
    rule: FilterRule,
    num_workers: int,
    commit_samples: int,
) -> Iterable[Mapping[str, Sequence[int]]]:
    sample_count = len(dataset)
    workers = min(num_workers, sample_count)
    chunk_samples = min(commit_samples, (sample_count + workers - 1) // workers)
    context = _multiprocessing_context()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_filter_worker,
        initargs=(dataset, dict(rule.schema), rule.predicate),
    ) as executor:
        yield from _map_range_chunks(
            executor,
            _range_chunks(sample_count, chunk_samples),
            max_pending=workers * 2,
        )


def _map_range_chunks(
    executor: ProcessPoolExecutor,
    chunks: Iterable[tuple[int, int]],
    *,
    max_pending: int,
) -> Iterable[Mapping[str, Sequence[int]]]:
    chunk_iter = iter(chunks)
    pending = {}
    next_submit = 0
    next_yield = 0

    def submit_next() -> None:
        nonlocal next_submit
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            return
        pending[next_submit] = executor.submit(_collect_worker_range, chunk)
        next_submit += 1

    for _ in range(max_pending):
        submit_next()

    while pending:
        future = pending.pop(next_yield)
        yield future.result()
        next_yield += 1
        submit_next()


def _init_filter_worker(
    dataset: AnyDataset,
    schema: Schema,
    predicate: FilterPredicate,
) -> None:
    global _WORKER_DATASET, _WORKER_SCHEMA, _WORKER_PREDICATE
    _WORKER_DATASET = dataset
    _WORKER_SCHEMA = schema
    _WORKER_PREDICATE = predicate


def _collect_worker_range(bounds: tuple[int, int]) -> dict[str, array[int]]:
    if _WORKER_DATASET is None or _WORKER_SCHEMA is None or _WORKER_PREDICATE is None:
        raise RuntimeError("filter worker was not initialized.")
    start, stop = bounds
    return _collect_range(
        _WORKER_DATASET,
        _WORKER_SCHEMA,
        _WORKER_PREDICATE,
        start,
        stop,
    )


def _range_chunks(sample_count: int, chunk_samples: int) -> Iterable[tuple[int, int]]:
    for start in range(0, sample_count, chunk_samples):
        yield start, min(start + chunk_samples, sample_count)


def _multiprocessing_context():
    if "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


def _write_cache(
    path: Path,
    metadata: Mapping[str, Any],
    dataset: AnyDataset,
    rule: FilterRule,
    *,
    num_workers: int,
    commit_samples: int,
    max_shard_samples: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        write_json(tmp / "rule.json", dict(metadata))
        _write_partitions(
            tmp,
            dataset,
            rule,
            num_workers=num_workers,
            commit_samples=commit_samples,
            max_shard_samples=max_shard_samples,
        )
        (tmp / ".ready").write_text("ready\n", encoding="utf-8")
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise


def _read_partitions(path: Path) -> dict[str, _Index]:
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


class _PartitionWriter:
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
            / _label_file_id(self._label)
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


def _index_sequence(indices: Sequence[int]) -> _Index:
    return array("q", indices)


def _merged_index(indexes: Sequence[_Index]) -> _Index:
    if len(indexes) == 0:
        return ()
    if len(indexes) == 1:
        return array("q", indexes[0])
    output = array("q")
    heads = [0] * len(indexes)
    while True:
        next_index = _next_index(indexes, heads)
        if next_index is None:
            return output
        label_index, value = next_index
        output.append(value)
        heads[label_index] += 1


def _next_index(indexes: Sequence[_Index], heads: Sequence[int]) -> tuple[int, int] | None:
    best_index = None
    best_value = None
    for label_index, index in enumerate(indexes):
        head = heads[label_index]
        if head >= len(index):
            continue
        value = index[head]
        if best_value is None or value < best_value:
            best_index = label_index
            best_value = value
    if best_index is None or best_value is None:
        return None
    return best_index, best_value


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


def _partition_files(manifest: Mapping[str, Any]) -> Iterable[str]:
    for item in manifest["partitions"]:
        files = item["files"]
        if not isinstance(files, list):
            raise TypeError("partition files must be a list.")
        for file in files:
            yield str(file["file"])


def _label(value: FilterLabel) -> str:
    if isinstance(value, bool):
        return "accept" if value else "reject"
    if isinstance(value, Enum):
        enum_value = value.value
        label = enum_value if isinstance(enum_value, str) else value.name
        return _validate_label(str(label))
    if isinstance(value, str):
        return _validate_label(value)
    raise TypeError("filter predicate must return bool, str, or Enum.")


def _validate_label(label: str) -> str:
    if label == "":
        raise ValueError("filter label must not be empty.")
    return label


def _unique_labels(labels: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            output.append(label)
    return tuple(output)


def _schema_identity(schema: Mapping[tuple[Role, Modality], Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for reference, requirement in schema.items():
        role, modality = _reference(reference)
        views = _enum_values(requirement.views, name="views")
        meta = _enum_values(requirement.meta, name="meta")
        entries.append(
            {
                "role": role.value,
                "modality": modality.value,
                "views": views,
                "meta": meta,
            }
        )
    return sorted(entries, key=lambda item: (item["role"], item["modality"]))


def _reference(reference: tuple[Role, Modality]) -> tuple[Role, Modality]:
    if not isinstance(reference, tuple) or len(reference) != 2:
        raise TypeError("schema keys must be (Role, Modality) tuples.")
    role, modality = reference
    if not isinstance(role, Role):
        raise TypeError("schema role must be a Role.")
    if not isinstance(modality, Modality):
        raise TypeError("schema modality must be a Modality.")
    return role, modality


def _enum_values(values: Any, *, name: str) -> list[str]:
    try:
        items = list(values)
    except TypeError as exc:
        raise TypeError(f"schema requirement {name} must be iterable.") from exc
    output: list[str] = []
    for item in items:
        if not isinstance(item, Enum):
            raise TypeError(f"schema requirement {name} must contain enum values.")
        enum_value = item.value
        output.append(enum_value if isinstance(enum_value, str) else item.name)
    return sorted(output)


def _label_file_id(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()[:16]


def _json_text(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError("filter rule identity must be JSON serializable.") from exc


def _validate_string(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value == "":
        raise ValueError(f"{name} must not be empty.")


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _positive_int(name, value)


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Cached filters require pyarrow.") from exc
    return pa, pq
