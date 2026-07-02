from __future__ import annotations

"""Filter cache resume fragments.

This module stores completed filter chunks while a cache is being rebuilt. Each
fragment records scan-space indexes for skipping predicate calls on rerun, plus
the global partition and metrics rows that will be replayed into the final
filter cache.
"""

import os
from array import array
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .._resume import (
    cleanup_resume_dir,
    index_batch_id,
    resume_dir,
    validate_completed_indexes,
)
from .._io.atomic import replace_dir
from ..store.jsonio import read_json, write_json
from .storage import (
    read_index_rows,
    read_metric_rows,
    write_index_rows,
    write_metric_rows,
)
from .types import JsonValue, _FilterChunk, _FilterMetricsRow

_INDEXES_FILE = "indexes.parquet"
_METRICS_FILE = "metrics.parquet"
_RESUME_DIR = "filter"


def prepare_filter_resume_dir(
    cache_path: Path,
    metadata: Mapping[str, object],
    *,
    metrics: bool,
) -> Path:
    path = resume_dir(cache_path, _RESUME_DIR)
    expected = _resume_metadata(metadata, metrics=metrics)
    if path.exists() and _stored_resume_metadata(path) != expected:
        cleanup_resume_dir(cache_path)
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "resume.json", expected)
    return path


def cleanup_filter_resume_dir(cache_path: Path) -> None:
    cleanup_resume_dir(cache_path)


def completed_filter_indexes(path: Path, *, expected: int) -> frozenset[int]:
    indexes: set[int] = set()
    for fragment in filter_fragments(path):
        for index in _fragment_scan_indexes(fragment):
            if index in indexes:
                raise ValueError(f"Duplicate filter resume index {index}.")
            indexes.add(index)
    return validate_completed_indexes(indexes, expected)


def write_filter_fragment(
    path: Path,
    scan_indexes: Sequence[int],
    chunk: _FilterChunk,
) -> None:
    if not scan_indexes:
        raise ValueError("filter resume fragment indexes must not be empty.")
    ordered = tuple(sorted(int(index) for index in scan_indexes))
    if len(set(ordered)) != len(ordered):
        raise ValueError("filter resume fragment indexes must be unique.")
    fragment_id = index_batch_id(ordered)
    replace_dir(
        path / fragment_id,
        lambda tmp: _write_fragment(tmp, fragment_id, ordered, chunk),
    )


def filter_fragments(path: Path) -> tuple[Path, ...]:
    if not path.is_dir():
        return ()
    return tuple(
        sorted(
            (
                child
                for child in path.iterdir()
                if child.is_dir()
                if not child.name.startswith(".")
                if (child / "fragment.json").is_file()
            ),
            key=_fragment_sort_key,
        )
    )


def iter_filter_fragment_chunks(path: Path) -> Iterable[_FilterChunk]:
    for fragment in filter_fragments(path):
        yield _read_fragment_chunk(fragment)


def _write_fragment(
    path: Path,
    fragment_id: str,
    scan_indexes: Sequence[int],
    chunk: _FilterChunk,
) -> None:
    partitions = []
    for label, indexes in chunk.partitions.items():
        relpath = Path("partitions") / f"{len(partitions):06d}.parquet"
        write_index_rows(path / relpath, indexes)
        partitions.append(
            {
                "label": label,
                "file": relpath.as_posix(),
                "count": len(indexes),
            }
        )
    metrics_file = None
    if chunk.metrics:
        write_metric_rows(path / _METRICS_FILE, chunk.metrics)
        metrics_file = _METRICS_FILE
    write_index_rows(path / _INDEXES_FILE, scan_indexes)
    write_json(
        path / "fragment.json",
        {
            "schema_version": 1,
            "pid": os.getpid(),
            "fragment_id": fragment_id,
            "scan_count": len(scan_indexes),
            "scan_indexes": list(scan_indexes),
            "indexes_file": _INDEXES_FILE,
            "partitions": partitions,
            "metrics": {
                "file": metrics_file,
                "count": len(chunk.metrics),
            },
        },
    )


def _resume_metadata(
    metadata: Mapping[str, object],
    *,
    metrics: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "metrics": metrics,
        "cache_metadata": dict(metadata),
    }


def _stored_resume_metadata(path: Path) -> Mapping[str, object] | None:
    metadata_path = path / "resume.json"
    if not metadata_path.is_file():
        return None
    return read_json(metadata_path)


def _read_fragment_chunk(path: Path) -> _FilterChunk:
    data = _read_fragment_manifest(path)
    partitions: dict[str, array[int]] = {}
    for item in _fragment_partitions(data):
        label = str(item["label"])
        indexes = read_index_rows(path / str(item["file"]))
        if int(item["count"]) != len(indexes):
            raise ValueError(f"Filter resume fragment {path} partition count mismatch.")
        partitions[label] = indexes
    metrics = tuple(_fragment_metric_rows(path, data))
    return _FilterChunk(partitions=partitions, metrics=metrics)


def _fragment_metric_rows(
    path: Path,
    data: Mapping[str, object],
) -> Iterable[_FilterMetricsRow]:
    metrics = data.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Filter resume fragment metrics must be a mapping.")
    raw_file = metrics.get("file")
    count = int(metrics.get("count", 0))
    if raw_file is None:
        if count != 0:
            raise ValueError("Filter resume fragment metrics count mismatch.")
        return ()
    rows = tuple(_metric_row(row) for row in read_metric_rows(path / str(raw_file)))
    if len(rows) != count:
        raise ValueError("Filter resume fragment metrics count mismatch.")
    return rows


def _metric_row(row: Mapping[str, object]) -> _FilterMetricsRow:
    return _FilterMetricsRow(
        index=int(row["index"]),
        label=str(row["label"]),
        metrics=_json_mapping(row["metrics"]),
    )


def _json_mapping(value: object) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError("filter resume metrics must be mappings.")
    return value


def _fragment_sort_key(path: Path) -> tuple[int, str]:
    return min(_fragment_scan_indexes(path)), path.name


def _fragment_scan_indexes(path: Path) -> tuple[int, ...]:
    data = _read_fragment_manifest(path)
    raw = data.get("scan_indexes")
    if not isinstance(raw, list):
        raise ValueError("Filter resume fragment scan_indexes must be a list.")
    indexes = tuple(_scan_index(value) for value in raw)
    if data.get("scan_count") != len(indexes):
        raise ValueError("Filter resume fragment scan_count mismatch.")
    stored = tuple(int(index) for index in read_index_rows(path / _INDEXES_FILE))
    if indexes != stored:
        raise ValueError("Filter resume fragment index file mismatch.")
    return indexes


def _read_fragment_manifest(path: Path) -> Mapping[str, object]:
    data = read_json(path / "fragment.json")
    if data.get("schema_version") != 1:
        raise ValueError("Filter resume fragment schema_version mismatch.")
    if data.get("fragment_id") != path.name:
        raise ValueError(f"Filter resume fragment {path} id mismatch.")
    return data


def _fragment_partitions(data: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    raw = data.get("partitions")
    if not isinstance(raw, list):
        raise ValueError("Filter resume fragment partitions must be a list.")
    return tuple(_partition_entry(item) for item in raw)


def _partition_entry(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Filter resume fragment partition must be a mapping.")
    return value


def _scan_index(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("Filter resume fragment scan index must be an integer.")
    return value
