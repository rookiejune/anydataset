from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from .._io.parquet import (
    ParquetRowWriter,
    ParquetSchema,
    parquet_schema,
    pyarrow,
    read_int_column,
    read_int_string_columns,
    read_row_group,
    read_rows,
    row_count,
    row_groups,
)
from ..types.item import Modality, Role, View
from .manifest import (
    SampleManifestEntry,
    ViewManifestEntry,
    view_from_dict,
)
from .paths import samples_parquet_path, view_manifest_parquet_path


def samples_manifest_exists(root: str | Path) -> bool:
    return samples_parquet_path(root).is_file()


def read_samples_manifest(root: str | Path) -> Iterator[SampleManifestEntry]:
    for row in _read_parquet_rows(samples_parquet_path(root)):
        yield _sample_entry(row)


def sample_manifest_row_count(root: str | Path) -> int:
    return row_count(samples_parquet_path(root))


def sample_manifest_row_groups(root: str | Path) -> tuple[int, ...]:
    return row_groups(samples_parquet_path(root))


def read_samples_manifest_row_group(
    root: str | Path,
    row_group: int,
) -> tuple[SampleManifestEntry, ...]:
    return tuple(
        _sample_entry(row)
        for row in _read_parquet_row_group(samples_parquet_path(root), row_group)
    )


def read_sample_manifest_index(root: str | Path) -> Iterator[tuple[int, str]]:
    yield from read_int_string_columns(
        samples_parquet_path(root),
        int_column="sample_index",
        string_column="sample_id",
    )


def write_samples_manifest(
    root: str | Path,
    entries: Iterable[SampleManifestEntry],
) -> None:
    writer = sample_manifest_writer(root)
    try:
        for entry in entries:
            writer.write(entry)
        writer.close()
    except Exception:
        writer.abort()
        raise


def read_view_manifest(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> Iterator[ViewManifestEntry]:
    for row in _read_view_manifest_rows(root, view):
        yield _view_entry(row)


def view_manifest_row_count(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> int:
    return row_count(view_manifest_parquet_path(root, view))


def view_manifest_row_groups(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> tuple[int, ...]:
    return row_groups(view_manifest_parquet_path(root, view))


def read_view_manifest_row_group(
    root: str | Path,
    view: tuple[Role, Modality, View],
    row_group: int,
) -> tuple[ViewManifestEntry, ...]:
    return tuple(
        _view_entry(row)
        for row in _read_view_manifest_rows(root, view, row_group=row_group)
    )


def read_view_manifest_indexes(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> Iterator[int]:
    yield from _read_view_manifest_indexes(root, view)


def write_view_manifest(
    root: str | Path,
    view: tuple[Role, Modality, View],
    entries: Iterable[ViewManifestEntry],
) -> None:
    writer = view_manifest_writer(root, view)
    try:
        for entry in entries:
            writer.write(entry)
        writer.close()
    except Exception:
        writer.abort()
        raise


def sample_manifest_writer(root: str | Path) -> ParquetRowWriter:
    return ParquetRowWriter(samples_parquet_path(root), _SAMPLE_SCHEMA, _sample_row)


def view_manifest_writer(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> ParquetRowWriter:
    return ParquetRowWriter(
        view_manifest_parquet_path(root, view),
        _VIEW_SCHEMA,
        _view_row,
    )


def _sample_row(entry: SampleManifestEntry) -> dict[str, Any]:
    return {
        "sample_id": entry.sample_id,
        "sample_index": entry.sample_index,
        "items": _json_text(
            tuple(
                (
                    (role.value, modality.value),
                    dict(meta),
                )
                for (role, modality), meta in entry.items
            )
        ),
    }


def _view_row(entry: ViewManifestEntry) -> dict[str, Any]:
    return {
        "role": entry.role.value,
        "modality": entry.modality.value,
        "view": entry.view.value,
        "sample_index": entry.sample_index,
        "shard": entry.shard,
        "key": entry.key,
    }


def _sample_entry(row: dict[str, Any]) -> SampleManifestEntry:
    return SampleManifestEntry(
        **{
            **row,
            "items": tuple(
                (
                    (Role(item[0][0]), Modality(item[0][1])),
                    item[1],
                )
                for item in row["items"]
            ),
        }
    )


def _view_entry(row: dict[str, Any]) -> ViewManifestEntry:
    role, modality, view = view_from_dict(row)
    return ViewManifestEntry(
        **{
            **row,
            "role": role,
            "modality": modality,
            "view": view,
        }
    )


def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("items",):
        value = row.get(key)
        if isinstance(value, str):
            row[key] = json.loads(value)
    return row


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _read_view_manifest_rows(
    root: str | Path,
    view: tuple[Role, Modality, View],
    *,
    row_group: int | None = None,
) -> Iterator[dict[str, Any]]:
    path = view_manifest_parquet_path(root, view)
    sample_index_by_id = None
    if not _has_column(path, "sample_index"):
        sample_index_by_id = _sample_index_by_id(root)
    rows = (
        _read_parquet_rows(path)
        if row_group is None
        else _read_parquet_row_group(path, row_group)
    )
    for row in rows:
        yield _view_manifest_row(row, sample_index_by_id)


def _read_view_manifest_indexes(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> Iterator[int]:
    path = view_manifest_parquet_path(root, view)
    if _has_column(path, "sample_index"):
        yield from read_int_column(path, "sample_index")
        return
    sample_index_by_id = _sample_index_by_id(root)
    for row in _read_parquet_rows(path, columns=["sample_id"]):
        yield _view_manifest_row(row, sample_index_by_id)["sample_index"]


def _view_manifest_row(
    row: dict[str, Any],
    sample_index_by_id: dict[str, int] | None,
) -> dict[str, Any]:
    row = dict(row)
    sample_id = row.pop("sample_id", None)
    if "sample_index" not in row:
        if sample_index_by_id is None or sample_id is None:
            raise ValueError("legacy view manifest entries must contain sample_id.")
        row["sample_index"] = sample_index_by_id[str(sample_id)]
    return row


def _sample_index_by_id(root: str | Path) -> dict[str, int]:
    return {
        sample_id: sample_index
        for sample_index, sample_id in read_sample_manifest_index(root)
    }


def _has_column(path: str | Path, name: str) -> bool:
    _, pq = pyarrow()
    return name in pq.ParquetFile(path).schema_arrow.names


def _read_parquet_rows(
    path: Path,
    *,
    columns: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    for row in read_rows(path, columns=columns):
        yield _decode_row(row)


def _read_parquet_row_group(
    path: Path,
    row_group: int,
    *,
    columns: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    for row in read_row_group(path, row_group, columns=columns):
        yield _decode_row(row)


def _schema(pa, fields: tuple[tuple[str, str], ...]):
    return parquet_schema(pa, fields)


_SAMPLE_SCHEMA = (
    ("sample_id", "string"),
    ("sample_index", "int64"),
    ("items", "string"),
)
_VIEW_SCHEMA = (
    ("modality", "string"),
    ("role", "string"),
    ("view", "string"),
    ("sample_index", "int64"),
    ("shard", "string"),
    ("key", "string"),
)
