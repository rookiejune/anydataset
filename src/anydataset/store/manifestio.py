from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
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
    for row in _read_parquet_rows(view_manifest_parquet_path(root, view)):
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
        for row in _read_parquet_row_group(
            view_manifest_parquet_path(root, view),
            row_group,
        )
    )


def read_view_manifest_indexes(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> Iterator[int]:
    yield from read_int_column(
        view_manifest_parquet_path(root, view),
        "sample_index",
    )


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
    data = asdict(entry)
    data["items"] = _json_text(data["items"])
    return data


def _view_row(entry: ViewManifestEntry) -> dict[str, Any]:
    return asdict(entry)


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
