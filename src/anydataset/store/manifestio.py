from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

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


def sample_manifest_writer(root: str | Path) -> _ParquetRowWriter:
    return _manifest_writer(samples_parquet_path(root), _SAMPLE_SCHEMA, _sample_row)


def view_manifest_writer(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> _ParquetRowWriter:
    return _manifest_writer(
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


def _read_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    _, pq = _pyarrow()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            yield _decode_row(row)


class _ParquetRowWriter:
    def __init__(
        self,
        path: Path,
        schema,
        encode: Callable[[Any], dict[str, Any]],
    ) -> None:
        pa, pq = _pyarrow()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pa = pa
        self.pq = pq
        self.path = path
        self.tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        self.schema = _schema(pa, schema)
        self.encode = encode
        self.writer = pq.ParquetWriter(self.tmp, self.schema)
        self.rows: list[dict[str, Any]] = []
        self.closed = False

    def write(self, entry: Any) -> None:
        self.rows.append(self.encode(entry))
        if len(self.rows) >= 4096:
            self._flush()

    def close(self) -> None:
        if self.closed:
            return
        self._flush()
        self.writer.close()
        os.replace(self.tmp, self.path)
        self.closed = True

    def abort(self) -> None:
        if not self.closed:
            self.writer.close()
        if self.tmp.exists():
            self.tmp.unlink()
        self.closed = True

    def _flush(self) -> None:
        table = self.pa.Table.from_pylist(self.rows, schema=self.schema)
        self.writer.write_table(table)
        self.rows.clear()


def _manifest_writer(
    path: Path,
    schema: tuple[tuple[str, str], ...],
    encode: Callable[[Any], dict[str, Any]],
) -> _ParquetRowWriter:
    return _ParquetRowWriter(path, schema, encode)


def _schema(pa, fields: tuple[tuple[str, str], ...]):
    return pa.schema([(name, _field_type(pa, type_name)) for name, type_name in fields])


def _field_type(pa, type_name: str):
    match type_name:
        case "int64":
            return pa.int64()
        case "string":
            return pa.string()
    raise ValueError(f"Unsupported parquet field type: {type_name!r}.")


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Store manifests require pyarrow.") from exc
    return pa, pq


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
