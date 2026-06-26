from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal, TextIO

from .jsonio import read_jsonl
from .manifest import SampleManifestEntry, ViewManifestEntry, ViewRef
from .paths import (
    samples_jsonl_path,
    samples_parquet_path,
    view_manifest_parquet_path,
    view_manifest_path,
)

type ManifestFormat = Literal["jsonl", "parquet"]


def samples_manifest_exists(root: str | Path) -> bool:
    return samples_parquet_path(root).is_file() or samples_jsonl_path(root).is_file()


def read_samples_manifest(root: str | Path) -> Iterator[SampleManifestEntry]:
    parquet_path = samples_parquet_path(root)
    if parquet_path.is_file():
        for row in _read_parquet_rows(parquet_path):
            yield SampleManifestEntry.from_dict(row)
        return
    for row in read_jsonl(samples_jsonl_path(root)):
        yield SampleManifestEntry.from_dict(row)


def write_samples_manifest(
    root: str | Path,
    entries: Iterable[SampleManifestEntry],
    manifest_format: ManifestFormat = "parquet",
) -> None:
    writer = SampleManifestWriter(root, manifest_format)
    try:
        for entry in entries:
            writer.write(entry)
        writer.close()
    except Exception:
        writer.abort()
        raise


def read_view_manifest(
    root: str | Path,
    ref: ViewRef,
    revision: str,
) -> Iterator[ViewManifestEntry]:
    parquet_path = view_manifest_parquet_path(root, ref, revision)
    if parquet_path.is_file():
        for row in _read_parquet_rows(parquet_path):
            yield ViewManifestEntry.from_dict(row)
        return
    for row in read_jsonl(view_manifest_path(root, ref, revision)):
        yield ViewManifestEntry.from_dict(row)


def write_view_manifest(
    root: str | Path,
    ref: ViewRef,
    revision: str,
    entries: Iterable[ViewManifestEntry],
    manifest_format: ManifestFormat = "parquet",
) -> None:
    writer = ViewManifestWriter(root, ref, revision, manifest_format)
    try:
        for entry in entries:
            writer.write(entry)
        writer.close()
    except Exception:
        writer.abort()
        raise


class SampleManifestWriter:
    def __init__(
        self,
        root: str | Path,
        manifest_format: ManifestFormat = "parquet",
    ) -> None:
        _validate_format(manifest_format)
        self.manifest_format = manifest_format
        path = (
            samples_parquet_path(root)
            if manifest_format == "parquet"
            else samples_jsonl_path(root)
        )
        self.rows = _row_writer(path, manifest_format, "sample")

    def write(self, entry: SampleManifestEntry) -> None:
        row = _sample_row(entry) if self.manifest_format == "parquet" else entry.to_dict()
        self.rows.write(row)

    def close(self) -> None:
        self.rows.close()

    def abort(self) -> None:
        self.rows.abort()


class ViewManifestWriter:
    def __init__(
        self,
        root: str | Path,
        ref: ViewRef,
        revision: str,
        manifest_format: ManifestFormat = "parquet",
    ) -> None:
        _validate_format(manifest_format)
        self.manifest_format = manifest_format
        path = (
            view_manifest_parquet_path(root, ref, revision)
            if manifest_format == "parquet"
            else view_manifest_path(root, ref, revision)
        )
        self.rows = _row_writer(path, manifest_format, "view")

    def write(self, entry: ViewManifestEntry) -> None:
        row = _view_row(entry) if self.manifest_format == "parquet" else entry.to_dict()
        self.rows.write(row)

    def close(self) -> None:
        self.rows.close()

    def abort(self) -> None:
        self.rows.abort()


def preflight_manifest_format(manifest_format: ManifestFormat) -> None:
    _validate_format(manifest_format)
    if manifest_format == "parquet":
        _pyarrow()


def _row_writer(path: Path, manifest_format: ManifestFormat, row_type: str):
    if manifest_format == "parquet":
        return _ParquetRowWriter(path, row_type)
    return _JsonlRowWriter(path)


def _validate_format(manifest_format: ManifestFormat) -> None:
    if manifest_format not in {"jsonl", "parquet"}:
        raise ValueError("manifest_format must be 'jsonl' or 'parquet'.")


def _sample_row(entry: SampleManifestEntry) -> dict[str, Any]:
    data = entry.to_dict()
    data["source"] = _json_text(data["source"])
    data["items"] = _json_text(data["items"])
    data["metadata"] = _json_text(data["metadata"])
    return data


def _view_row(entry: ViewManifestEntry) -> dict[str, Any]:
    data = entry.to_dict()
    data["shape"] = _json_text(data["shape"])
    data["provenance"] = _json_text(data["provenance"])
    return data


def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("source", "items", "metadata", "shape", "provenance"):
        value = row.get(key)
        if isinstance(value, str):
            row[key] = json.loads(value)
    return row


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _read_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    _, pq = _pyarrow()
    table = pq.read_table(path)
    for row in table.to_pylist():
        yield _decode_row(row)


class _JsonlRowWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        self.file: TextIO | None = self.tmp.open("w", encoding="utf-8")
        self.closed = False

    def write(self, row: dict[str, Any]) -> None:
        if self.file is None:
            raise ValueError("manifest writer is closed.")
        self.file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    def close(self) -> None:
        if self.closed:
            return
        if self.file is None:
            return
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        self.file = None
        os.replace(self.tmp, self.path)
        self.closed = True

    def abort(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.tmp.exists():
            self.tmp.unlink()
        self.closed = True


class _ParquetRowWriter:
    def __init__(self, path: Path, row_type: str) -> None:
        pa, pq = _pyarrow()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pa = pa
        self.pq = pq
        self.path = path
        self.tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        self.schema = _schema(pa, row_type)
        self.writer = pq.ParquetWriter(self.tmp, self.schema)
        self.rows: list[dict[str, Any]] = []
        self.closed = False

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
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


def _schema(pa, row_type: str):
    if row_type == "sample":
        return pa.schema(
            [
                ("sample_id", pa.string()),
                ("dataset_name", pa.string()),
                ("sample_index", pa.int64()),
                ("source", pa.string()),
                ("items", pa.string()),
                ("metadata", pa.string()),
            ]
        )
    if row_type == "view":
        return pa.schema(
            [
                ("modality", pa.string()),
                ("role", pa.string()),
                ("view_key", pa.string()),
                ("revision", pa.string()),
                ("sample_id", pa.string()),
                ("shard", pa.string()),
                ("key", pa.string()),
                ("shape", pa.string()),
                ("dtype", pa.string()),
                ("checksum", pa.string()),
                ("provenance", pa.string()),
            ]
        )
    raise ValueError(f"Unsupported manifest row type: {row_type!r}.")


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Parquet manifests require pyarrow. Install pyarrow or use manifest_format='jsonl'."
        ) from exc
    return pa, pq
