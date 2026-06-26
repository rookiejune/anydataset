from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from .jsonio import read_jsonl, write_jsonl
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
    _validate_format(manifest_format)
    if manifest_format == "parquet":
        _write_parquet_rows(
            samples_parquet_path(root),
            (_sample_row(entry) for entry in entries),
        )
        return
    write_jsonl(samples_jsonl_path(root), (entry.to_dict() for entry in entries))


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
    _validate_format(manifest_format)
    if manifest_format == "parquet":
        _write_parquet_rows(
            view_manifest_parquet_path(root, ref, revision),
            (_view_row(entry) for entry in entries),
        )
        return
    write_jsonl(
        view_manifest_path(root, ref, revision),
        (entry.to_dict() for entry in entries),
    )


def preflight_manifest_format(manifest_format: ManifestFormat) -> None:
    _validate_format(manifest_format)
    if manifest_format == "parquet":
        _pyarrow()


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


def _write_parquet_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    pa, pq = _pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Parquet manifests require pyarrow. Install pyarrow or use manifest_format='jsonl'."
        ) from exc
    return pa, pq
