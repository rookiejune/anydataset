from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .._io.parquet import (
    ParquetRowWriter,
    parquet_schema,
    pyarrow,
)
from ..types.item import Modality, Role, View
from .manifest import (
    SampleManifestEntry,
    STORE_SCHEMA_VERSION,
    ViewManifestEntry,
    view_from_dict,
)
from .paths import samples_parquet_path, view_manifest_parquet_path


def samples_manifest_exists(root: str | Path) -> bool:
    return samples_parquet_path(root).is_file()


def read_samples_manifest(root: str | Path) -> Iterator[SampleManifestEntry]:
    for row in _read_manifest_rows(
        samples_parquet_path(root),
        _SAMPLE_SCHEMA,
        kind="sample",
    ):
        yield _sample_entry(row)


def sample_manifest_row_count(root: str | Path) -> int:
    return sample_manifest_layout(root)[0]


def sample_manifest_row_groups(root: str | Path) -> tuple[int, ...]:
    return sample_manifest_layout(root)[1]


def sample_manifest_layout(root: str | Path) -> tuple[int, tuple[int, ...]]:
    return _manifest_layout(
        samples_parquet_path(root),
        _SAMPLE_SCHEMA,
        kind="sample",
    )


def read_samples_manifest_row_group(
    root: str | Path,
    row_group: int,
) -> tuple[SampleManifestEntry, ...]:
    return tuple(
        _sample_entry(row)
        for row in _read_manifest_rows(
            samples_parquet_path(root),
            _SAMPLE_SCHEMA,
            kind="sample",
            row_group=row_group,
        )
    )


def read_sample_manifest_index(root: str | Path) -> Iterator[tuple[int, str]]:
    parquet = _validated_parquet(
        samples_parquet_path(root),
        _SAMPLE_SCHEMA,
        kind="sample",
    )
    for batch in parquet.iter_batches(
        batch_size=4096,
        columns=["sample_index", "sample_id"],
    ):
        indexes = batch.column(0)
        sample_ids = batch.column(1)
        for position in range(len(indexes)):
            sample_index = indexes[position].as_py()
            sample_id = sample_ids[position].as_py()
            if sample_index is None or sample_id is None:
                raise ValueError("Sample manifest index columns cannot contain nulls.")
            yield int(sample_index), str(sample_id)


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
    return view_manifest_layout(root, view)[0]


def view_manifest_row_groups(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> tuple[int, ...]:
    return view_manifest_layout(root, view)[1]


def view_manifest_layout(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> tuple[int, tuple[int, ...]]:
    return _manifest_layout(
        view_manifest_parquet_path(root, view),
        _VIEW_SCHEMA,
        kind="view",
    )


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
    yield from _read_manifest_rows(
        path,
        _VIEW_SCHEMA,
        kind="view",
        row_group=row_group,
    )


def _read_view_manifest_indexes(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> Iterator[int]:
    path = view_manifest_parquet_path(root, view)
    parquet = _validated_parquet(path, _VIEW_SCHEMA, kind="view")
    for batch in parquet.iter_batches(batch_size=4096, columns=["sample_index"]):
        indexes = batch.column(0)
        for position in range(len(indexes)):
            sample_index = indexes[position].as_py()
            if sample_index is None:
                raise ValueError("View manifest sample_index cannot contain nulls.")
            yield int(sample_index)


def _manifest_layout(
    path: str | Path,
    fields: tuple[tuple[str, str], ...],
    *,
    kind: str,
) -> tuple[int, tuple[int, ...]]:
    parquet = _validated_parquet(path, fields, kind=kind)
    metadata = parquet.metadata
    return int(metadata.num_rows), tuple(
        int(metadata.row_group(index).num_rows)
        for index in range(metadata.num_row_groups)
    )


def _validated_parquet(
    path: str | Path,
    fields: tuple[tuple[str, str], ...],
    *,
    kind: str,
):
    pa, pq = pyarrow()
    parquet = pq.ParquetFile(path)
    actual = parquet.schema_arrow
    expected = parquet_schema(pa, fields)
    if not actual.equals(expected, check_metadata=False):
        raise ValueError(
            f"Store schema {STORE_SCHEMA_VERSION} {kind} manifest schema "
            "does not match expected fields."
        )
    return parquet


def _read_manifest_rows(
    path: str | Path,
    fields: tuple[tuple[str, str], ...],
    *,
    kind: str,
    row_group: int | None = None,
) -> Iterator[dict[str, Any]]:
    parquet = _validated_parquet(path, fields, kind=kind)
    if row_group is None:
        rows = (
            row
            for batch in parquet.iter_batches(batch_size=4096)
            for row in batch.to_pylist()
        )
    else:
        rows = iter(parquet.read_row_group(row_group).to_pylist())
    for row in rows:
        yield _decode_row(row)


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
