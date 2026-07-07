from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


ParquetSchema = tuple[tuple[str, str], ...]


class ParquetRowWriter:
    def __init__(
        self,
        path: str | Path,
        schema: ParquetSchema,
        encode: Callable[[Any], dict[str, Any]],
    ) -> None:
        pa, pq = pyarrow()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pa = pa
        self.pq = pq
        self.path = path
        self.tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        self.schema = parquet_schema(pa, schema)
        self.encode = encode
        self.writer = pq.ParquetWriter(self.tmp, self.schema)
        self.rows: list[dict[str, Any]] = []
        self._wrote_rows = False
        self.closed = False

    def write(self, entry: Any) -> None:
        self.rows.append(self.encode(entry))
        if len(self.rows) >= 4096:
            self._flush()

    def close(self) -> None:
        if self.closed:
            return
        if self.rows or not self._wrote_rows:
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
        if self.rows:
            self._wrote_rows = True
        self.rows.clear()


def read_rows(
    path: str | Path,
    *,
    columns: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    _, pq = pyarrow()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


def read_row_group(
    path: str | Path,
    row_group: int,
    *,
    columns: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    _, pq = pyarrow()
    table = pq.ParquetFile(path).read_row_group(row_group, columns=columns)
    yield from table.to_pylist()


def read_int_column(path: str | Path, column: str) -> Iterator[int]:
    _, pq = pyarrow()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096, columns=[column]):
        values = batch.column(0)
        for index in range(len(values)):
            yield int(values[index].as_py())


def read_int_string_columns(
    path: str | Path,
    *,
    int_column: str,
    string_column: str,
) -> Iterator[tuple[int, str]]:
    _, pq = pyarrow()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=4096,
        columns=[int_column, string_column],
    ):
        int_values = batch.column(0)
        string_values = batch.column(1)
        for index in range(len(int_values)):
            yield int(int_values[index].as_py()), str(string_values[index].as_py())


def row_count(path: str | Path) -> int:
    _, pq = pyarrow()
    return int(pq.ParquetFile(path).metadata.num_rows)


def row_groups(path: str | Path) -> tuple[int, ...]:
    _, pq = pyarrow()
    metadata = pq.ParquetFile(path).metadata
    return tuple(
        int(metadata.row_group(index).num_rows)
        for index in range(metadata.num_row_groups)
    )


def write_rows(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
    schema: ParquetSchema,
) -> None:
    pa, pq = pyarrow()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=parquet_schema(pa, schema))
    pq.write_table(table, path)


def write_columns(
    path: str | Path,
    columns: Mapping[str, Iterable[Any]],
    schema: ParquetSchema,
) -> None:
    pa, pq = pyarrow()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = parquet_schema(pa, schema)
    arrays = [
        pa.array(columns[name], type=field_type(pa, type_name))
        for name, type_name in schema
    ]
    table = pa.Table.from_arrays(arrays, schema=fields)
    pq.write_table(table, path)


def parquet_schema(pa, fields: ParquetSchema):
    return pa.schema([(name, field_type(pa, type_name)) for name, type_name in fields])


def field_type(pa, type_name: str):
    if type_name == "int64":
        return pa.int64()
    if type_name == "string":
        return pa.string()
    raise ValueError(f"Unsupported parquet field type: {type_name!r}.")


def pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("anydataset parquet files require pyarrow.") from exc
    return pa, pq
