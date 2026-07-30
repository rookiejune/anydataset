from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
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
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


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
