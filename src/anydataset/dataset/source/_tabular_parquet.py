"""Shared delimited-text → Parquet prepare and map-style row-group reads."""

from __future__ import annotations

import csv
import hashlib
import multiprocessing
import os
import sys
import tempfile
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.csv import ConvertOptions  # pyright: ignore[reportPrivateImportUsage]
from pyarrow.csv import ParseOptions  # pyright: ignore[reportPrivateImportUsage]
from pyarrow.csv import ReadOptions  # pyright: ignore[reportPrivateImportUsage]
from pyarrow.csv import read_csv  # pyright: ignore[reportPrivateImportUsage]

from ..._compat import strict_zip
from ..._runtime.parallel import multiprocessing_context
from ..._runtime.sharding import validate_shard
from ..._validation import validate_path_segment
from ...cache import FileLock
from ...store.jsonio import read_json, write_json

JsonMapping = Mapping[str, Any]
TabularRow = Mapping[str, str]

CACHE_SCHEMA_VERSION = 1
PARQUET_ROW_GROUP_SIZE = 4096
MAX_CACHED_ROW_GROUPS = 2
MAX_OPEN_PARQUET_FILES = 8
PREPARE_LOCK_TIMEOUT = 3600.0
PREPARE_LOCK_POLL = 0.2


@dataclass(frozen=True)
class ParquetPart:
    path: Path
    start: int
    stop: int
    row_groups: tuple[int, ...]
    row_group_stops: tuple[int, ...]


def validate_prepare_workers(value: object) -> None:
    if type(value) is not int:
        raise TypeError("prepare_workers must be an integer or None.")
    if value < 0:
        raise ValueError("prepare_workers must be non-negative.")


def stops(counts: Sequence[int]) -> tuple[int, ...]:
    total = 0
    result = []
    for count in counts:
        total += count
        result.append(total)
    return tuple(result)


def ensure_parts(
    sources: Sequence[Path],
    *,
    cache_path: Path,
    manifest_name: str,
    cache_dir_name: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    prepare_workers: int | None = None,
    progress_label: str = "prepare tabular",
    source_label: str = "tabular",
) -> tuple[ParquetPart, ...]:
    if prepare_workers is not None:
        validate_prepare_workers(prepare_workers)
    records = ensure_records(
        sources,
        cache_path=cache_path,
        manifest_name=manifest_name,
        cache_dir_name=cache_dir_name,
        delimiter=delimiter,
        encoding=encoding,
        prepare_workers=prepare_workers,
        progress_label=progress_label,
        source_label=source_label,
    )
    cache_dir = cache_path / cache_dir_name
    start = 0
    parts = []
    for record in records:
        count = int(record["row_count"])
        stop = start + count
        row_groups = tuple(int(value) for value in record["row_groups"])
        parts.append(
            ParquetPart(
                path=cache_dir / str(record["part"]),
                start=start,
                stop=stop,
                row_groups=row_groups,
                row_group_stops=stops(row_groups),
            )
        )
        start = stop
    return tuple(parts)


def ensure_records(
    sources: Sequence[Path],
    *,
    cache_path: Path,
    manifest_name: str,
    cache_dir_name: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    prepare_workers: int | None = None,
    progress_label: str = "prepare tabular",
    source_label: str = "tabular",
) -> tuple[JsonMapping, ...]:
    lock_path = cache_path / ".prepare.lock"
    manifest_path = cache_path / manifest_name
    cache_dir = cache_path / cache_dir_name
    cached = read_cache(
        sources,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        source_label=source_label,
    )
    if cached is not None:
        return cached
    with FileLock(
        lock_path,
        wait_timeout=PREPARE_LOCK_TIMEOUT,
        poll_interval=PREPARE_LOCK_POLL,
    ):
        cached = read_cache(
            sources,
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            source_label=source_label,
        )
        if cached is not None:
            return cached
        return build_cache(
            sources,
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            delimiter=delimiter,
            encoding=encoding,
            prepare_workers=prepare_workers,
            progress_label=progress_label,
            source_label=source_label,
        )


def read_cache(
    sources: Sequence[Path],
    *,
    manifest_path: Path,
    cache_dir: Path,
    source_label: str,
) -> tuple[JsonMapping, ...] | None:
    if not manifest_path.is_file():
        return None
    data = read_json(manifest_path)
    if not isinstance(data, Mapping):
        raise ValueError(f"Invalid {source_label} parquet manifest: {manifest_path}")
    if not valid_cache_schema(data):
        return None
    records = data.get("files")
    if not isinstance(records, list) or len(records) != len(sources):
        return None
    validated = []
    for source, record in strict_zip(sources, records):
        if not isinstance(record, Mapping) or not valid_record(
            source,
            record,
            cache_dir,
        ):
            return None
        validated.append(record)
    return tuple(validated)


def build_cache(
    sources: Sequence[Path],
    *,
    manifest_path: Path,
    cache_dir: Path,
    delimiter: str,
    encoding: str,
    prepare_workers: int | None,
    progress_label: str,
    source_label: str,
) -> tuple[JsonMapping, ...]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    previous = previous_records(manifest_path)
    records: list[JsonMapping | None] = []
    jobs = []
    for source in sources:
        part = part_name(source)
        existing = previous.get(str(source))
        if existing is not None and valid_record(source, existing, cache_dir):
            records.append(existing)
            continue
        records.append(None)
        jobs.append(
            (
                len(records) - 1,
                source,
                cache_dir / part,
                delimiter,
                encoding,
                source_label,
            )
        )

    converted = convert_files(
        jobs,
        workers=prepare_workers,
        progress_label=progress_label,
    )
    for index, record in converted:
        records[index] = record
    complete = tuple(record for record in records if record is not None)
    if len(complete) != len(sources):
        raise RuntimeError(f"{source_label} parquet cache is incomplete.")
    write_json(
        manifest_path,
        {"schema_version": CACHE_SCHEMA_VERSION, "files": complete},
    )
    return complete


def previous_records(manifest_path: Path) -> dict[str, JsonMapping]:
    if not manifest_path.is_file():
        return {}
    data = read_json(manifest_path)
    if not isinstance(data, Mapping) or not valid_cache_schema(data):
        return {}
    records = data.get("files")
    if not isinstance(records, list):
        return {}
    return {
        str(record["path"]): record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }


def valid_cache_schema(data: Mapping[str, object]) -> bool:
    version = data.get("schema_version")
    return type(version) is int and version == CACHE_SCHEMA_VERSION


def source_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def same_file_record(path: Path, record: JsonMapping) -> bool:
    stat = path.stat()
    return (
        record.get("path") == str(path)
        and record.get("size") == stat.st_size
        and record.get("mtime_ns") == stat.st_mtime_ns
        and record.get("ctime_ns") == stat.st_ctime_ns
    )


def valid_record(path: Path, record: JsonMapping, cache_dir: Path) -> bool:
    if not same_file_record(path, record):
        return False
    part = record.get("part")
    row_count = record.get("row_count")
    row_groups = record.get("row_groups")
    if (
        not isinstance(part, str)
        or type(row_count) is not int
        or row_count < 0
        or not isinstance(row_groups, list)
        or any(type(value) is not int or value < 0 for value in row_groups)
        or sum(row_groups) != row_count
    ):
        return False
    try:
        validate_path_segment("cached parquet part", part)
    except (TypeError, ValueError):
        return False
    parquet_path = cache_dir / part
    if not parquet_path.is_file():
        return False
    try:
        parquet = pq.ParquetFile(parquet_path)
    except (OSError, pa.ArrowException):
        return False
    try:
        actual_groups = [
            int(parquet.metadata.row_group(group).num_rows)
            for group in range(parquet.metadata.num_row_groups)
        ]
        return parquet.metadata.num_rows == row_count and actual_groups == row_groups
    finally:
        parquet.close()


def part_name(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return f"{digest}.parquet"


def convert_files(
    jobs: Sequence[tuple[int, Path, Path, str, str, str]],
    *,
    workers: int | None = None,
    progress_label: str,
) -> tuple[tuple[int, JsonMapping], ...]:
    if not jobs:
        return ()
    if workers is None:
        workers = min(len(jobs), os.cpu_count() or 1, 8)
    else:
        workers = min(workers, len(jobs))
    if workers <= 1 or multiprocessing.current_process().daemon:
        return _convert_files_inline(jobs, progress_label=progress_label)
    try:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing_context("spawn"),
        )
    except (NotImplementedError, PermissionError):
        return _convert_files_inline(jobs, progress_label=progress_label)
    with executor as pool:
        return tuple(
            conversion_progress(
                pool.map(convert_file_job, jobs),
                total=len(jobs),
                label=progress_label,
            )
        )


def _convert_files_inline(
    jobs: Sequence[tuple[int, Path, Path, str, str, str]],
    *,
    progress_label: str,
) -> tuple[tuple[int, JsonMapping], ...]:
    converted = (convert_file_job(job) for job in jobs)
    return tuple(conversion_progress(converted, total=len(jobs), label=progress_label))


def conversion_progress(
    converted: Iterator[tuple[int, JsonMapping]],
    *,
    total: int,
    label: str,
) -> Iterator[tuple[int, JsonMapping]]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from converted
        return
    with tqdm(
        converted,
        total=total,
        unit="file",
        desc=label,
        disable=not sys.stderr.isatty(),
    ) as progress:
        yield from progress


def convert_file_job(
    job: tuple[int, Path, Path, str, str, str],
) -> tuple[int, JsonMapping]:
    index, source, target, delimiter, encoding, source_label = job
    fingerprint = source_record(source)
    names = column_names(source, delimiter=delimiter, encoding=encoding)
    table = read_csv(
        source,
        read_options=ReadOptions(use_threads=False, encoding=encoding),
        parse_options=ParseOptions(delimiter=delimiter, newlines_in_values=True),
        convert_options=ConvertOptions(
            column_types={name: pa.string() for name in names},
            strings_can_be_null=False,
        ),
    )
    validate_unchanged_source(source, fingerprint, source_label=source_label)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        ) as file:
            tmp = Path(file.name)
        pq.write_table(table, tmp, row_group_size=PARQUET_ROW_GROUP_SIZE)
        validate_unchanged_source(source, fingerprint, source_label=source_label)
        os.replace(tmp, target)
    except Exception:
        if tmp is not None and tmp.exists():
            tmp.unlink()
        raise
    parquet = pq.ParquetFile(target)
    try:
        record = {
            **fingerprint,
            "part": target.name,
            "row_count": int(parquet.metadata.num_rows),
            "row_groups": [
                int(parquet.metadata.row_group(group).num_rows)
                for group in range(parquet.metadata.num_row_groups)
            ],
        }
    finally:
        parquet.close()
    validate_unchanged_source(source, fingerprint, source_label=source_label)
    return index, record


def validate_unchanged_source(
    path: Path,
    record: JsonMapping,
    *,
    source_label: str,
) -> None:
    if not same_file_record(path, record):
        raise ValueError(
            f"{source_label} source changed while preparing parquet cache: {path}"
        )


def column_names(
    path: Path,
    *,
    delimiter: str,
    encoding: str,
) -> tuple[str, ...]:
    with path.open("r", encoding=encoding, newline="") as file:
        names = next(csv.reader(file, delimiter=delimiter), None)
    if names is None or not names or any(not name for name in names):
        raise ValueError(f"Delimited file must have a non-empty header: {path}")
    if len(set(names)) != len(names):
        raise ValueError(f"Delimited file must have unique column names: {path}")
    return tuple(names)


def iter_part_shard(
    parts: Sequence[ParquetPart],
    *,
    num_shards: int,
    shard_id: int,
    read_group: Callable[[Path, int], Sequence[Any]],
) -> Iterator[tuple[int, Any]]:
    validate_shard(num_shards, shard_id)
    for part in parts:
        for row_group, row_count in enumerate(part.row_groups):
            if row_count == 0:
                continue
            group_start = part.start + (
                0 if row_group == 0 else part.row_group_stops[row_group - 1]
            )
            first_offset = (shard_id - group_start) % num_shards
            if first_offset >= row_count:
                continue
            rows = read_group(part.path, row_group)
            for offset in range(first_offset, row_count, num_shards):
                yield group_start + offset, rows[offset]


class ParquetPartsReader:
    def __init__(self, parts: Sequence[ParquetPart]) -> None:
        self._parts = tuple(parts)
        self._part_stops = tuple(part.stop for part in self._parts)
        self._row_group_cache: OrderedDict[
            tuple[Path, int], tuple[dict[str, str], ...]
        ] = OrderedDict()
        self._parquet_cache: OrderedDict[Path, Any] = OrderedDict()
        self._pid = os.getpid()

    @property
    def parts(self) -> tuple[ParquetPart, ...]:
        return self._parts

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_row_group_cache"] = OrderedDict()
        state["_parquet_cache"] = OrderedDict()
        state["_pid"] = os.getpid()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def close(self) -> None:
        parquets = tuple(self._parquet_cache.values())
        self._parquet_cache.clear()
        self._row_group_cache.clear()
        for parquet in parquets:
            parquet.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self._parts[-1].stop if self._parts else 0

    def part_index(self, index: int) -> int:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("ParquetPartsReader index out of range.")
        return bisect_right(self._part_stops, index)

    def __getitem__(self, index: int) -> dict[str, str]:
        if index < 0:
            index += len(self)
        part = self._parts[self.part_index(index)]
        return self.read_row(part, index - part.start)

    def __iter__(self) -> Iterator[dict[str, str]]:
        for _index, row in self.iter_shard(1, 0):
            yield row

    def iter_index_groups(self) -> Iterator[range]:
        for part in self._parts:
            start = part.start
            for count in part.row_groups:
                stop = start + count
                if stop > start:
                    yield range(start, stop)
                start = stop
            if start != part.stop:
                raise RuntimeError("Parquet row group index is inconsistent.")

    def iter_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, dict[str, str]]]:
        for index, row in iter_part_shard(
            self._parts,
            num_shards=num_shards,
            shard_id=shard_id,
            read_group=self.read_group,
        ):
            yield index, row

    def read_row(self, part: ParquetPart, index: int) -> dict[str, str]:
        row_stops = part.row_group_stops
        row_group = bisect_right(row_stops, index)
        start = 0 if row_group == 0 else row_stops[row_group - 1]
        rows = self.read_group(part.path, row_group)
        return rows[index - start]

    def read_group(
        self,
        path: Path,
        row_group: int,
    ) -> tuple[dict[str, str], ...]:
        self.reset_after_fork()
        key = (path, row_group)
        rows = self._row_group_cache.get(key)
        if rows is None:
            rows = tuple(self.parquet(path).read_row_group(row_group).to_pylist())
            self._row_group_cache[key] = rows
            while len(self._row_group_cache) > MAX_CACHED_ROW_GROUPS:
                self._row_group_cache.popitem(last=False)
        else:
            self._row_group_cache.move_to_end(key)
        return rows

    def parquet(self, path: Path):
        handle = self._parquet_cache.get(path)
        if handle is not None:
            self._parquet_cache.move_to_end(path)
            return handle
        handle = pq.ParquetFile(path)
        self._parquet_cache[path] = handle
        while len(self._parquet_cache) > MAX_OPEN_PARQUET_FILES:
            _path, evicted = self._parquet_cache.popitem(last=False)
            evicted.close()
        return handle

    def reset_after_fork(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        parquets = tuple(self._parquet_cache.values())
        self._parquet_cache = OrderedDict()
        self._row_group_cache = OrderedDict()
        self._pid = pid
        for parquet in parquets:
            parquet.close()
