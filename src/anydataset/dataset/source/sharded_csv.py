from __future__ import annotations

import csv
import hashlib
import json
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
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from ..._compat import strict_zip
from ..._parallel import multiprocessing_context
from ..._logging import write_warning
from ...cache import FileLock
from ...types import Spec
from .protocol import validate_load_options


CsvRow = Mapping[str, str]
JsonMapping = Mapping[str, Any]

_CACHE_SCHEMA_VERSION = 1
_CACHE_MANIFEST = "sharded_csv_parquet.json"
_CACHE_DIR = "sharded_csv_parquet"
_PARQUET_ROW_GROUP_SIZE = 4096
_MAX_CACHED_ROW_GROUPS = 2
_PREPARE_LOCK_TIMEOUT = 3600.0
_PREPARE_LOCK_POLL = 0.2


class _TextWriter(Protocol):
    def write(self, text: str, /) -> object: ...


@dataclass(frozen=True)
class CsvShard:
    index: int
    path: Path


@dataclass(frozen=True)
class CsvFile:
    path: Path
    start: int
    stop: int
    row_groups: tuple[int, ...]


class ShardedCsvSource:
    def prepare(self, spec: Spec, cache_path: Path) -> ShardedCsvDataset:
        validate_load_options(spec, (), source="sharded_csv")
        dataset = ShardedCsvDataset(
            Path(spec.path),
            split=spec.split,
            cache_path=cache_path,
        )
        dataset.prepare()
        return dataset

    def iter_indexed_shard(
        self,
        dataset: ShardedCsvDataset,
        *,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, CsvRow]]:
        yield from dataset.iter_indexed_shard(num_shards, shard_id)


class ShardedCsvDataset:
    def __init__(
        self,
        root: Path,
        split: str | None = None,
        *,
        cache_path: Path | None = None,
    ) -> None:
        self.root = root
        self.split = split
        self.cache_path = cache_path
        self._shards_cache: tuple[CsvShard, ...] | None = None
        self._files_cache: tuple[CsvFile, ...] | None = None
        self._file_stops_cache: tuple[int, ...] | None = None
        self._row_group_cache: OrderedDict[
            tuple[Path, int], tuple[dict[str, str], ...]
        ] = OrderedDict()
        self._ignored_csv_warning_paths: set[Path] = set()

    def prepare(self) -> None:
        self._files()

    def __iter__(self) -> Iterator[CsvRow]:
        yield from self.shard(num_shards=1, index=0)

    def __len__(self) -> int:
        files = self._files()
        return files[-1].stop if files else 0

    def __getitem__(self, index: int) -> CsvRow:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("ShardedCsvDataset index out of range.")

        files = self._files()
        file_index = bisect_right(self._file_stops(), index)
        file = files[file_index]
        return self._read_parquet_row(file, index - file.start)

    def shard(self, *, num_shards: int, index: int) -> Iterator[CsvRow]:
        if num_shards <= 0:
            raise ValueError("num_shards must be positive.")
        if index < 0 or index >= num_shards:
            raise ValueError("index must satisfy 0 <= index < num_shards.")

        for shard in self._shards():
            if shard.index % num_shards == index:
                yield from self._read_shard(shard)

    def iter_indexed_range(self, start: int, stop: int) -> Iterator[tuple[int, CsvRow]]:
        length = len(self)
        if start < 0 or stop < start or stop > length:
            raise ValueError("range must satisfy 0 <= start <= stop <= len(dataset).")

        for index in range(start, stop):
            yield index, self[index]

    def iter_indexed_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, CsvRow]]:
        if num_shards <= 0:
            raise ValueError("num_shards must be positive.")
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards.")
        for index in range(shard_id, len(self), num_shards):
            yield index, self[index]

    def _base_dir(self) -> Path:
        return self.root / self.split if self.split is not None else self.root

    def _shards(self) -> tuple[CsvShard, ...]:
        if self._shards_cache is not None:
            return self._shards_cache

        base = self._base_dir()
        if not base.exists():
            raise FileNotFoundError(f"Missing sharded CSV directory: {base}")

        shards = []
        by_index: dict[int, Path] = {}
        for path in base.iterdir():
            if not path.is_dir() or (index := _shard_index(path)) is None:
                continue
            previous = by_index.get(index)
            if previous is not None:
                raise ValueError(
                    "Sharded CSV directory indexes must be unique: "
                    f"{previous.name} and {path.name} both resolve to {index}."
                )
            by_index[index] = path
            shards.append(CsvShard(index=index, path=path))
        if not shards:
            raise FileNotFoundError(f"No shard_* directories found under: {base}")
        ordered = tuple(sorted(shards, key=lambda shard: shard.index))
        _warn_missing_shards(base, ordered)
        self._shards_cache = ordered
        return ordered

    def _read_shard(self, shard: CsvShard) -> Iterator[CsvRow]:
        paths = self._csv_files(shard.path)
        if not paths:
            raise FileNotFoundError(f"No CSV files found under: {shard.path}")
        for path in paths:
            yield from self._read_file(path)

    def _read_file(self, path: Path) -> Iterator[CsvRow]:
        with path.open("r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f, **self._csv_options())

    def _files(self) -> tuple[CsvFile, ...]:
        if self._files_cache is not None:
            return self._files_cache

        paths = self._csv_paths()
        records = self._prepared_records(paths)
        start = 0
        files = []
        for record in records:
            count = int(record["row_count"])
            stop = start + count
            files.append(
                CsvFile(
                    path=self._cache_dir() / str(record["part"]),
                    start=start,
                    stop=stop,
                    row_groups=tuple(int(value) for value in record["row_groups"]),
                )
            )
            start = stop

        self._files_cache = tuple(files)
        self._file_stops_cache = tuple(file.stop for file in self._files_cache)
        return self._files_cache

    def _csv_paths(self) -> tuple[Path, ...]:
        paths = []
        for shard in self._shards():
            shard_paths = self._csv_files(shard.path)
            if not shard_paths:
                raise FileNotFoundError(f"No CSV files found under: {shard.path}")
            paths.extend(shard_paths)
        return tuple(paths)

    def _csv_files(self, path: Path) -> tuple[Path, ...]:
        paths, ignored = _split_csv_paths(path)
        if ignored and path not in self._ignored_csv_warning_paths:
            self._ignored_csv_warning_paths.add(path)
            ignored_names = ", ".join(file.name for file in ignored)
            _write_warning(
                f"Ignored non-numeric CSV files under {path}: {ignored_names}."
            )
        return paths

    def _file_stops(self) -> tuple[int, ...]:
        if self._file_stops_cache is None:
            self._files()
        if self._file_stops_cache is None:
            raise RuntimeError("CSV file index cache was not initialized.")
        return self._file_stops_cache

    def _prepared_records(self, paths: Sequence[Path]) -> tuple[JsonMapping, ...]:
        if self.cache_path is None:
            raise ValueError("sharded_csv requires a source cache path.")
        lock_path = self.cache_path / ".prepare.lock"
        cached = self._read_cache(paths)
        if cached is not None:
            return cached
        with FileLock(
            lock_path,
            wait_timeout=_PREPARE_LOCK_TIMEOUT,
            poll_interval=_PREPARE_LOCK_POLL,
        ):
            cached = self._read_cache(paths)
            if cached is not None:
                return cached
            return self._build_cache(paths)

    def _read_cache(self, paths: Sequence[Path]) -> tuple[JsonMapping, ...] | None:
        path = self._manifest_path()
        if path is None or not path.is_file():
            return None
        data = _read_json(path)
        if not isinstance(data, Mapping):
            raise ValueError(f"Invalid sharded CSV parquet manifest: {path}")
        if data.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return None
        records = data.get("files")
        if not isinstance(records, list) or len(records) != len(paths):
            return None
        validated = []
        for source, record in strict_zip(paths, records):
            if not isinstance(record, Mapping) or not _valid_record(
                source,
                record,
                self._cache_dir(),
            ):
                return None
            validated.append(record)
        return tuple(validated)

    def _build_cache(self, paths: Sequence[Path]) -> tuple[JsonMapping, ...]:
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        previous = self._previous_records()
        records: list[JsonMapping | None] = []
        jobs = []
        for source in paths:
            part = _part_name(source)
            existing = previous.get(str(source))
            if existing is not None and _valid_record(source, existing, cache_dir):
                records.append(existing)
                continue
            records.append(None)
            jobs.append((len(records) - 1, source, cache_dir / part))

        converted = _convert_files(jobs)
        for index, record in converted:
            records[index] = record
        complete = tuple(record for record in records if record is not None)
        if len(complete) != len(paths):
            raise RuntimeError("Sharded CSV parquet cache is incomplete.")
        manifest = self._manifest_path()
        if manifest is None:
            raise ValueError("sharded_csv requires a source cache path.")
        _write_json(
            manifest,
            {"schema_version": _CACHE_SCHEMA_VERSION, "files": complete},
        )
        return complete

    def _previous_records(self) -> dict[str, JsonMapping]:
        path = self._manifest_path()
        if path is None or not path.is_file():
            return {}
        data = _read_json(path)
        if not isinstance(data, Mapping) or data.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return {}
        records = data.get("files")
        if not isinstance(records, list):
            return {}
        return {
            str(record["path"]): record
            for record in records
            if isinstance(record, Mapping) and isinstance(record.get("path"), str)
        }

    def _manifest_path(self) -> Path | None:
        return None if self.cache_path is None else self.cache_path / _CACHE_MANIFEST

    def _cache_dir(self) -> Path:
        if self.cache_path is None:
            raise ValueError("sharded_csv requires a source cache path.")
        return self.cache_path / _CACHE_DIR

    def _read_parquet_row(self, file: CsvFile, index: int) -> dict[str, str]:
        stops = _stops(file.row_groups)
        row_group = bisect_right(stops, index)
        start = 0 if row_group == 0 else stops[row_group - 1]
        key = (file.path, row_group)
        rows = self._row_group_cache.get(key)
        if rows is None:
            rows = tuple(pq.ParquetFile(file.path).read_row_group(row_group).to_pylist())
            self._row_group_cache[key] = rows
            while len(self._row_group_cache) > _MAX_CACHED_ROW_GROUPS:
                self._row_group_cache.popitem(last=False)
        else:
            self._row_group_cache.move_to_end(key)
        return rows[index - start]

    def _csv_options(self) -> dict[str, Any]:
        return {}


def _shard_index(path: Path) -> int | None:
    prefix = "shard_"
    name = path.name
    if not name.startswith(prefix):
        return None

    suffix = name[len(prefix) :]
    if not suffix.isdecimal():
        return None
    return int(suffix)


def _split_csv_paths(path: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    paths = []
    ignored = []
    for child in path.glob("*.csv"):
        if not child.is_file():
            continue
        if child.stem.isdecimal():
            paths.append(child)
        else:
            ignored.append(child)
    ordered = tuple(sorted(paths, key=_csv_path_key))
    for previous, current in zip(ordered, ordered[1:]):
        if _csv_path_key(previous) == _csv_path_key(current):
            raise ValueError(
                "Numeric CSV file indexes must be unique: "
                f"{previous.name} and {current.name} both resolve to "
                f"{_csv_path_key(current)}."
            )
    return ordered, tuple(sorted(ignored, key=lambda child: child.name))


def _csv_path_key(path: Path) -> int:
    return int(path.stem)


def _source_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _same_file_record(path: Path, record: JsonMapping) -> bool:
    stat = path.stat()
    return (
        record.get("path") == str(path)
        and record.get("size") == stat.st_size
        and record.get("mtime_ns") == stat.st_mtime_ns
    )


def _valid_record(path: Path, record: JsonMapping, cache_dir: Path) -> bool:
    if not _same_file_record(path, record):
        return False
    part = record.get("part")
    row_count = record.get("row_count")
    row_groups = record.get("row_groups")
    if (
        not isinstance(part, str)
        or not isinstance(row_count, int)
        or row_count < 0
        or not isinstance(row_groups, list)
        or any(not isinstance(value, int) or value < 0 for value in row_groups)
        or sum(row_groups) != row_count
    ):
        return False
    parquet = cache_dir / part
    return parquet.is_file() and pq.ParquetFile(parquet).metadata.num_rows == row_count


def _part_name(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return f"{digest}.parquet"


def _convert_files(
    jobs: Sequence[tuple[int, Path, Path]],
) -> tuple[tuple[int, JsonMapping], ...]:
    if not jobs:
        return ()
    workers = min(len(jobs), os.cpu_count() or 1, 8)
    if workers == 1 or multiprocessing.current_process().daemon:
        converted = (_convert_file_job(job) for job in jobs)
        return tuple(_conversion_progress(converted, total=len(jobs)))
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing_context("spawn"),
    ) as executor:
        return tuple(
            _conversion_progress(
                executor.map(_convert_file_job, jobs),
                total=len(jobs),
            )
        )


def _conversion_progress(
    converted: Iterator[tuple[int, JsonMapping]],
    *,
    total: int,
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
        desc="prepare sharded CSV",
        disable=not sys.stderr.isatty(),
    ) as progress:
        yield from progress


def _convert_file_job(job: tuple[int, Path, Path]) -> tuple[int, JsonMapping]:
    index, source, target = job
    names = _csv_names(source)
    table = pa_csv.read_csv(
        source,
        read_options=pa_csv.ReadOptions(use_threads=False),
        convert_options=pa_csv.ConvertOptions(
            column_types={name: pa.string() for name in names},
            strings_can_be_null=False,
        ),
    )
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        ) as file:
            tmp = Path(file.name)
        pq.write_table(table, tmp, row_group_size=_PARQUET_ROW_GROUP_SIZE)
        os.replace(tmp, target)
    except Exception:
        if tmp is not None and tmp.exists():
            tmp.unlink()
        raise
    parquet = pq.ParquetFile(target)
    record = {
        **_source_record(source),
        "part": target.name,
        "row_count": int(parquet.metadata.num_rows),
        "row_groups": [
            int(parquet.metadata.row_group(group).num_rows)
            for group in range(parquet.metadata.num_row_groups)
        ],
    }
    return index, record


def _csv_names(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as file:
        names = next(csv.reader(file), None)
    if names is None or not names or any(not name for name in names):
        raise ValueError(f"CSV file must have a non-empty header: {path}")
    if len(set(names)) != len(names):
        raise ValueError(f"CSV file must have unique column names: {path}")
    return tuple(names)


def _stops(counts: Sequence[int]) -> tuple[int, ...]:
    total = 0
    stops = []
    for count in counts:
        total += count
        stops.append(total)
    return tuple(stops)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    def write(file: _TextWriter) -> None:
        file.write(text)

    _atomic_write(path, write)


def _atomic_write(path: Path, write: Callable[[_TextWriter], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=path.parent,
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as file:
            tmp_path = Path(file.name)
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise


def _warn_missing_shards(base: Path, shards: Sequence[CsvShard]) -> None:
    missing = _missing_shard_ranges(shards)
    if not missing:
        return

    missing_names = ", ".join(
        f"shard_{start}"
        if start == stop
        else f"shard_{start}..shard_{stop}"
        for start, stop in missing
    )
    _write_warning(
        f"Missing sharded CSV directories under {base}: {missing_names}."
    )


def _missing_shard_ranges(
    shards: Sequence[CsvShard],
) -> tuple[tuple[int, int], ...]:
    missing = []
    previous = -1
    for shard in shards:
        if shard.index > previous + 1:
            missing.append((previous + 1, shard.index - 1))
        previous = shard.index
    return tuple(missing)


def _write_warning(message: str) -> None:
    write_warning("sharded_csv", message)
