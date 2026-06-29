from __future__ import annotations

import csv
import json
import os
import tempfile
from bisect import bisect_right
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..._logging import write_warning
from ...types import Spec


type CsvRow = Mapping[str, str]
type JsonMapping = Mapping[str, Any]

_INDEX_SCHEMA_VERSION = 1
_INDEX_FILE = "sharded_csv_index.json"


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


class ShardedCsvSource:
    def prepare(self, spec: Spec, cache_path: Path) -> ShardedCsvDataset:
        return ShardedCsvDataset(Path(spec.path), split=spec.split, cache_path=cache_path)


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
        self._ignored_csv_warning_paths: set[Path] = set()

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
        return self._read_file_row(file.path, index - file.start)

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

        for file in self._files():
            if file.stop <= start:
                continue
            if file.start >= stop:
                return
            for row_index, row in enumerate(self._read_file(file.path), start=file.start):
                if row_index < start:
                    continue
                if row_index >= stop:
                    break
                yield row_index, row

    def iter_indexed_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, CsvRow]]:
        if num_shards <= 0:
            raise ValueError("num_shards must be positive.")
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards.")

        for file in self._files():
            for row_index, row in enumerate(self._read_file(file.path), start=file.start):
                if row_index % num_shards == shard_id:
                    yield row_index, row

    def _base_dir(self) -> Path:
        return self.root / self.split if self.split is not None else self.root

    def _shards(self) -> tuple[CsvShard, ...]:
        if self._shards_cache is not None:
            return self._shards_cache

        base = self._base_dir()
        if not base.exists():
            raise FileNotFoundError(f"Missing sharded CSV directory: {base}")

        shards = [
            CsvShard(index=index, path=path)
            for path in base.iterdir()
            if path.is_dir() and (index := _shard_index(path)) is not None
        ]
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

    def _read_file_row(self, path: Path, row_index: int) -> CsvRow:
        for index, row in enumerate(self._read_file(path)):
            if index == row_index:
                return row
        raise IndexError("CSV file row count changed after indexing.")

    def _files(self) -> tuple[CsvFile, ...]:
        if self._files_cache is not None:
            return self._files_cache

        start = 0
        files = []
        paths = self._csv_paths()
        counts = self._row_counts(paths)
        for path, count in zip(paths, counts, strict=True):
            stop = start + count
            files.append(CsvFile(path=path, start=start, stop=stop))
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

    def _row_counts(self, paths: Sequence[Path]) -> tuple[int, ...]:
        cached = self._read_index_cache(paths)
        if cached is not None:
            return cached

        counts = tuple(self._row_count(path) for path in paths)
        self._write_index_cache(paths, counts)
        return counts

    def _read_index_cache(self, paths: Sequence[Path]) -> tuple[int, ...] | None:
        path = self._index_path()
        if path is None or not path.is_file():
            return None

        data = _read_json(path)
        if not isinstance(data, Mapping):
            raise ValueError(f"Invalid sharded CSV index cache: {path}")
        if data.get("schema_version") != _INDEX_SCHEMA_VERSION:
            return None

        files = data.get("files")
        if not isinstance(files, list):
            raise ValueError(f"Invalid sharded CSV index cache: {path}")
        if len(files) != len(paths):
            return None

        counts = []
        for csv_path, record in zip(paths, files, strict=True):
            if not isinstance(record, Mapping):
                raise ValueError(f"Invalid sharded CSV index cache: {path}")
            if not _same_file_record(csv_path, record):
                return None
            count = record.get("row_count")
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"Invalid sharded CSV index cache: {path}")
            counts.append(count)
        return tuple(counts)

    def _write_index_cache(self, paths: Sequence[Path], counts: Sequence[int]) -> None:
        path = self._index_path()
        if path is None:
            return
        _write_json(
            path,
            {
                "schema_version": _INDEX_SCHEMA_VERSION,
                "files": [
                    _file_record(csv_path, count)
                    for csv_path, count in zip(paths, counts, strict=True)
                ],
            },
        )

    def _index_path(self) -> Path | None:
        if self.cache_path is None:
            return None
        return self.cache_path / _INDEX_FILE

    def _row_count(self, path: Path) -> int:
        with path.open("r", encoding="utf-8", newline="") as f:
            return sum(1 for _row in csv.DictReader(f, **self._csv_options()))

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
    return (
        tuple(sorted(paths, key=_csv_path_key)),
        tuple(sorted(ignored, key=lambda child: child.name)),
    )


def _csv_path_key(path: Path) -> int:
    return int(path.stem)


def _file_record(path: Path, row_count: int) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "row_count": row_count,
    }


def _same_file_record(path: Path, record: JsonMapping) -> bool:
    stat = path.stat()
    return (
        record.get("path") == str(path)
        and record.get("size") == stat.st_size
        and record.get("mtime_ns") == stat.st_mtime_ns
    )


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
    indices = {shard.index for shard in shards}
    missing = [index for index in range(max(indices) + 1) if index not in indices]
    if not missing:
        return

    missing_names = ", ".join(f"shard_{index}" for index in missing)
    _write_warning(
        f"Missing sharded CSV directories under {base}: {missing_names}."
    )


def _write_warning(message: str) -> None:
    write_warning("sharded_csv", message)
