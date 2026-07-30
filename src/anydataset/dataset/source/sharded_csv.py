from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..._runtime.logging import write_warning
from ...types import Spec
from . import _tabular_parquet as tabular
from .protocol import validate_load_options


CsvRow = Mapping[str, str]

_CACHE_MANIFEST = "sharded_csv_parquet.json"
_CACHE_DIR = "sharded_csv_parquet"


@dataclass(frozen=True)
class CsvShard:
    index: int
    path: Path


CsvFile = tabular.ParquetPart


class ShardedCsvSource:
    def prepare(self, spec: Spec, cache_path: Path) -> ShardedCsvDataset:
        validate_load_options(spec, ("prepare_workers",), source="sharded_csv")
        prepare_workers = spec.load_options.get("prepare_workers")
        if prepare_workers is not None:
            tabular.validate_prepare_workers(prepare_workers)
        dataset = ShardedCsvDataset(
            Path(spec.path),
            split=spec.split,
            cache_path=cache_path,
            prepare_workers=prepare_workers,
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
        prepare_workers: int | None = None,
    ) -> None:
        if prepare_workers is not None:
            tabular.validate_prepare_workers(prepare_workers)
        self.root = root
        self.split = split
        self.cache_path = cache_path
        self.prepare_workers = prepare_workers
        self._shards_cache: tuple[CsvShard, ...] | None = None
        self._reader: tabular.ParquetPartsReader | None = None
        self._ignored_csv_warning_paths: set[Path] = set()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        reader = self._reader
        if reader is not None:
            state["_reader"] = tabular.ParquetPartsReader(reader.parts)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "prepare_workers" not in state:
            self.prepare_workers = None
        if "_reader" not in state:
            self._reader = None
        if "_files_cache" in state or "_file_stops_cache" in state:
            # Old pickles stored file caches; drop them and rebuild via prepare.
            self._reader = None
            self.__dict__.pop("_files_cache", None)
            self.__dict__.pop("_file_stops_cache", None)
            self.__dict__.pop("_row_group_cache", None)
            self.__dict__.pop("_parquet_cache", None)
            self.__dict__.pop("_pid", None)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def prepare(self) -> None:
        self._files()

    def __iter__(self) -> Iterator[CsvRow]:
        yield from self.shard(num_shards=1, index=0)

    def __len__(self) -> int:
        return len(self._parts_reader())

    def iter_index_groups(self) -> Iterator[range]:
        yield from self._parts_reader().iter_index_groups()

    def __getitem__(self, index: int) -> CsvRow:
        try:
            return self._parts_reader()[index]
        except IndexError as exc:
            raise IndexError("ShardedCsvDataset index out of range.") from exc

    def shard(self, *, num_shards: int, index: int) -> Iterator[CsvRow]:
        """Iterate rows from physical CSV shard directories selected by directory index.

        This is a low-level helper for directory-oriented scans. Anydataset
        training, write, and filter paths use ``iter_indexed_shard`` (dense
        global sample-index modulo) instead of this method.
        """
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
        # Prefer ``_read_parquet_group`` so callers can observe/wrap row-group IO.
        for file in self._files():
            for row_group, row_count in enumerate(file.row_groups):
                if row_count == 0:
                    continue
                group_start = file.start + (
                    0
                    if row_group == 0
                    else file.row_group_stops[row_group - 1]
                )
                first_offset = (shard_id - group_start) % num_shards
                if first_offset >= row_count:
                    continue
                rows = self._read_parquet_group(file.path, row_group)
                for offset in range(first_offset, row_count, num_shards):
                    yield group_start + offset, rows[offset]

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
        return self._parts_reader().parts

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

    def _parts_reader(self) -> tabular.ParquetPartsReader:
        if self._reader is not None:
            return self._reader
        if self.cache_path is None:
            raise ValueError("sharded_csv requires a source cache path.")
        parts = tabular.ensure_parts(
            self._csv_paths(),
            cache_path=self.cache_path,
            manifest_name=_CACHE_MANIFEST,
            cache_dir_name=_CACHE_DIR,
            delimiter=",",
            encoding="utf-8",
            prepare_workers=self.prepare_workers,
            progress_label="prepare sharded CSV",
            source_label="sharded CSV",
        )
        self._reader = tabular.ParquetPartsReader(parts)
        return self._reader

    def _read_parquet_group(
        self,
        path: Path,
        row_group: int,
    ) -> tuple[dict[str, str], ...]:
        return self._parts_reader().read_group(path, row_group)

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


def _warn_missing_shards(base: Path, shards: Sequence[CsvShard]) -> None:
    missing = _missing_shard_ranges(shards)
    if not missing:
        return

    missing_names = ", ".join(
        f"shard_{start}" if start == stop else f"shard_{start}..shard_{stop}"
        for start, stop in missing
    )
    _write_warning(f"Missing sharded CSV directories under {base}: {missing_names}.")


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
