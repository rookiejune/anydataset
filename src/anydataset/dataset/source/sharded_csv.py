from __future__ import annotations

import csv
from datetime import datetime
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...types import Spec


type CsvRow = Mapping[str, str]


@dataclass(frozen=True)
class CsvShard:
    index: int
    path: Path


class ShardedCsvSource:
    def prepare(self, spec: Spec, cache_path: Path) -> ShardedCsvDataset:
        return ShardedCsvDataset(Path(spec.path), split=spec.split)


class ShardedCsvDataset:
    def __init__(self, root: Path, split: str | None = None) -> None:
        self.root = root
        self.split = split
        self._shards_cache: tuple[CsvShard, ...] | None = None

    def __iter__(self) -> Iterator[CsvRow]:
        yield from self.shard(num_shards=1, index=0)

    def shard(self, *, num_shards: int, index: int) -> Iterator[CsvRow]:
        if num_shards <= 0:
            raise ValueError("num_shards must be positive.")
        if index < 0 or index >= num_shards:
            raise ValueError("index must satisfy 0 <= index < num_shards.")

        for shard in self._shards():
            if shard.index % num_shards == index:
                yield from self._read_shard(shard)

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
        paths = sorted(shard.path.glob("*.csv"), key=_csv_path_key)
        if not paths:
            raise FileNotFoundError(f"No CSV files found under: {shard.path}")
        for path in paths:
            yield from self._read_file(path)

    def _read_file(self, path: Path) -> Iterator[CsvRow]:
        with path.open("r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f, **self._csv_options())

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


def _csv_path_key(path: Path) -> tuple[int, int | str]:
    if path.stem.isdecimal():
        return (0, int(path.stem))
    return (1, path.name)


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
    log_dir = Path.home() / ".anydataset" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with (log_dir / "sharded_csv.log").open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} WARNING {message}\n")
