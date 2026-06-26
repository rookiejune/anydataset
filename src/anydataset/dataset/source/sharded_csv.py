from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ...types import Spec


type CsvRow = Mapping[str, str]


class ShardedCsvSource:
    def prepare(self, spec: Spec, cache_path: Path) -> ShardedCsvDataset:
        return ShardedCsvDataset(Path(spec.path), split=spec.split)


class ShardedCsvDataset:
    def __init__(self, root: Path, split: str | None = None) -> None:
        self.root = root
        self.split = split

    def __iter__(self) -> Iterator[CsvRow]:
        yield from self.shard(num_shards=1, index=0)

    def shard(self, *, num_shards: int, index: int) -> Iterator[CsvRow]:
        if num_shards <= 0:
            raise ValueError("num_shards must be positive.")
        if index < 0 or index >= num_shards:
            raise ValueError("index must satisfy 0 <= index < num_shards.")

        shard_file = self._shard_dir(num_shards) / f"{index}.csv"
        if not shard_file.exists():
            raise FileNotFoundError(f"Missing sharded CSV file: {shard_file}")
        yield from self._read_file(shard_file)

    def _shard_dir(self, num_shards: int) -> Path:
        base = self.root / self.split if self.split is not None else self.root
        return base / f"shard_{num_shards}"

    def _read_file(self, path: Path) -> Iterator[CsvRow]:
        with path.open("r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f, **self._csv_options())

    def _csv_options(self) -> dict[str, Any]:
        return {}
