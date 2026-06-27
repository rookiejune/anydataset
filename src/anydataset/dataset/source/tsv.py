from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path

from ...types import Spec


type TsvRow = Mapping[str, str]


class TsvSource:
    def prepare(self, spec: Spec, cache_path: Path) -> TsvDataset:
        return TsvDataset(
            Path(spec.path),
            split=spec.split,
            encoding=str(spec.load_options.get("encoding", "utf-8")),
        )


class TsvDataset:
    def __init__(
        self,
        root: Path,
        split: str | None = None,
        *,
        encoding: str = "utf-8",
    ) -> None:
        self.root = root
        self.split = split
        self.encoding = encoding

    def __iter__(self) -> Iterator[TsvRow]:
        yield from self.shard(num_shards=1, index=0)

    def shard(self, *, num_shards: int, index: int) -> Iterator[TsvRow]:
        if num_shards <= 0:
            raise ValueError("num_shards must be positive.")
        if index < 0 or index >= num_shards:
            raise ValueError("index must satisfy 0 <= index < num_shards.")

        for row_index, row in enumerate(self._read_rows()):
            if row_index % num_shards == index:
                yield row

    def _path(self) -> Path:
        if self.root.is_file():
            return self.root
        if self.split is None:
            raise ValueError("TSV source requires split when path is a directory.")
        return self.root / f"{self.split}.tsv"

    def _read_rows(self) -> Iterator[TsvRow]:
        path = self._path()
        with path.open("r", encoding=self.encoding, newline="") as f:
            yield from csv.DictReader(f, delimiter="\t")
