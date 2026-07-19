from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...types import Spec
from .protocol import validate_load_options


TsvRow = Mapping[str, str]


class TsvSource:
    def prepare(self, spec: Spec, cache_path: Path) -> TsvDataset:
        validate_load_options(
            spec,
            {"encoding", "root_field", "subdirs"},
            source="TSV",
        )
        return TsvDataset(
            Path(spec.path),
            split=spec.split,
            encoding=_required_str(
                spec.load_options.get("encoding", "utf-8"),
                "encoding",
            ),
            subdirs=_optional_str_sequence(spec.load_options.get("subdirs"), "subdirs"),
            root_field=_optional_str(spec.load_options.get("root_field"), "root_field"),
        )


class TsvDataset:
    def __init__(
        self,
        root: Path,
        split: str | None = None,
        *,
        encoding: str = "utf-8",
        subdirs: Sequence[str] | None = None,
        root_field: str | None = None,
    ) -> None:
        self.root = root
        self.split = split
        self.encoding = encoding
        self.subdirs = None if subdirs is None else tuple(subdirs)
        self.root_field = root_field

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

    def _roots(self) -> Iterator[Path]:
        if self.subdirs is None:
            yield self.root
            return
        if self.root.is_file():
            raise ValueError("TSV source subdirs require path to be a directory.")
        for subdir in self.subdirs:
            yield self.root / subdir

    def _path(self, root: Path) -> Path:
        if root.is_file():
            return root
        if self.split is None:
            raise ValueError("TSV source requires split when path is a directory.")
        return root / f"{self.split}.tsv"

    def _read_rows(self) -> Iterator[TsvRow]:
        for root in self._roots():
            path = self._path(root)
            with path.open("r", encoding=self.encoding, newline="") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    if self.root_field is None:
                        yield row
                        continue
                    if self.root_field in row:
                        raise ValueError(
                            f"TSV row already has root field: {self.root_field}"
                        )
                    yield {**row, self.root_field: str(root)}


def _optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"TSV {name} must be a string.")
    if not value:
        raise ValueError(f"TSV {name} must not be empty.")
    return value


def _required_str(value: Any, name: str) -> str:
    result = _optional_str(value, name)
    if result is None:
        raise TypeError(f"TSV {name} must be a string.")
    return result


def _optional_str_sequence(value: Any, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        result = (value,)
    else:
        if not isinstance(value, Sequence):
            raise TypeError(f"TSV {name} must be a string sequence.")
        result = tuple(value)
    if not result:
        raise ValueError(f"TSV {name} must not be empty.")
    for item in result:
        if not isinstance(item, str):
            raise TypeError(f"TSV {name} items must be strings.")
        if not item:
            raise ValueError(f"TSV {name} items must not be empty.")
    return result
