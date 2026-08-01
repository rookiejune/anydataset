from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..._validation import validate_path_segment
from ...types import Spec
from . import _tabular_parquet as tabular
from .protocol import _validate_load_options


TsvRow = Mapping[str, str]

_CACHE_MANIFEST = "tsv_parquet.json"
_CACHE_DIR = "tsv_parquet"


class TsvSource:
    def prepare(self, spec: Spec, cache_path: Path) -> _TsvDataset:
        _validate_load_options(
            spec,
            {"encoding", "root_field", "subdirs", "prepare_workers"},
            source="TSV",
        )
        prepare_workers = spec.load_options.get("prepare_workers")
        if prepare_workers is not None:
            tabular.validate_prepare_workers(prepare_workers)
        dataset = _TsvDataset(
            Path(spec.path),
            split=spec.split,
            cache_path=cache_path,
            encoding=_required_str(
                spec.load_options.get("encoding", "utf-8"),
                "encoding",
            ),
            subdirs=_optional_str_sequence(spec.load_options.get("subdirs"), "subdirs"),
            root_field=_optional_str(spec.load_options.get("root_field"), "root_field"),
            prepare_workers=prepare_workers,
        )
        dataset.prepare()
        return dataset

    def iter_shard(
        self,
        dataset: _TsvDataset,
        *,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, TsvRow]]:
        yield from dataset.iter_shard(num_shards, shard_id)


class _TsvDataset:
    def __init__(
        self,
        root: Path,
        split: str | None = None,
        *,
        cache_path: Path | None = None,
        encoding: str = "utf-8",
        subdirs: Sequence[str] | None = None,
        root_field: str | None = None,
        prepare_workers: int | None = None,
    ) -> None:
        if prepare_workers is not None:
            tabular.validate_prepare_workers(prepare_workers)
        self.root = root
        self.split = split
        self.cache_path = cache_path
        self.encoding = encoding
        self.subdirs = None if subdirs is None else tuple(subdirs)
        self.root_field = root_field
        self.prepare_workers = prepare_workers
        self._sources_cache: tuple[tuple[Path, Path], ...] | None = None
        self._reader: tabular.ParquetPartsReader | None = None

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

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def prepare(self) -> None:
        self._parts_reader()

    def __iter__(self) -> Iterator[TsvRow]:
        for _index, row in self.iter_shard(1, 0):
            yield row

    def __len__(self) -> int:
        return len(self._parts_reader())

    def iter_index_groups(self) -> Iterator[range]:
        yield from self._parts_reader().iter_index_groups()

    def __getitem__(self, index: int) -> TsvRow:
        reader = self._parts_reader()
        length = len(reader)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("TSV dataset index out of range.")
        return self._enrich(reader[index], self._part_root(index))

    def iter_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, TsvRow]]:
        sources = self._sources()
        reader = self._parts_reader()
        for index, row in reader.iter_shard(num_shards, shard_id):
            yield index, self._enrich(row, sources[reader.part_index(index)][0])

    def _enrich(self, row: Mapping[str, str], language_root: Path) -> dict[str, str]:
        if self.root_field is None:
            return dict(row)
        if self.root_field in row:
            raise ValueError(f"TSV row already has root field: {self.root_field}")
        return {**row, self.root_field: str(language_root)}

    def _part_root(self, index: int) -> Path:
        return self._sources()[self._parts_reader().part_index(index)][0]

    def _sources(self) -> tuple[tuple[Path, Path], ...]:
        if self._sources_cache is not None:
            return self._sources_cache
        sources = []
        for language_root in self._roots():
            sources.append((language_root, self._path(language_root)))
        if not sources:
            raise FileNotFoundError(f"No TSV sources found under: {self.root}")
        self._sources_cache = tuple(sources)
        return self._sources_cache

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
            if self.split is not None:
                raise ValueError(
                    "TSV source split is only supported for directory paths."
                )
            return root
        if self.split is None:
            raise ValueError("TSV source requires split when path is a directory.")
        validate_path_segment("TSV split", self.split)
        return root / f"{self.split}.tsv"

    def _parts_reader(self) -> tabular.ParquetPartsReader:
        if self._reader is not None:
            return self._reader
        if self.cache_path is None:
            raise ValueError("tsv requires a source cache path.")
        paths = tuple(path for _root, path in self._sources())
        parts = tabular.ensure_parts(
            paths,
            cache_path=self.cache_path,
            manifest_name=_CACHE_MANIFEST,
            cache_dir_name=_CACHE_DIR,
            delimiter="\t",
            encoding=self.encoding,
            prepare_workers=self.prepare_workers,
            progress_label="prepare TSV",
            source_label="TSV",
        )
        self._reader = tabular.ParquetPartsReader(parts)
        return self._reader


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
            raise TypeError(f"TSV {name} must contain strings.")
        if not item:
            raise ValueError(f"TSV {name} must not contain empty strings.")
    return result
