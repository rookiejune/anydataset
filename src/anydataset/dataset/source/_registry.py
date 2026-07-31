from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, cast

from ...types import Source, SourceKey, source_key
from .protocol import DatasetSource

DatasetSourceFactory = Callable[[], Any]


class SourceFactory:
    _factories: ClassVar[dict[str, DatasetSourceFactory]] = {}

    @classmethod
    def register(cls, source: SourceKey, factory: DatasetSourceFactory) -> None:
        key = source_key(source)
        if key in cls._factories:
            raise ValueError(f"Dataset source {key!r} is already registered.")
        if not callable(factory):
            raise TypeError("Dataset source factory must be callable.")
        cls._factories[key] = factory

    @classmethod
    def create(cls, source: SourceKey) -> DatasetSource:
        key = source_key(source)
        factory = cls._factories.get(key)
        if factory is None:
            raise KeyError(f"Unknown dataset source: {key!r}.")
        return cast(DatasetSource, factory())

    @classmethod
    def exist(cls, source: SourceKey) -> bool:
        return source_key(source) in cls._factories


def register_source(source: SourceKey, factory: DatasetSourceFactory) -> None:
    SourceFactory.register(source, factory)


def create_source(source: SourceKey) -> DatasetSource:
    return SourceFactory.create(source)


def source_exists(source: SourceKey) -> bool:
    return SourceFactory.exist(source)


def _register_builtin_sources() -> None:
    from .hf_files import HuggingFaceFilesSource
    from .huggingface import HuggingFaceDiskSource, HuggingFaceSource
    from .sharded_csv import ShardedCsvSource
    from .store import StoreSource
    from .tsv import TsvSource

    SourceFactory.register(Source.HF, HuggingFaceSource)
    SourceFactory.register(Source.HF_DISK, HuggingFaceDiskSource)
    SourceFactory.register(Source.HF_FILES, HuggingFaceFilesSource)
    SourceFactory.register(Source.STORE, StoreSource)
    SourceFactory.register("sharded_csv", ShardedCsvSource)
    SourceFactory.register("tsv", TsvSource)


_register_builtin_sources()
