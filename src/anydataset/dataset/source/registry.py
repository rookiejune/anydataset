from __future__ import annotations

from collections.abc import Callable

from ...types import Source, SourceKey, source_key
from .protocol import DatasetSource

type SourceFactory = Callable[[], DatasetSource]

_SOURCE_FACTORIES: dict[str, SourceFactory] = {}


def register_source(source: SourceKey, factory: SourceFactory) -> None:
    key = source_key(source)
    if key in _SOURCE_FACTORIES:
        raise ValueError(f"Dataset source {key!r} is already registered.")
    _SOURCE_FACTORIES[key] = factory


def for_source(source: SourceKey) -> DatasetSource:
    key = source_key(source)
    factory = _SOURCE_FACTORIES.get(key)
    if factory is None:
        raise KeyError(f"Unknown dataset source: {key!r}.")
    return factory()


def has_source(source: SourceKey) -> bool:
    return source_key(source) in _SOURCE_FACTORIES


def _register_builtin_sources() -> None:
    from .huggingface import HuggingFaceDiskSource, HuggingFaceSource
    from .sharded_csv import ShardedCsvSource
    from .store import StoreSource
    from .tsv import TsvSource

    register_source(Source.HF, HuggingFaceSource)
    register_source(Source.HF_DISK, HuggingFaceDiskSource)
    register_source(Source.STORE, StoreSource)
    register_source("sharded_csv", ShardedCsvSource)
    register_source("tsv", TsvSource)


_register_builtin_sources()
