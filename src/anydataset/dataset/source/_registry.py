from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from ..._source_identity import SourceIdentityPolicy, register_source_identity
from ...types import Source, SourceKey, source_key
from .protocol import DatasetSource

DatasetSourceFactory = Callable[[], Any]


@dataclass(frozen=True)
class _SourceRegistration:
    factory: DatasetSourceFactory
    identity: SourceIdentityPolicy


class SourceFactory:
    _registrations: ClassVar[dict[str, _SourceRegistration]] = {}

    @classmethod
    def register(
        cls,
        source: SourceKey,
        factory: DatasetSourceFactory,
        *,
        operational_load_options: Collection[str] = (),
    ) -> None:
        key = source_key(source)
        if key in cls._registrations:
            raise ValueError(f"Dataset source {key!r} is already registered.")
        if not callable(factory):
            raise TypeError("Dataset source factory must be callable.")
        cls._registrations[key] = _SourceRegistration(
            factory=factory,
            identity=register_source_identity(key, operational_load_options),
        )

    @classmethod
    def create(cls, source: SourceKey) -> DatasetSource:
        key = source_key(source)
        registration = cls._registrations.get(key)
        if registration is None:
            raise KeyError(f"Unknown dataset source: {key!r}.")
        return cast(DatasetSource, registration.factory())

    @classmethod
    def exist(cls, source: SourceKey) -> bool:
        return source_key(source) in cls._registrations


def register_source(
    source: SourceKey,
    factory: DatasetSourceFactory,
    *,
    operational_load_options: Collection[str] = (),
) -> None:
    SourceFactory.register(
        source,
        factory,
        operational_load_options=operational_load_options,
    )


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
    SourceFactory.register(
        "tsv",
        TsvSource,
        operational_load_options=("root_field",),
    )


_register_builtin_sources()
