from __future__ import annotations

from collections.abc import Collection

from ...types import SourceKey
from ._registry import DatasetSourceFactory
from ._registry import register_source as _register_source

__all__ = ["register_source"]


def register_source(
    source: SourceKey,
    factory: DatasetSourceFactory,
    *,
    operational_load_options: Collection[str] = (),
) -> None:
    _register_source(
        source,
        factory,
        operational_load_options=operational_load_options,
    )
