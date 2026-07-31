from __future__ import annotations

from ...types import SourceKey
from ._registry import DatasetSourceFactory
from ._registry import register_source as _register_source

__all__ = ["register_source"]


def register_source(source: SourceKey, factory: DatasetSourceFactory) -> None:
    _register_source(source, factory)
