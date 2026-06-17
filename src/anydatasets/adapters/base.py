from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from anydatasets.cache import CacheManifest
from anydatasets.registry import DatasetSpec


class DatasetAdapter(ABC):
    @abstractmethod
    def prepare(self, spec: DatasetSpec, cache: CacheManifest) -> Any:
        raise NotImplementedError

    @abstractmethod
    def iter_samples(self, manifest: Any) -> Iterator[dict]:
        raise NotImplementedError
