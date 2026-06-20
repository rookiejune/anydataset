from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, Mapping

if TYPE_CHECKING:
    from anydataset.api.cache import CacheManifest
    from anydataset.api.spec import DatasetSpec


class DatasetAdapter(ABC):
    @abstractmethod
    def prepare(self, spec: DatasetSpec, cache: CacheManifest) -> Any:
        raise NotImplementedError

    @abstractmethod
    def iter_samples(self, manifest: Any) -> Iterator[dict]:
        raise NotImplementedError

    def iter_indexed_samples(
        self,
        manifest: Any,
        num_shards: int = 1,
        shard_id: int = 0,
    ) -> Iterator[tuple[int, dict]]:
        for index, row in enumerate(self.iter_samples(manifest)):
            if index % num_shards == shard_id:
                yield index, row


class TaskSampleAdapter(ABC):
    @abstractmethod
    def adapt(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError
