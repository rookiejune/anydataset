from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from ..modalities import ModalityRole

if TYPE_CHECKING:
    from ..api.cache import CacheManifest
    from ..api.spec import DatasetSpec


class MissingModalityError(KeyError):
    def __init__(self, modality: str, role: ModalityRole = None):
        suffix = "" if role is None else f" role {role!r}"
        super().__init__(f"Dataset adapter does not provide {modality!r}{suffix}.")
        self.modality = modality
        self.role = role


class DatasetAdapter(ABC):
    @abstractmethod
    def prepare(self, spec: "DatasetSpec", cache: "CacheManifest") -> Any:
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

    def audio(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        raise MissingModalityError("audio", role)

    def text(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        raise MissingModalityError("text", role)


ModalityAdapter = DatasetAdapter
