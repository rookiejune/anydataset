from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from torch.utils.data import Dataset, IterableDataset

from .. import types
from ..types import Source, Spec

if TYPE_CHECKING:
    from ..cache import CacheManager, CacheManifest
    from ..types.item import Reference, Sample, Schema


type Ref = types.Reference
type Sample = types.Sample
type Schema = types.Schema


def _validate_shard(num_shards: int, shard_id: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards.")


class _Base(ABC):
    def __init__(
        self,
        spec: Spec,
        parse_fn: Callable[[Any], Sample] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.spec = spec
        self._cache_manager = None
        if cache_dir is not None:
            from ..cache import CacheManager

            self._cache_manager = CacheManager(cache_dir)
        self._dataset = None
        self.parse_fn = parse_fn or _identity_sample

    def prepare(self) -> Any:
        if self._dataset is not None:
            return self._dataset

        cache_manifest = self.cache_manager.prepare(self.spec)
        match self.spec.source:
            case Source.HF:
                self._dataset = _prepare_hf(self.spec, cache_manifest)
            case Source.HF_DISK:
                self._dataset = _prepare_hf_disk(self.spec)
            case Source.LOCAL:
                from .source.local_files import prepare_local

                self._dataset = prepare_local(self.spec, cache_manifest)
            case Source.UNIFIED:
                from ..store.reader import read_store_dataset

                self._dataset = read_store_dataset(
                    self.spec.path,
                    split=self.spec.split,
                    cache_path=cache_manifest.cache_path,
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported dataset source: {self.spec.source!r}."
                )
        return self._dataset

    @property
    def cache_manager(self) -> CacheManager:
        if self._cache_manager is None:
            from ..cache import CacheManager

            self._cache_manager = CacheManager()
        return self._cache_manager

    @property
    def dataset(self) -> Any:
        return self.prepare()

    def __iter__(self) -> Iterator[Sample]:
        yield from self.iter_shard(num_shards=1, shard_id=0)

    @abstractmethod
    def iter_shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]:
        raise NotImplementedError

    @staticmethod
    def resolve_sample(sample: Sample, schema: Schema) -> Sample:
        return {
            reference: sample[reference].select_by(requirement)
            for reference, requirement in schema.items()
        }


class IterableAnyDataset(_Base, IterableDataset):
    def __iter__(self) -> Iterator[Sample]:
        rows = self.iter_rows()
        for row in rows:
            yield self.parse_fn(row)

    def iter_rows(self) -> Iterator[Any]:
        if self.spec.source is Source.LOCAL:
            from .source.local_files import iter_local

            yield from iter_local(self.dataset)
            return
        if self.spec.source is Source.UNIFIED:
            yield from self.dataset
            return
        yield from self.dataset

    def iter_shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]:
        _validate_shard(num_shards, shard_id)
        for row in self.iter_shard_rows(num_shards, shard_id):
            yield self.parse_fn(row)

    def iter_shard_rows(self, num_shards: int, shard_id: int) -> Iterator[Any]:
        _validate_shard(num_shards, shard_id)
        dataset = self.dataset
        shard = getattr(dataset, "shard", None)
        if shard is not None:
            yield from shard(num_shards=num_shards, index=shard_id)
            return

        yield from _iter_modulo(self.iter_rows(), num_shards, shard_id)


class AnyDataset(_Base, Dataset):
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        return self.parse_fn(self.dataset[index])

    def iter_shard(self, num_shards: int, shard_id: int):
        _validate_shard(num_shards, shard_id)
        for index in range(shard_id, len(self), num_shards):
            yield self[index]


def _identity_sample(row: Any) -> Sample:
    return row


def _iter_modulo(
    rows: Iterator[Any],
    num_shards: int,
    shard_id: int,
) -> Iterator[Any]:
    for index, row in enumerate(rows):
        if index % num_shards == shard_id:
            yield row


def _prepare_hf(spec: Spec, cache: CacheManifest) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets support requires `pip install anydataset[huggingface]`."
        ) from exc

    split = spec.split or "train"
    load_kwargs = dict(spec.load_options)
    config_name = load_kwargs.pop("config_name", None)
    if config_name is not None:
        if "name" in load_kwargs:
            raise ValueError("Use either `config_name` or `name`, not both.")
        load_kwargs["name"] = config_name
    return load_dataset(
        spec.path,
        split=split,
        cache_dir=str(cache.cache_path),
        **load_kwargs,
    )


def _prepare_hf_disk(spec: Spec) -> Any:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets support requires `pip install anydataset[huggingface]`."
        ) from exc

    dataset = load_from_disk(spec.path, **dict(spec.load_options))
    if not isinstance(dataset, DatasetDict):
        return dataset

    if spec.split is None:
        raise ValueError("huggingface_disk DatasetDict specs must set split.")
    if spec.split not in dataset:
        raise KeyError(f"HuggingFace disk dataset is missing split {spec.split!r}.")
    return dataset[spec.split]
