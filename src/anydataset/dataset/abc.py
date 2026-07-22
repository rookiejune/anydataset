from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset, IterableDataset

from .._parallel import iter_indexed_shard as iter_source_indexed_shard
from .._sharding import Shard, runtime_shard, validate_shard
from ..types import Preset, Spec
from ..types._sample import merge as merge_samples
from ..types._sample import select as select_sample
from ..types.item import Modality, Role, View
from ..resolver import resolve_dataset

if TYPE_CHECKING:
    from ..cache import CacheManager
    from ..types.item import Sample, Schema, Transforms
    from .source import DatasetSource


_DEFAULT_MAX_SHARD_SAMPLES = 100_000


class _Base(ABC):
    def __init__(
        self,
        spec: str | Preset | Spec,
        parse_fn: Callable[[Any], Sample] | None = None,
        transforms: Transforms | None = None,
    ) -> None:
        self.spec = resolve_dataset(spec)
        self._cache_manager = None
        self._dataset = None
        self._source: DatasetSource | None = None
        if parse_fn is not None and not callable(parse_fn):
            raise TypeError("parse_fn must be callable or None.")
        self.parse_fn = _identity_sample if parse_fn is None else parse_fn
        self.transforms = None if transforms is None else dict(transforms)

    def prepare(self) -> Any:
        if self._dataset is not None:
            return self._dataset

        cache = self.cache_manager.prepare(self.spec)
        self._dataset = self.source.prepare(self.spec, cache.cache_path)
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

    @property
    def source(self) -> DatasetSource:
        if self._source is None:
            from .source import for_source

            self._source = for_source(self.spec.source)
        return self._source

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache_manager"] = None
        state["_dataset"] = None
        state["_source"] = self.source
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._cache_manager = None
        self._dataset = None

    def __iter__(self) -> Iterator[Sample]:
        shard = runtime_shard()
        yield from self.iter_runtime_shard(shard)

    def transform_sample(self, sample: Sample) -> Sample:
        if self.transforms is None:
            return sample
        transformed = dict(sample)
        for reference, transform in self.transforms.items():
            transformed[reference] = transform(sample[reference])
        return transformed

    def write(
        self,
        output_dir: str | Path,
        *,
        dataset_id: str | None = None,
        split: str | None = None,
        views: tuple[tuple[Role, Modality, View], ...] | None = None,
        max_shard_samples: int = _DEFAULT_MAX_SHARD_SAMPLES,
        num_shards: int = 1,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        dataset_factory: Callable[[], Any] | None = None,
    ) -> Path:
        return _write_dataset(
            self,
            output_dir,
            dataset_id=dataset_id,
            split=self.spec.split if split is None else split,
            views=views,
            max_shard_samples=max_shard_samples,
            num_shards=num_shards,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            dataset_factory=dataset_factory,
        )

    @staticmethod
    def resolve_sample(sample: Sample, schema: Schema) -> Sample:
        return select_sample(sample, schema)


class IterableAnyDataset(_Base, IterableDataset):
    @classmethod
    def preset(
        cls,
        preset: str | Preset,
        split: str | None = None,
        *,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> IterableAnyDataset:
        from ..presets.registry import create_iterable_preset

        return create_iterable_preset(
            Preset(preset),
            split=split,
            transforms=transforms,
            **load_options,
        )

    def iter_rows(self) -> Iterator[Any]:
        yield from self.dataset

    def iter_runtime_shard(self, shard: Shard) -> Iterator[Sample]:
        yield from self.iter_shard(shard.count, shard.index)

    def iter_shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]:
        validate_shard(num_shards, shard_id)
        for row in self.iter_shard_rows(num_shards, shard_id):
            yield self.transform_sample(self.parse_fn(row))

    def iter_shard_rows(self, num_shards: int, shard_id: int) -> Iterator[Any]:
        validate_shard(num_shards, shard_id)
        dataset = self.dataset
        shard = getattr(dataset, "shard", None)
        if callable(shard):
            yield from shard(num_shards=num_shards, index=shard_id)
            return

        yield from _iter_modulo(self.iter_rows(), num_shards, shard_id)

    def iter_indexed_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Sample]]:
        from .source.protocol import native_indexed_shard

        validate_shard(num_shards, shard_id)
        dataset = self.dataset
        indexed = native_indexed_shard(
            self.source,
            dataset,
            num_shards=num_shards,
            shard_id=shard_id,
        )
        if indexed is not None:
            for index, row in indexed:
                yield index, self.transform_sample(self.parse_fn(row))
            return

        for index, row in enumerate(self.iter_rows()):
            if index % num_shards == shard_id:
                yield index, self.transform_sample(self.parse_fn(row))

    def iter_indexed_runtime_shard(self) -> Iterator[tuple[int, Sample]]:
        shard = runtime_shard()
        yield from self.iter_indexed_shard(shard.flat_count, shard.flat_index)


class MapStyleABC(Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Sample:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Sample]:
        shard = runtime_shard()
        yield from self.iter_runtime_shard(shard)

    def merge(self, dataset: Any) -> MergedDataset:
        _validate_map_style_dataset("merge dataset", dataset)
        return MergedDataset(self, dataset)

    def iter_shard(self, num_shards: int, shard_id: int) -> Iterator[Sample]:
        for _index, sample in self.iter_indexed_shard(num_shards, shard_id):
            yield sample

    def iter_indexed_range(
        self,
        start: int,
        stop: int,
    ) -> Iterator[tuple[int, Sample]]:
        if start < 0 or stop < start or stop > len(self):
            raise ValueError("range must satisfy 0 <= start <= stop <= len(dataset).")
        for index in range(start, stop):
            yield index, self[index]

    def iter_indexed_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Sample]]:
        validate_shard(num_shards, shard_id)
        for index in range(shard_id, len(self), num_shards):
            yield index, self[index]

    def iter_indexed_runtime_shard(self) -> Iterator[tuple[int, Sample]]:
        shard = runtime_shard()
        yield from self.iter_indexed_shard(shard.flat_count, shard.flat_index)

    def iter_runtime_shard(self, shard: Shard) -> Iterator[Sample]:
        usable = len(self) // shard.rank_count * shard.rank_count
        if shard.flat_count > 1:
            for index, sample in self.iter_indexed_shard(
                shard.flat_count,
                shard.flat_index,
            ):
                if index < usable:
                    yield sample
            return

        for _index, sample in self.iter_indexed_range(0, usable):
            yield sample

    def write(
        self,
        output_dir: str | Path,
        *,
        dataset_id: str | None = None,
        split: str | None = None,
        views: tuple[tuple[Role, Modality, View], ...] | None = None,
        max_shard_samples: int = _DEFAULT_MAX_SHARD_SAMPLES,
        num_shards: int = 1,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        dataset_factory: Callable[[], Any] | None = None,
    ) -> Path:
        return _write_dataset(
            self,
            output_dir,
            dataset_id=dataset_id,
            split=split,
            views=views,
            max_shard_samples=max_shard_samples,
            num_shards=num_shards,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            dataset_factory=dataset_factory,
        )


@dataclass(frozen=True)
class MergedDataset(MapStyleABC):
    left: Any
    right: Any

    def __post_init__(self) -> None:
        _validate_map_style_dataset("left dataset", self.left)
        _validate_map_style_dataset("right dataset", self.right)
        _validate_lengths(self.left, self.right)

    def __len__(self) -> int:
        return _validate_lengths(self.left, self.right)

    def __getitem__(self, index: int) -> Sample:
        return merge_samples(
            self.left[index],
            self.right[index],
            context=f"Merge sample {index}",
        )


class AnyDataset(_Base, MapStyleABC):
    @classmethod
    def preset(
        cls,
        preset: str | Preset,
        split: str | None = None,
        *,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> AnyDataset:
        from ..presets.registry import create_map_preset

        return create_map_preset(
            Preset(preset),
            split=split,
            transforms=transforms,
            **load_options,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        return self.transform_sample(self.parse_fn(self.dataset[index]))

    def iter_indexed_range(self, start: int, stop: int):
        if start < 0 or stop < start or stop > len(self):
            raise ValueError("range must satisfy 0 <= start <= stop <= len(dataset).")

        dataset = self.dataset
        iter_indexed = getattr(dataset, "iter_indexed_range", None)
        if callable(iter_indexed):
            for index, row in iter_indexed(start, stop):
                yield index, self.transform_sample(self.parse_fn(row))
            return

        for index in range(start, stop):
            yield index, self[index]

    def iter_indexed_shard(self, num_shards: int, shard_id: int):
        for index, row in iter_source_indexed_shard(
            self.dataset,
            num_shards,
            shard_id,
        ):
            yield index, self.transform_sample(self.parse_fn(row))


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


def _write_dataset(
    dataset: Any,
    output_dir: str | Path,
    *,
    dataset_id: str | None,
    split: str | None,
    views: tuple[tuple[Role, Modality, View], ...] | None,
    max_shard_samples: int,
    num_shards: int,
    num_workers: int,
    prefetch_factor: int | None,
    dataset_factory: Callable[[], Any] | None,
) -> Path:
    from .write import DatasetStoreWriter

    writer = DatasetStoreWriter(
        output_dir,
        dataset_id=dataset_id,
        split=split,
        views=views,
        max_shard_samples=max_shard_samples,
        num_shards=num_shards,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    if dataset_factory is not None:
        return writer.write(dataset_factory=dataset_factory)
    return writer.write(dataset)


def _validate_map_style_dataset(name: str, dataset: Any) -> None:
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TypeError(f"{name} must be a map-style dataset.")


def _validate_lengths(left: Any, right: Any) -> int:
    left_len = len(left)
    right_len = len(right)
    if left_len != right_len:
        raise ValueError(
            f"merge datasets must have the same length: {left_len} != {right_len}."
        )
    return left_len
