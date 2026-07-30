from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from torch.utils.data import Dataset, IterableDataset

from .._parallel import iter_indexed_shard as iter_source_indexed_shard
from .._sharding import Shard, runtime_shard, validate_shard
from ..types import Preset, Source, Spec
from ..types._sample import select as select_sample
from ..types.item import Modality, Role, View
from ..resolver import resolve_dataset
from ._shuffle import index_groups, shuffle_index_groups

if TYPE_CHECKING:
    from ..cache import CacheManager
    from ..types.item import Sample, Schema, Transforms
    from .source import DatasetSource
    from torch.utils.data import DataLoader, Sampler


_DEFAULT_MAX_SHARD_SAMPLES = 100_000
_DEFAULT_SHUFFLE_GROUP_SAMPLES = 4096


@runtime_checkable
class _IndexGrouped(Protocol):
    def iter_index_groups(self) -> Iterator[Sequence[int]]: ...


class _RuntimeSharded(Protocol):
    def iter_runtime_shard(self, shard: Shard) -> Iterator[Sample]: ...


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
        vars(self).update(state)
        self._cache_manager = None
        self._dataset = None

    def __iter__(self) -> Iterator[Sample]:
        shard = runtime_shard()
        dataset = cast(_RuntimeSharded, self)
        yield from dataset.iter_runtime_shard(shard)

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
        from .source.protocol import native_indexed_shard

        validate_shard(num_shards, shard_id)
        indexed = native_indexed_shard(
            self.source,
            self.dataset,
            num_shards=num_shards,
            shard_id=shard_id,
        )
        if indexed is not None:
            for _index, row in indexed:
                yield row
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

    def dataloader(
        self,
        *,
        costs: int | Sequence[int],
        max_batch_memory: int,
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
        seed: int = 0,
        epoch: int = 0,
        planning_window: int = 256,
        max_batch_samples: int | None = None,
        drop_distributed_tail: bool = True,
        **loader_kwargs: Any,
    ) -> DataLoader[Any]:
        from .batching import _DataLoader

        return _DataLoader(
            self,
            costs=costs,
            max_batch_memory=max_batch_memory,
            shuffle=shuffle,
            sampler=sampler,
            seed=seed,
            epoch=epoch,
            planning_window=planning_window,
            max_batch_samples=max_batch_samples,
            drop_distributed_tail=drop_distributed_tail,
            **loader_kwargs,
        )

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        if not shuffle:
            yield range(rank, len(self), num_replicas)
            return
        yield from shuffle_index_groups(
            index_groups(len(self), _DEFAULT_SHUFFLE_GROUP_SAMPLES),
            seed=seed,
            epoch=epoch,
            num_replicas=num_replicas,
            rank=rank,
        )

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


class AnyDataset(_Base, MapStyleABC):
    @property
    def selected_store_views(
        self,
    ) -> tuple[tuple[Role, Modality, View], ...] | None:
        if self.spec.source != Source.STORE:
            return None
        from .source.store import StoreSource

        source = self.source
        if not isinstance(source, StoreSource):
            raise TypeError("store datasets require StoreSource.")
        return source.views

    @classmethod
    def from_store(
        cls,
        path: str | Path,
        split: str | None = None,
        *,
        views: tuple[tuple[Role, Modality, View], ...] | None = None,
        transforms: Transforms | None = None,
    ) -> AnyDataset:
        """Open a canonical store while loading only the selected views."""

        if views is not None and not isinstance(views, tuple):
            raise TypeError("views must be a tuple or None.")
        from .source.store import StoreSource

        dataset = cls(
            Spec(source=Source.STORE, path=str(path), split=split),
            transforms=transforms,
        )
        dataset._source = StoreSource(views)
        return dataset

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

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        dataset = self.dataset
        if isinstance(dataset, MapStyleABC):
            yield from dataset._shuffle(
                shuffle=shuffle,
                seed=seed,
                epoch=epoch,
                num_replicas=num_replicas,
                rank=rank,
            )
            return
        if isinstance(dataset, _IndexGrouped):
            yield from shuffle_index_groups(
                dataset.iter_index_groups(),
                shuffle=shuffle,
                seed=seed,
                epoch=epoch,
                num_replicas=num_replicas,
                rank=rank,
            )
            return
        yield from super()._shuffle(
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            num_replicas=num_replicas,
            rank=rank,
        )

    def iter_indexed_range(
        self,
        start: int,
        stop: int,
    ) -> Iterator[tuple[int, Sample]]:
        if start < 0 or stop < start or stop > len(self):
            raise ValueError("range must satisfy 0 <= start <= stop <= len(dataset).")

        dataset = self.dataset
        iter_indexed = getattr(dataset, "iter_indexed_range", None)
        if callable(iter_indexed):
            method = cast(
                Callable[[int, int], Iterator[tuple[int, Any]]],
                iter_indexed,
            )
            for index, row in method(start, stop):
                yield index, self.transform_sample(self.parse_fn(row))
            return

        for index in range(start, stop):
            yield index, self[index]

    def iter_indexed_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Sample]]:
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
    from ..store.writer import DatasetWriter

    writer = DatasetWriter(
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
