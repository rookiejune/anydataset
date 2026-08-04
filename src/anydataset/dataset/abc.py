from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from torch.utils.data import Dataset, IterableDataset

from .._runtime.sharding import (
    Shard,
    iter_map_style_range,
    iter_map_style_shard,
    runtime_shard,
    validated_range_rows,
    validate_range,
    validate_shard,
)
from ..types import Preset, Source, Spec
from ..types._sample import select as select_sample
from ..types.item import Modality, Role, View
from ..resolver import resolve_dataset, split_name_and_split
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


@runtime_checkable
class _IndexedRange(Protocol):
    def iter_indexed_range(
        self,
        start: int,
        stop: int,
    ) -> Iterable[tuple[int, Any]]: ...


class _RuntimeSharded(Protocol):
    def iter_runtime_shard(self, shard: Shard) -> Iterator[Sample]: ...


class _DatasetOperations:
    def __iter__(self) -> Iterator[Sample]:
        shard = runtime_shard()
        dataset = cast(_RuntimeSharded, self)
        yield from dataset.iter_runtime_shard(shard)

    def _default_write_split(self) -> str | None:
        return None

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
            split=self._default_write_split() if split is None else split,
            views=views,
            max_shard_samples=max_shard_samples,
            num_shards=num_shards,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            dataset_factory=dataset_factory,
        )


class _Base(_DatasetOperations, ABC):
    def __init__(
        self,
        spec: str | Preset | Spec,
        parse_fn: Callable[[Any], Sample] | None = None,
        transforms: Transforms | None = None,
    ) -> None:
        _reject_preset_input(spec)
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
            from .source._registry import create_source

            self._source = create_source(self.spec.source)
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

    def transform_sample(self, sample: Sample) -> Sample:
        if self.transforms is None:
            return sample
        transformed = dict(sample)
        for reference, transform in self.transforms.items():
            transformed[reference] = transform(sample[reference])
        return transformed

    def _default_write_split(self) -> str | None:
        return self.spec.split

    @staticmethod
    def resolve_sample(sample: Sample, schema: Schema) -> Sample:
        return select_sample(sample, schema)

    def _iter_native_source_shard(
        self,
        dataset: object,
        *,
        num_shards: int,
        shard_id: int,
        sample_count: int | None,
    ) -> Iterator[tuple[int, Sample]] | None:
        from .source.protocol import _native_shard

        rows = _native_shard(
            self.source,
            dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            sample_count=sample_count,
        )
        if rows is None:
            return None
        return self._iter_sample_rows(rows)

    def _iter_sample_rows(
        self,
        rows: Iterator[tuple[int, Any]],
    ) -> Iterator[tuple[int, Sample]]:
        for index, row in rows:
            yield index, self.transform_sample(self.parse_fn(row))


class IterableAnyDataset(_Base, IterableDataset):
    def iter_rows(self) -> Iterator[Any]:
        yield from self.dataset

    def iter_runtime_shard(self, shard: Shard) -> Iterator[Sample]:
        for _index, sample in self.iter_shard(shard.count, shard.index):
            yield sample

    def iter_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Sample]]:
        validate_shard(num_shards, shard_id)
        native = self._iter_native_source_shard(
            self.dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            sample_count=None,
        )
        if native is not None:
            yield from native
            return

        for index, row in enumerate(self.iter_rows()):
            if index % num_shards == shard_id:
                yield index, self.transform_sample(self.parse_fn(row))


class MapStyleABC(_DatasetOperations, Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Sample:
        raise NotImplementedError

    def dataloader(
        self,
        *,
        costs: None | Iterable[int] | Callable[[Any], int],
        max_batch_memory: int,
        cost_aggregation: Literal["sum", "padded_max"] = "sum",
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
        seed: int = 0,
        epoch: int = 0,
        planning_window: int = 256,
        distributed_plan_window: int = 32,
        max_batch_samples: int | None = None,
        max_padding_ratio: float = 0.2,
        drop_distributed_tail: bool = True,
        **loader_kwargs: Any,
    ) -> DataLoader[Any]:
        from .batching import _DataLoader

        return _DataLoader(
            self,
            costs=costs,
            max_batch_memory=max_batch_memory,
            cost_aggregation=cost_aggregation,
            shuffle=shuffle,
            sampler=sampler,
            seed=seed,
            epoch=epoch,
            planning_window=planning_window,
            distributed_plan_window=distributed_plan_window,
            max_batch_samples=max_batch_samples,
            max_padding_ratio=max_padding_ratio,
            drop_distributed_tail=drop_distributed_tail,
            **loader_kwargs,
        )

    def cost_row(self, index: int) -> Any:
        """Return the lightweight row passed to callable dataloader costs."""

        return self[index]

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

    def iter_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Sample]]:
        yield from iter_map_style_shard(self, num_shards, shard_id)

    def iter_indexed_range(
        self,
        start: int,
        stop: int,
    ) -> Iterator[tuple[int, Sample]]:
        yield from iter_map_style_range(self, start, stop)

    def iter_runtime_shard(self, shard: Shard) -> Iterator[Sample]:
        usable = len(self) // shard.rank_count * shard.rank_count
        if shard.flat_count > 1:
            for index, sample in self.iter_shard(
                shard.flat_count,
                shard.flat_index,
            ):
                if index < usable:
                    yield sample
            return

        for _index, sample in self.iter_indexed_range(0, usable):
            yield sample


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
        legacy_policy: str = "reject",
        unsafe_pickle_payloads: bool = False,
    ) -> AnyDataset:
        """Open a canonical store while loading only the selected views."""

        if views is not None and not isinstance(views, tuple):
            raise TypeError("views must be a tuple or None.")
        if type(unsafe_pickle_payloads) is not bool:
            raise TypeError("unsafe_pickle_payloads must be a boolean.")
        from .source.store import StoreSource

        load_options: dict[str, object] = {}
        if legacy_policy != "reject":
            load_options["legacy_policy"] = legacy_policy
        if unsafe_pickle_payloads:
            load_options["unsafe_pickle_payloads"] = True
        dataset = cls(
            Spec(
                source=Source.STORE,
                path=str(path),
                split=split,
                load_options=load_options,
            ),
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

    def cost_row(self, index: int) -> Any:
        dataset = self.dataset
        cost_row = getattr(dataset, "cost_row", None)
        if callable(cost_row):
            return cost_row(index)
        return dataset[index]

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
        validate_range(len(self), start, stop)
        dataset = self.dataset
        if isinstance(dataset, _IndexedRange):
            rows = validated_range_rows(
                dataset.iter_indexed_range(start, stop),
                start=start,
                stop=stop,
                label="Prepared dataset range",
            )
            for index, row in rows:
                yield index, self.transform_sample(self.parse_fn(row))
            return
        for index, row in iter_map_style_range(dataset, start, stop):
            yield index, self.transform_sample(self.parse_fn(row))

    def iter_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Sample]]:
        validate_shard(num_shards, shard_id)
        dataset = self.dataset
        native = self._iter_native_source_shard(
            dataset,
            num_shards=num_shards,
            shard_id=shard_id,
            sample_count=len(dataset),
        )
        if native is not None:
            yield from native
            return

        for index, row in iter_map_style_shard(dataset, num_shards, shard_id):
            yield index, self.transform_sample(self.parse_fn(row))


def _identity_sample(row: Any) -> Sample:
    return row


def _reject_preset_input(spec: str | Preset | Spec) -> None:
    preset: Preset | None = None
    if isinstance(spec, Preset):
        preset = spec
    elif isinstance(spec, str) and "://" not in spec:
        name, _split = split_name_and_split(spec)
        try:
            preset = Preset(name)
        except ValueError:
            pass
    if preset is not None:
        raise TypeError(
            "Preset inputs require AnyDataset.preset(...); pass "
            "preset.spec() explicitly to use only the physical Spec."
        )


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
