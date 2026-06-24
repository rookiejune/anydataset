from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from torch.utils.data import IterableDataset, get_worker_info

from ..adapters import DatasetAdapter, HuggingFaceAdapter, LocalFilesAdapter, UnifiedDatasetAdapter
from ..adapters.catalog import DEFAULT_ADAPTER_MAP, AdapterBinding
from ..samples import Sample
from ..tasks import Task, get_task_adapter

from .cache import CacheManager
from .resolver import DatasetRef, DatasetResolver
from .spec import DatasetSpec
from .strategy import IterationStrategy, SequentialStrategy


_SHARD_UNSET = object()

type AdapterMap = Mapping[str, AdapterBinding]


class DatasetSource(IterableDataset):
    def __init__(
        self,
        spec: DatasetSpec,
        task: Task,
        *,
        cache_dir: str | Path = "~/.cache/anydataset",
        adapter: DatasetAdapter | None = None,
        dataset_name: str | None = None,
        shard: "_ShardInfo | None" = None,
    ):
        if not isinstance(spec, DatasetSpec):
            raise TypeError("spec must be a DatasetSpec.")
        if not isinstance(task, Task):
            raise TypeError("task must be a Task enum value.")
        if adapter is not None and not isinstance(adapter, DatasetAdapter):
            raise TypeError("adapter must be a DatasetAdapter.")

        super().__init__()
        self.task = task
        self.spec = spec
        self.adapter = adapter if adapter is not None else _adapter_for(spec)
        self._name = dataset_name or spec.key
        self.cache = CacheManager(cache_dir)
        self._shard = shard
        if self._shard is not None:
            _validate_shard(self._shard.num_shards, self._shard.shard_id)

    @property
    def name(self) -> str:
        return self._name

    def __iter__(self) -> Iterator[Sample]:
        self._ensure_distributed_shard()
        yield from self._iter_samples(_worker_shard(self._base_shard()))

    def shard(self, num_shards: int, shard_id: int) -> "DatasetSource":
        _validate_shard(num_shards, shard_id)
        return self._clone(shard=_ShardInfo(num_shards=num_shards, shard_id=shard_id))

    def matches(self, dataset_ref: str) -> bool:
        return dataset_ref in {self.name, self.spec.key, self.spec.name}

    def _clone(self, shard: Any = _SHARD_UNSET) -> "DatasetSource":
        clone_shard = self._shard if shard is _SHARD_UNSET else shard
        return DatasetSource(
            spec=self.spec,
            task=self.task,
            cache_dir=self.cache.root,
            adapter=self.adapter,
            dataset_name=self.name,
            shard=clone_shard,
        )

    def _base_shard(self) -> "_ShardInfo":
        return self._shard or _ShardInfo(num_shards=1, shard_id=0)

    def _iter_samples(self, shard: "_ShardInfo") -> Iterator[Sample]:
        cache = self.cache.prepare(self.spec)
        adapter = self.adapter
        manifest = self._prepare_manifest(cache, adapter)
        rows = adapter.iter_indexed_samples(
            manifest,
            num_shards=shard.num_shards,
            shard_id=shard.shard_id,
        )
        yield from self._wrap_samples(rows, adapter)

    def _prepare_manifest(self, cache, adapter: DatasetAdapter):
        if not self.cache.is_ready(cache):
            with self.cache.prepare_lock(cache):
                if not self.cache.is_ready(cache):
                    manifest = adapter.prepare(self.spec, cache)
                    self.cache.mark_ready(cache)
                    return manifest
        return adapter.prepare(self.spec, cache)

    def _wrap_samples(
        self,
        rows: Iterator[tuple[int, dict]],
        adapter: DatasetAdapter,
    ) -> Iterator[Sample]:
        task_adapter = _task_adapter_for(self.task)
        for index, row in rows:
            data = dict(row)
            if task_adapter is not None:
                data = dict(task_adapter.adapt(data, adapter))
            sample = Sample(data=data, dataset_name=self.name, sample_index=index)
            yield sample

    def _ensure_distributed_shard(self) -> None:
        if _distributed_world_size() > 1 and self._shard is None:
            raise ValueError("DDP iteration requires an explicitly sharded dataset.")


class AnyDataset(IterableDataset):
    def __init__(
        self,
        datasets: DatasetRef | Sequence[DatasetRef] | Sequence[DatasetSource],
        task: Task | None = None,
        dataset_map: Mapping[str, DatasetSpec] | None = None,
        adapter_map: AdapterMap | None = None,
        cache_dir: str | Path | None = None,
        strategy: IterationStrategy | None = None,
    ):
        if strategy is not None and not isinstance(strategy, IterationStrategy):
            raise TypeError("strategy must be an IterationStrategy.")

        super().__init__()
        self.strategy = strategy or SequentialStrategy()
        self.dataset_map = dict(dataset_map or {})
        self.adapter_map = _adapter_map(adapter_map)
        self.resolver = DatasetResolver(self.dataset_map)
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else Path("~/.cache/anydataset").expanduser()
        )

        dataset_values = _dataset_values(datasets)
        if not dataset_values:
            raise ValueError("datasets must contain at least one dataset.")

        if all(isinstance(dataset, DatasetSource) for dataset in dataset_values):
            if dataset_map:
                raise ValueError("dataset_map does not apply to pre-built DatasetSource values.")
            if adapter_map:
                raise ValueError("adapter_map does not apply to pre-built DatasetSource values.")
            if cache_dir is not None:
                raise ValueError("cache_dir does not apply to pre-built DatasetSource values.")
            self.datasets = list(dataset_values)
            self.task = task or self.datasets[0].task
            for dataset in self.datasets:
                if dataset.task is not self.task:
                    raise ValueError("All DatasetSource values must use the same task.")
        elif any(isinstance(dataset, DatasetSource) for dataset in dataset_values):
            raise TypeError("datasets must be all refs/specs or all DatasetSource values.")
        else:
            if not isinstance(task, Task):
                raise TypeError("task must be a Task enum value.")
            self.task = task
            self.datasets = [self._source_for(dataset, task) for dataset in dataset_values]

        self.specs = [dataset.spec for dataset in self.datasets]

    def __iter__(self) -> Iterator[Sample]:
        yield from self.strategy.iter(self.datasets)

    def __getitem__(self, dataset_ref: str) -> DatasetSource:
        for dataset in self.datasets:
            if dataset.matches(dataset_ref):
                return dataset
        raise KeyError(f"Dataset {dataset_ref!r} is not part of this dataset.")

    def shard(self, num_shards: int, shard_id: int) -> "AnyDataset":
        _validate_shard(num_shards, shard_id)
        return AnyDataset(
            datasets=[
                dataset.shard(num_shards=num_shards, shard_id=shard_id)
                for dataset in self.datasets
            ],
            task=self.task,
            strategy=self.strategy,
        )

    def _source_for(
        self,
        dataset: DatasetRef,
        task: Task,
    ) -> DatasetSource:
        if isinstance(dataset, DatasetSpec):
            spec = dataset
            dataset_name = spec.key
        else:
            spec = self.resolver.resolve(dataset)
            dataset_name = dataset
        return DatasetSource(
            spec=spec,
            task=task,
            cache_dir=self.cache_dir,
            adapter=_adapter_for(spec, self.adapter_map),
            dataset_name=dataset_name,
        )


def _adapter_for(
    spec: DatasetSpec,
    adapter_map: AdapterMap | None = None,
) -> DatasetAdapter:
    bindings = DEFAULT_ADAPTER_MAP if adapter_map is None else adapter_map
    if spec.name in bindings:
        return _adapter_from_binding(spec, bindings[spec.name])
    if spec.source == "huggingface":
        return HuggingFaceAdapter()
    if spec.source == "local_files":
        return LocalFilesAdapter()
    if spec.source == "unified":
        return UnifiedDatasetAdapter()
    raise ValueError(f"No default adapter for source {spec.source!r}.")


def _adapter_map(adapter_map: AdapterMap | None) -> dict[str, AdapterBinding]:
    merged = dict(DEFAULT_ADAPTER_MAP)
    if adapter_map:
        for name, binding in adapter_map.items():
            _validate_adapter_binding(name, binding)
        merged.update(adapter_map)
    return merged


def _adapter_from_binding(
    spec: DatasetSpec,
    binding: AdapterBinding,
) -> DatasetAdapter:
    if isinstance(binding, DatasetAdapter):
        return binding
    adapter = binding(spec)
    if not isinstance(adapter, DatasetAdapter):
        raise TypeError(f"adapter_map factory for {spec.name!r} must return a DatasetAdapter.")
    return adapter


def _validate_adapter_binding(name: str, binding: AdapterBinding) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("adapter_map keys must be non-empty dataset names.")
    if isinstance(binding, DatasetAdapter):
        return
    if not callable(binding):
        raise TypeError("adapter_map values must be DatasetAdapter instances or factories.")


def _task_adapter_for(task: Task):
    try:
        return get_task_adapter(task)
    except ValueError:
        return None


def _dataset_values(
    datasets: DatasetRef | Sequence[DatasetRef] | Sequence[DatasetSource],
) -> list[DatasetRef | DatasetSource]:
    if isinstance(datasets, (str, DatasetSpec, DatasetSource)):
        return [datasets]
    return list(datasets)


@dataclass(frozen=True)
class _ShardInfo:
    num_shards: int
    shard_id: int


def _worker_shard(base: _ShardInfo) -> _ShardInfo:
    worker = get_worker_info()
    if worker is None:
        return base
    return _ShardInfo(
        num_shards=base.num_shards * worker.num_workers,
        shard_id=base.shard_id * worker.num_workers + worker.id,
    )


def _validate_shard(num_shards: int, shard_id: int | None) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_id is None or shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards.")


def _distributed_world_size() -> int:
    return _distributed_rank_world_size()[1]


def _distributed_rank_world_size() -> tuple[int, int]:
    try:
        import torch.distributed as dist
    except ImportError:
        dist = None

    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be positive.")
    if rank < 0 or rank >= world_size:
        raise ValueError("RANK must satisfy 0 <= RANK < WORLD_SIZE.")
    return rank, world_size
