from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from torch.utils.data import IterableDataset, get_worker_info

from anydataset.datasets.base import DatasetAdapter
from anydataset.datasets.huggingface import HuggingFaceDataset
from anydataset.datasets.local_files import LocalFilesDataset
from anydataset.datasets.task_adapters import (
    TaskAdapterRegistry,
    default_task_adapter_registry,
)
from anydataset.samples import Sample
from anydataset.tasks import SampleFormatter, Task

from .cache import CacheManager
from .resolver import DatasetResolver
from .spec import DatasetSpec
from .strategy import IterationStrategy, SequentialStrategy


_SHARD_UNSET = object()
SampleFormatterLike = SampleFormatter | Callable[[Sample], Sample]


class DatasetSource(IterableDataset):
    def __init__(
        self,
        spec: DatasetSpec,
        task: Task,
        *,
        formatter: SampleFormatterLike | None = None,
        cache_dir: str | Path = "~/.cache/anydataset",
        task_adapter_registry: TaskAdapterRegistry | None = None,
        shard: "_ShardInfo | None" = None,
    ):
        if not isinstance(spec, DatasetSpec):
            raise TypeError("spec must be a DatasetSpec.")
        if not isinstance(task, Task):
            raise TypeError("task must be a Task enum value.")
        if formatter is not None and not callable(formatter):
            raise TypeError("formatter must be callable.")
        if task_adapter_registry is not None and not isinstance(
            task_adapter_registry,
            TaskAdapterRegistry,
        ):
            raise TypeError("task_adapter_registry must be a TaskAdapterRegistry.")

        super().__init__()
        self.task = task
        self.spec = spec
        self.formatter = formatter
        self.cache = CacheManager(cache_dir)
        self.task_adapter_registry = (
            task_adapter_registry
            if task_adapter_registry is not None
            else default_task_adapter_registry()
        )
        self._shard = shard
        if self._shard is not None:
            _validate_shard(self._shard.num_shards, self._shard.shard_id)

    @property
    def name(self) -> str:
        return self.spec.key

    def __iter__(self) -> Iterator[Sample]:
        self._ensure_distributed_shard()
        yield from self._iter_samples(_worker_shard(self._base_shard()))

    def shard(self, num_shards: int, shard_id: int) -> "DatasetSource":
        _validate_shard(num_shards, shard_id)
        return self._clone(shard=_ShardInfo(num_shards=num_shards, shard_id=shard_id))

    def matches(self, dataset_ref: str) -> bool:
        return dataset_ref in {self.spec.key, self.spec.ref, self.spec.name}

    def _clone(self, shard: Any = _SHARD_UNSET) -> "DatasetSource":
        clone_shard = self._shard if shard is _SHARD_UNSET else shard
        return DatasetSource(
            spec=self.spec,
            task=self.task,
            formatter=self.formatter,
            cache_dir=self.cache.root,
            task_adapter_registry=self.task_adapter_registry,
            shard=clone_shard,
        )

    def _base_shard(self) -> "_ShardInfo":
        return self._shard or _ShardInfo(num_shards=1, shard_id=0)

    def _iter_samples(self, shard: "_ShardInfo") -> Iterator[Sample]:
        cache = self.cache.prepare(self.spec)
        adapter = _adapter_for(self.spec)
        manifest = self._prepare_manifest(cache, adapter)
        rows = adapter.iter_indexed_samples(
            manifest,
            num_shards=shard.num_shards,
            shard_id=shard.shard_id,
        )
        yield from self._wrap_samples(rows)

    def _prepare_manifest(self, cache, adapter: DatasetAdapter):
        if not self.cache.is_ready(cache):
            with self.cache.prepare_lock(cache):
                if not self.cache.is_ready(cache):
                    manifest = adapter.prepare(self.spec, cache)
                    self.cache.mark_ready(cache)
                    return manifest
        return adapter.prepare(self.spec, cache)

    def _wrap_samples(self, rows: Iterator[tuple[int, dict]]) -> Iterator[Sample]:
        task_adapter = self.task_adapter_registry.resolve(self.spec, self.task)
        for index, row in rows:
            data = dict(self.spec.sample_metadata)
            data.update(row)
            if task_adapter is not None:
                data = dict(task_adapter.adapt(data))
            sample = Sample(data=data, dataset_name=self.spec.key, sample_index=index)
            yield self._format_sample(sample)

    def _format_sample(self, sample: Sample) -> Sample:
        if self.formatter is None:
            return sample
        formatted = self.formatter(sample)
        if not isinstance(formatted, Sample):
            raise TypeError("formatter must return a Sample.")
        return formatted

    def _ensure_distributed_shard(self) -> None:
        if _distributed_world_size() > 1 and self._shard is None:
            raise ValueError("DDP iteration requires an explicitly sharded dataset.")


class AnyDataset(IterableDataset):
    def __init__(
        self,
        datasets: Sequence[str | DatasetSpec | DatasetSource],
        task: Task | None = None,
        dataset_map: Mapping[str, DatasetSpec] | None = None,
        formatter: SampleFormatterLike | None = None,
        cache_dir: str | Path | None = None,
        task_adapter_registry: TaskAdapterRegistry | None = None,
        strategy: IterationStrategy | None = None,
    ):
        if not datasets:
            raise ValueError("datasets must contain at least one dataset.")
        if formatter is not None and not callable(formatter):
            raise TypeError("formatter must be callable.")
        if strategy is not None and not isinstance(strategy, IterationStrategy):
            raise TypeError("strategy must be an IterationStrategy.")
        if task_adapter_registry is not None and not isinstance(
            task_adapter_registry,
            TaskAdapterRegistry,
        ):
            raise TypeError("task_adapter_registry must be a TaskAdapterRegistry.")

        super().__init__()
        self.strategy = strategy or SequentialStrategy()
        self.dataset_map = dict(dataset_map or {})
        self.resolver = DatasetResolver(self.dataset_map)
        self.formatter = formatter
        self.task_adapter_registry = (
            task_adapter_registry
            if task_adapter_registry is not None
            else default_task_adapter_registry()
        )
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else Path("~/.cache/anydataset").expanduser()
        )

        if all(isinstance(dataset, DatasetSource) for dataset in datasets):
            if dataset_map:
                raise ValueError("dataset_map does not apply to pre-built DatasetSource values.")
            if formatter is not None:
                raise ValueError("formatter does not apply to pre-built DatasetSource values.")
            if cache_dir is not None:
                raise ValueError("cache_dir does not apply to pre-built DatasetSource values.")
            if task_adapter_registry is not None:
                raise ValueError(
                    "task_adapter_registry does not apply to pre-built DatasetSource values."
                )
            self.datasets = list(datasets)
            self.task = task or self.datasets[0].task
            for dataset in self.datasets:
                if dataset.task is not self.task:
                    raise ValueError("All DatasetSource values must use the same task.")
        elif any(isinstance(dataset, DatasetSource) for dataset in datasets):
            raise TypeError("datasets must be all refs/specs or all DatasetSource values.")
        else:
            if not isinstance(task, Task):
                raise TypeError("task must be a Task enum value.")
            self.task = task
            self.datasets = [
                DatasetSource(
                    spec=dataset if isinstance(dataset, DatasetSpec) else self.resolver.resolve(dataset),
                    task=task,
                    formatter=formatter,
                    cache_dir=self.cache_dir,
                    task_adapter_registry=self.task_adapter_registry,
                )
                for dataset in datasets
            ]

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

def _adapter_for(spec: DatasetSpec) -> DatasetAdapter:
    if spec.adapter is not None:
        return spec.adapter
    if spec.source == "huggingface":
        return HuggingFaceDataset()
    if spec.source == "local_files":
        return LocalFilesDataset()
    raise ValueError(f"No default adapter for source {spec.source!r}.")


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
