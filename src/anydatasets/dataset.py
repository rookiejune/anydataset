from __future__ import annotations

from typing import Iterator, Mapping, Sequence

from torch.utils.data import IterableDataset

from .adapters import HuggingFaceAdapter, LocalFilesAdapter
from .adapters.base import DatasetAdapter
from .cache import CacheManager
from .mixing import SampleStream, WeightedDatasetMixer
from .registry import DatasetRegistry, DatasetSpec
from .samples import Sample
from .tasks import Task, get_batch_builder


class AnyIterableDataset(IterableDataset):
    def __init__(
        self,
        datasets: Sequence[str],
        task: Task,
        batch_size: int,
        dataset_map: Mapping[str, DatasetSpec] | None = None,
        weights: Mapping[str, float] | None = None,
        cache_dir: str = "~/.cache/anydatasets",
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
    ):
        if not datasets:
            raise ValueError("datasets must contain at least one dataset reference.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not isinstance(task, Task):
            raise TypeError("task must be a Task enum value.")

        super().__init__()
        self.datasets = list(datasets)
        self.task = task
        self.batch_size = batch_size
        self.weights = dict(weights or {})
        self.cache = CacheManager(cache_dir)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.registry = DatasetRegistry(dataset_map)
        self.specs = [self.registry.resolve(dataset) for dataset in self.datasets]

    def __iter__(self):
        builder = get_batch_builder(self.task)
        mixer = WeightedDatasetMixer(self._streams(), seed=self.seed)
        batch: list[Sample] = []

        for sample in mixer:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield builder.build(batch)
                batch = []

        if batch and not self.drop_last:
            yield builder.build(batch)

    def _streams(self) -> list[SampleStream]:
        streams: list[SampleStream] = []
        for spec in self.specs:
            cache = self.cache.prepare(spec)
            adapter = _adapter_for(spec)
            manifest = adapter.prepare(spec, cache)
            iterator = self._wrap_samples(spec, adapter.iter_samples(manifest))
            weight = self.weights.get(spec.key, self.weights.get(spec.name or spec.key, 1.0))
            streams.append(SampleStream(name=spec.key, iterator=iterator, weight=weight))
        return streams

    def _wrap_samples(self, spec: DatasetSpec, rows: Iterator[dict]) -> Iterator[Sample]:
        for index, row in enumerate(rows):
            yield Sample(data=row, dataset_name=spec.key, sample_index=index)


def _adapter_for(spec: DatasetSpec) -> DatasetAdapter:
    if spec.adapter is not None:
        return spec.adapter
    if spec.source == "huggingface":
        return HuggingFaceAdapter()
    if spec.source == "local_files":
        return LocalFilesAdapter()
    raise ValueError(f"No default adapter for source {spec.source!r}.")
