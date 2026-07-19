from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from ..dataset.abc import AnyDataset, MapStyleABC, MergedDataset
from ..store.reader import StoreDataset
from ..types import Spec
from ..types.item import Sample
from ._options import options as apply_options
from .apply import (
    apply_filter,
    filter_base,
    filter_spec,
    filter_universe,
    make_filtered_dataset_factory,
)
from .generations import GenerationLease, lease_filter_generation
from .rules import label, unique_labels, validate_string
from .storage import merged_index, read_metrics, read_partitions
from .types import (
    DatasetFactory,
    FilterApplyKwargs,
    FilterDecision,
    FilterFactory,
    FilterLabel,
    FilterPredicate,
    _Index,
)

class FilterRule:
    __slots__ = ("_factory", "_name")

    def __init__(
        self,
        name: str,
        factory: FilterFactory,
    ) -> None:
        validate_string("name", name)
        if not callable(factory):
            raise TypeError("factory must be callable.")
        self._name = name
        self._factory = factory

    def __repr__(self) -> str:
        return f"FilterRule(name={self.name!r}, factory={self.factory!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FilterRule):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def factory(self) -> FilterFactory:
        return self._factory

    def apply(
        self,
        *,
        dataset_factory: DatasetFactory,
        labels: FilterLabel | Sequence[FilterLabel] | None = None,
        **apply_kwargs: Unpack[FilterApplyKwargs],
    ) -> FilteredDataset:
        return FilteredDataset(
            self.name,
            self.factory,
            dataset_factory=dataset_factory,
            labels=labels,
            **apply_kwargs,
        )


class _FilterCache:
    __slots__ = (
        "_base",
        "_cache_path",
        "_counts",
        "_dataset_factory",
        "_indexes",
        "_input_id",
        "_labels",
        "_lease",
        "_metrics_path",
        "_rule",
    )

    def __init__(
        self,
        base: AnyDataset | StoreDataset | MergedDataset | FilteredDataset,
        partitions: Mapping[str, _Index],
        rule: FilterRule,
        cache_path: Path,
        dataset_factory: DatasetFactory,
        lease: GenerationLease,
        metrics_path: Path | None = None,
        input_id: str | None = None,
    ) -> None:
        base = filter_base(base)
        if not isinstance(rule, FilterRule):
            raise TypeError("rule must be a FilterRule.")
        cache_path = Path(cache_path)
        if not isinstance(lease, GenerationLease):
            raise TypeError("lease must be a GenerationLease.")
        if lease.path != cache_path:
            raise ValueError("lease must pin the filter cache generation.")
        normalized = {key: indices for key, indices in partitions.items()}
        self._base = base
        self._indexes = MappingProxyType(normalized)
        self._labels = tuple(normalized)
        self._counts = MappingProxyType(
            {key: len(indices) for key, indices in normalized.items()}
        )
        self._dataset_factory = dataset_factory
        self._rule = rule
        self._cache_path = cache_path
        self._lease = lease
        self._metrics_path = None if metrics_path is None else Path(metrics_path)
        self._input_id = input_id

    def __repr__(self) -> str:
        return (
            "_FilterCache("
            f"base={self.base!r}, "
            f"labels={self.labels!r}, "
            f"rule={self.rule!r}, "
            f"cache_path={self.cache_path!r}, "
            f"metrics_path={self.metrics_path!r}"
            ")"
        )

    def __reduce__(self):
        return (
            _restore_filter_cache,
            (
                self.dataset_factory,
                self.rule.name,
                self.cache_path,
                self.metrics_path,
                self.input_id,
            ),
        )

    @property
    def base(self) -> AnyDataset | StoreDataset | MergedDataset | FilteredDataset:
        return self._base

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def counts(self) -> MappingProxyType[str, int]:
        return self._counts

    @property
    def partitions(self) -> MappingProxyType[str, tuple[int, ...]]:
        return MappingProxyType(
            {key: tuple(indices) for key, indices in self._indexes.items()}
        )

    @property
    def rule(self) -> FilterRule:
        return self._rule

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    @property
    def metrics_path(self) -> Path | None:
        return self._metrics_path

    @property
    def input_id(self) -> str | None:
        return self._input_id

    @property
    def dataset_factory(self) -> DatasetFactory:
        return self._dataset_factory

    @property
    def spec(self) -> Spec:
        return filter_spec(self.base)

    def iter_metrics(self) -> Iterable[Mapping[str, Any]]:
        if self.metrics_path is None:
            raise ValueError("filtered dataset does not include metrics.")
        yield from read_metrics(self.metrics_path)


class FilteredDataset(MapStyleABC):
    __slots__ = ("_cache", "_dataset_factory", "_indices", "_labels")

    def __init__(
        self,
        name: str,
        factory: FilterFactory,
        *,
        dataset_factory: DatasetFactory,
        labels: FilterLabel | Sequence[FilterLabel] | None = None,
        **apply_kwargs: Unpack[FilterApplyKwargs],
    ) -> None:
        normalized = None if labels is None else normalized_labels(labels)
        options = apply_options(apply_kwargs)
        cache = apply_filter(
            FilterRule(name, factory),
            input_id=options["input_id"],
            metrics=options["metrics"],
            device=options["device"],
            batch_size=options["batch_size"],
            num_workers=options["num_workers"],
            prefetch_factor=options["prefetch_factor"],
            commit_samples=options["commit_samples"],
            max_shard_samples=options["max_shard_samples"],
            write_workers=options["write_workers"],
            write_prefetch=options["write_prefetch"],
            worker_timeout=options["worker_timeout"],
            runtime=options["runtime"],
            dataset_factory=dataset_factory,
        )
        self._bind_cache(cache, labels=normalized)

    def _bind_cache(
        self,
        cache: _FilterCache,
        *,
        labels: tuple[str, ...] | None,
    ) -> None:
        normalized = selected_labels(labels, cache.labels)
        self._cache = cache
        self._indices = selected_index(cache._indexes, normalized)
        self._labels = normalized
        self._dataset_factory = make_filtered_dataset_factory(
            cache.dataset_factory,
            cache.rule,
            normalized,
            cache.cache_path,
            cache.metrics_path,
            cache.input_id,
        )

    @classmethod
    def _from_cache(
        cls,
        cache: _FilterCache,
        *,
        labels: FilterLabel | Sequence[FilterLabel] | None = None,
    ) -> FilteredDataset:
        instance = cls.__new__(cls)
        normalized = None if labels is None else normalized_labels(labels)
        instance._bind_cache(cache, labels=normalized)
        return instance

    @classmethod
    def _from_generation(
        cls,
        base: AnyDataset | StoreDataset | MergedDataset | FilteredDataset,
        rule: FilterRule,
        cache_path: Path,
        labels: Sequence[str],
        *,
        dataset_factory: DatasetFactory,
        metrics_path: Path | None = None,
        input_id: str | None = None,
    ) -> FilteredDataset:
        generation = lease_filter_generation(cache_path)
        try:
            cache = _FilterCache(
                base,
                read_partitions(generation.path),
                rule,
                generation.path,
                lease=generation.lease,
                dataset_factory=dataset_factory,
                metrics_path=metrics_path,
                input_id=input_id,
            )
            return cls._from_cache(cache, labels=tuple(labels))
        except Exception:
            generation.lease.close()
            raise

    def __repr__(self) -> str:
        return (
            "FilteredDataset("
            f"base={self.base!r}, "
            f"count={len(self)}, "
            f"labels={self.labels!r}, "
            f"rule={self.rule!r}, "
            f"cache_path={self.cache_path!r}"
            ")"
        )

    def __reduce__(self):
        return _restore_filtered_dataset, (self._cache, self.labels)

    @property
    def base(self) -> AnyDataset | StoreDataset | MergedDataset | FilteredDataset:
        return self._cache.base

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(self._indices)

    def global_index(self, index: int) -> int:
        return int(self._indices[index])

    @property
    def available_labels(self) -> tuple[str, ...]:
        return self._cache.labels

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def counts(self) -> MappingProxyType[str, int]:
        return MappingProxyType(
            {key: self._cache.counts.get(key, 0) for key in self.labels}
        )

    @property
    def available_counts(self) -> MappingProxyType[str, int]:
        return self._cache.counts

    @property
    def rule(self) -> FilterRule:
        return self._cache.rule

    @property
    def cache_path(self) -> Path:
        return self._cache.cache_path

    @property
    def metrics_path(self) -> Path | None:
        return self._cache.metrics_path

    @property
    def input_id(self) -> str | None:
        return self._cache.input_id

    @property
    def dataset_factory(self) -> DatasetFactory:
        return self._dataset_factory

    @property
    def spec(self) -> Spec:
        return filter_spec(self.base)

    def select_by(self, *labels: FilterLabel) -> FilteredDataset:
        return type(self)._from_cache(self._cache, labels=labels)

    def iter_metrics(self) -> Iterable[Mapping[str, Any]]:
        yield from self._cache.iter_metrics()

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Sample:
        return filter_universe(self.base)[self._indices[index]]


def _restore_filter_cache(
    dataset_factory: DatasetFactory,
    rule_name: str,
    cache_path: Path,
    metrics_path: Path | None,
    input_id: str | None,
) -> _FilterCache:
    generation = lease_filter_generation(cache_path)
    try:
        return _FilterCache(
            dataset_factory(),
            read_partitions(generation.path),
            FilterRule(rule_name, _unavailable_filter_factory),
            generation.path,
            lease=generation.lease,
            dataset_factory=dataset_factory,
            metrics_path=metrics_path,
            input_id=input_id,
        )
    except Exception:
        generation.lease.close()
        raise


def _restore_filtered_dataset(
    cache: _FilterCache,
    labels: tuple[str, ...],
) -> FilteredDataset:
    return FilteredDataset._from_cache(cache, labels=labels)


def _unavailable_filter_factory() -> FilterPredicate:
    raise RuntimeError("cached filtered dataset cannot rebuild its upstream rule.")


def selected_labels(
    labels: FilterLabel | Sequence[FilterLabel] | None,
    available: Sequence[str],
) -> tuple[str, ...]:
    if labels is None:
        return tuple(available)
    return normalized_labels(labels)


def normalized_labels(labels: FilterLabel | Sequence[FilterLabel]) -> tuple[str, ...]:
    if isinstance(labels, (bool, str, Enum)):
        values = (labels,)
    else:
        values = tuple(labels)
    if not values:
        raise ValueError("labels must not be empty.")
    return unique_labels(label(value) for value in values)


def selected_index(
    partitions: Mapping[str, _Index],
    labels: Sequence[str],
) -> _Index:
    indexes = tuple(partitions[key] for key in labels if key in partitions)
    if len(indexes) == 1:
        return indexes[0]
    return merged_index(indexes)


__all__ = [
    "DatasetFactory",
    "FilterDecision",
    "FilterFactory",
    "FilteredDataset",
    "FilterLabel",
    "FilterPredicate",
    "FilterRule",
]
