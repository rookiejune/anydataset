from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Unpack

from ..dataset.abc import AnyDataset, MapStyleABC, MergedDataset
from ..runtime import Runtime
from ..store.reader import StoreDataset
from ..types import Spec
from ..types.item import Sample
from .apply import (
    _DEFAULT_COMMIT_SAMPLES,
    _DEFAULT_MAX_SHARD_SAMPLES,
    apply_filter,
    filter_base,
    filter_spec,
    filter_universe,
    make_filtered_dataset_factory,
)
from .rules import label, unique_labels, validate_string
from .storage import merged_index, read_metrics
from .types import (
    DatasetFactory,
    FilterApplyKwargs,
    FilterDecision,
    FilterFactory,
    FilterLabel,
    FilterPredicate,
    _Index,
)

_FilterApplyOptions = FilterApplyKwargs

_FILTER_APPLY_DEFAULTS: _FilterApplyOptions = {
    "metrics": False,
    "device": "auto",
    "batch_size": 1,
    "num_workers": 0,
    "prefetch_factor": None,
    "commit_samples": _DEFAULT_COMMIT_SAMPLES,
    "max_shard_samples": _DEFAULT_MAX_SHARD_SAMPLES,
    "write_workers": 1,
    "write_prefetch": None,
    "worker_timeout": None,
    "runtime": Runtime(),
}


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
        "_labels",
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
        metrics_path: Path | None = None,
    ) -> None:
        base = filter_base(base)
        if not isinstance(rule, FilterRule):
            raise TypeError("rule must be a FilterRule.")
        normalized = {key: indices for key, indices in partitions.items()}
        self._base = base
        self._indexes = MappingProxyType(normalized)
        self._labels = tuple(normalized)
        self._counts = MappingProxyType(
            {key: len(indices) for key, indices in normalized.items()}
        )
        self._dataset_factory = dataset_factory
        self._rule = rule
        self._cache_path = Path(cache_path)
        self._metrics_path = None if metrics_path is None else Path(metrics_path)

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
        options = _apply_options(apply_kwargs)
        cache = apply_filter(
            FilterRule(name, factory),
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
    def _from_partitions(
        cls,
        base: AnyDataset | StoreDataset | MergedDataset | FilteredDataset,
        rule: FilterRule,
        cache_path: Path,
        partitions: Mapping[str, _Index],
        labels: Sequence[str],
        *,
        dataset_factory: DatasetFactory,
    ) -> FilteredDataset:
        cache = _FilterCache(
            base,
            partitions,
            rule,
            cache_path,
            dataset_factory=dataset_factory,
        )
        return cls._from_cache(cache, labels=tuple(labels))

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


def selected_labels(
    labels: FilterLabel | Sequence[FilterLabel] | None,
    available: Sequence[str],
) -> tuple[str, ...]:
    if labels is None:
        return tuple(available)
    return normalized_labels(labels)


def _apply_options(kwargs: FilterApplyKwargs) -> _FilterApplyOptions:
    extra = set(kwargs) - set(_FILTER_APPLY_DEFAULTS)
    if extra:
        name = sorted(extra)[0]
        raise TypeError(f"Unexpected filter apply option: {name}.")
    return {
        "metrics": kwargs.get("metrics", _FILTER_APPLY_DEFAULTS["metrics"]),
        "device": kwargs.get("device", _FILTER_APPLY_DEFAULTS["device"]),
        "batch_size": kwargs.get("batch_size", _FILTER_APPLY_DEFAULTS["batch_size"]),
        "num_workers": kwargs.get("num_workers", _FILTER_APPLY_DEFAULTS["num_workers"]),
        "prefetch_factor": kwargs.get(
            "prefetch_factor",
            _FILTER_APPLY_DEFAULTS["prefetch_factor"],
        ),
        "commit_samples": kwargs.get(
            "commit_samples",
            _FILTER_APPLY_DEFAULTS["commit_samples"],
        ),
        "max_shard_samples": kwargs.get(
            "max_shard_samples",
            _FILTER_APPLY_DEFAULTS["max_shard_samples"],
        ),
        "write_workers": kwargs.get(
            "write_workers",
            _FILTER_APPLY_DEFAULTS["write_workers"],
        ),
        "write_prefetch": kwargs.get(
            "write_prefetch",
            _FILTER_APPLY_DEFAULTS["write_prefetch"],
        ),
        "worker_timeout": kwargs.get(
            "worker_timeout",
            _FILTER_APPLY_DEFAULTS["worker_timeout"],
        ),
        "runtime": kwargs.get("runtime", _FILTER_APPLY_DEFAULTS["runtime"]),
    }


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
