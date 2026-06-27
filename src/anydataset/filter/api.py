from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .._sharding import validate_shard
from ..dataset.abc import AnyDataset, SampleDataset
from ..store.reader import StoreDataset
from ..types import Spec
from ..types.item import Sample
from .apply import (
    _DEFAULT_COMMIT_SAMPLES,
    _DEFAULT_MAX_SHARD_SAMPLES,
    apply_filter,
    ensure_filter,
    filter_base,
    filter_spec,
)
from .rules import (
    label,
    unique_labels,
    validate_string,
)
from .storage import index_sequence, merged_index, read_metrics, read_partitions
from .types import FilterDecision, FilterLabel, FilterPredicate, _Index


class FilterRule:
    __slots__ = ("_name", "_predicate")

    def __init__(
        self,
        name: str,
        predicate: FilterPredicate,
    ) -> None:
        validate_string("name", name)
        if not callable(predicate):
            raise TypeError("predicate must be callable.")
        self._name = name
        self._predicate = predicate

    def __repr__(self) -> str:
        return (
            "FilterRule("
            f"name={self.name!r}, "
            f"predicate={self.predicate!r}"
            ")"
        )

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
    def predicate(self) -> FilterPredicate:
        return self._predicate

    def apply(
        self,
        dataset: AnyDataset | StoreDataset,
        *,
        metrics: bool = False,
        num_workers: int = 1,
        commit_samples: int = _DEFAULT_COMMIT_SAMPLES,
        max_shard_samples: int | None = _DEFAULT_MAX_SHARD_SAMPLES,
        cache_root: str | Path | None = None,
    ) -> FilterResult:
        return apply_filter(
            dataset,
            self,
            metrics=metrics,
            num_workers=num_workers,
            commit_samples=commit_samples,
            max_shard_samples=max_shard_samples,
            cache_root=cache_root,
        )


class FilterResult:
    __slots__ = (
        "_base",
        "_cache_path",
        "_counts",
        "_indexes",
        "_labels",
        "_metrics_path",
        "_rule",
    )

    def __init__(
        self,
        base: AnyDataset | StoreDataset,
        partitions: Mapping[str, _Index],
        rule: FilterRule,
        cache_path: Path,
        metrics_path: Path | None = None,
    ) -> None:
        base = filter_base(base)
        if not isinstance(rule, FilterRule):
            raise TypeError("rule must be a FilterRule.")
        normalized = {key: index_sequence(indices) for key, indices in partitions.items()}
        self._base = base
        self._indexes = MappingProxyType(normalized)
        self._labels = tuple(normalized)
        self._counts = MappingProxyType(
            {key: len(indices) for key, indices in normalized.items()}
        )
        self._rule = rule
        self._cache_path = Path(cache_path)
        self._metrics_path = None if metrics_path is None else Path(metrics_path)

    def __repr__(self) -> str:
        return (
            "FilterResult("
            f"base={self.base!r}, "
            f"labels={self.labels!r}, "
            f"rule={self.rule!r}, "
            f"cache_path={self.cache_path!r}, "
            f"metrics_path={self.metrics_path!r}"
            ")"
        )

    @property
    def base(self) -> AnyDataset | StoreDataset:
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
    def spec(self) -> Spec:
        return filter_spec(self.base)

    def select(self, *labels: FilterLabel) -> FilteredDataset:
        normalized = normalized_labels(labels)
        return FilteredDataset._from_partitions(
            self.base,
            self.rule,
            self.cache_path,
            self._indexes,
            normalized,
        )

    def iter_metrics(self) -> Iterable[Mapping[str, Any]]:
        if self.metrics_path is None:
            raise ValueError("filter result does not include metrics.")
        yield from read_metrics(self.metrics_path)


class FilteredDataset(SampleDataset):
    __slots__ = ("_base", "_cache_path", "_indices", "_labels", "_rule")

    def __init__(
        self,
        base: AnyDataset | StoreDataset,
        rule: FilterRule,
        *,
        labels: FilterLabel | Sequence[FilterLabel],
        num_workers: int = 1,
        commit_samples: int = _DEFAULT_COMMIT_SAMPLES,
        max_shard_samples: int | None = _DEFAULT_MAX_SHARD_SAMPLES,
        cache_root: str | Path | None = None,
    ) -> None:
        base = filter_base(base)
        if not isinstance(rule, FilterRule):
            raise TypeError("rule must be a FilterRule.")
        normalized = normalized_labels(labels)
        cache_path, _ = ensure_filter(
            base,
            rule,
            metrics=False,
            num_workers=num_workers,
            commit_samples=commit_samples,
            max_shard_samples=max_shard_samples,
            cache_root=cache_root,
        )
        partitions = read_partitions(cache_path)
        self._base = base
        self._indices = selected_index(partitions, normalized)
        self._labels = normalized
        self._rule = rule
        self._cache_path = cache_path

    @classmethod
    def _from_partitions(
        cls,
        base: AnyDataset | StoreDataset,
        rule: FilterRule,
        cache_path: Path,
        partitions: Mapping[str, _Index],
        labels: Sequence[str],
    ) -> FilteredDataset:
        instance = cls.__new__(cls)
        instance._base = base
        instance._indices = selected_index(partitions, labels)
        instance._labels = tuple(labels)
        instance._rule = rule
        instance._cache_path = Path(cache_path)
        return instance

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
    def base(self) -> AnyDataset | StoreDataset:
        return self._base

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(self._indices)

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def rule(self) -> FilterRule:
        return self._rule

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    @property
    def spec(self) -> Spec:
        return filter_spec(self.base)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Sample:
        return self.base[self._indices[index]]

    def iter_shard(self, num_shards: int, shard_id: int):
        validate_shard(num_shards, shard_id)
        for index in range(shard_id, len(self), num_shards):
            yield self[index]


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
    return index_sequence(merged_index(indexes))


__all__ = [
    "FilterDecision",
    "FilteredDataset",
    "FilterLabel",
    "FilterPredicate",
    "FilterResult",
    "FilterRule",
]
