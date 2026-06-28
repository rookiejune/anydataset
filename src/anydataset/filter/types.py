from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from ..types.item import Sample

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type FilterLabel = bool | str | Enum
type _Index = Sequence[int]


@dataclass(frozen=True)
class FilterDecision:
    label: FilterLabel
    metrics: Mapping[str, JsonValue]


type FilterOutput = FilterLabel | FilterDecision
type FilterPredicate = Callable[[Sample], FilterOutput]
type FilterFactory = Callable[[], FilterPredicate]


@dataclass(frozen=True)
class _FilterMetricsRow:
    index: int
    label: str
    metrics: Mapping[str, JsonValue]


@dataclass(frozen=True)
class _FilterDecision:
    label: str
    metrics: Mapping[str, JsonValue] | None


@dataclass(frozen=True)
class _FilterChunk:
    partitions: Mapping[str, Sequence[int]]
    metrics: Sequence[_FilterMetricsRow]
