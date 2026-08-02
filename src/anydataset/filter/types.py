from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, Union, runtime_checkable

from .._compat import NotRequired
from .._runtime.devices import Devices
from ..dataset.abc import MapStyleABC
from ..runtime import Runtime
from ..types.item import Sample

JsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    Sequence["JsonValue"],
    Mapping[str, "JsonValue"],
]
FilterLabel = Union[bool, str, Enum]
_Index = Sequence[int]


@dataclass(frozen=True)
class FilterDecision:
    label: FilterLabel
    metrics: Mapping[str, JsonValue]


@dataclass(frozen=True)
class FilterApplyReport:
    """Wall-clock observability for one explicit filter apply call."""

    logs_dir: Path | None
    elapsed_seconds: float
    dataset_seconds: float
    cache_lookup_seconds: float
    cache_build_seconds: float
    partition_read_seconds: float
    sample_count: int
    cache_hit: bool
    cache_path: Path

    @property
    def samples_per_second(self) -> float:
        if self.elapsed_seconds <= 0.0:
            return math.inf if self.sample_count > 0 else 0.0
        return self.sample_count / self.elapsed_seconds


FilterOutput = Union[FilterLabel, FilterDecision]
FilterPredicate = Callable[[Sample], FilterOutput]
FilterFactory = Callable[[], FilterPredicate]


@runtime_checkable
class BatchFilterPredicate(Protocol):
    def __call__(self, sample: Sample) -> FilterOutput: ...

    def call_batch(self, samples: Sequence[Sample]) -> Sequence[FilterOutput]: ...


if TYPE_CHECKING:
    from ..dataset.abc import AnyDataset
    from ..store.reader import StoreDataset
    from .api import FilteredDataset

    FilterDataset = Union[AnyDataset, StoreDataset, FilteredDataset]
else:
    # Keep the runtime alias import-safe: ``FilteredDataset`` imports this
    # module while its type-only union is still being resolved by the checker.
    FilterDataset = MapStyleABC

DatasetFactory = Callable[[], FilterDataset]


class FilterApplyKwargs(TypedDict):
    input_id: NotRequired[str | None]
    metrics: NotRequired[bool]
    device: NotRequired[Devices]
    batch_size: NotRequired[int]
    num_workers: NotRequired[int]
    prefetch_factor: NotRequired[int | None]
    commit_samples: NotRequired[int]
    max_shard_samples: NotRequired[int | None]
    write_workers: NotRequired[int]
    write_prefetch: NotRequired[int | None]
    worker_timeout: NotRequired[float | None]
    runtime: NotRequired[Runtime]
    rebuild: NotRequired[bool]


class ResolvedFilterApplyOptions(TypedDict):
    input_id: str | None
    metrics: bool
    device: Devices
    batch_size: int
    num_workers: int
    prefetch_factor: int | None
    commit_samples: int
    max_shard_samples: int | None
    write_workers: int
    write_prefetch: int | None
    worker_timeout: float | None
    runtime: Runtime
    rebuild: bool


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
