"""One-to-one filter decisions aligned to dataset-owned snapshot segments."""

from __future__ import annotations

import hashlib
import json
import operator
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from typing_extensions import Unpack

from ..dataset._snapshot import SnapshotSegment
from ..dataset._snapshot import snapshot_segments as dataset_snapshot_segments
from ..dataset.abc import MapStyleABC
from ..dataset.universe import DatasetUniverse, SampleIdentity
from ..dataset.view import Selection, SelectionView, rebase_selection
from ..types.item import Sample
from .api import FilterRule, normalized_labels, selected_index
from .cache.generations import GenerationLease
from .cache.identity import filter_identity, filter_path, metadata
from .cache.ready import ready_filter_generation
from .cache.storage import read_partitions
from .runtime.apply import apply_filter
from .runtime.options import options as apply_options
from .types import FilterApplyKwargs, FilterLabel


@dataclass(frozen=True)
class DecisionStatus:
    """Published decision coverage over the input prefix visible at this call."""

    expected_samples: int
    completed_samples: int

    @property
    def complete(self) -> bool:
        return self.completed_samples == self.expected_samples

    @property
    def pending_samples(self) -> int:
        return self.expected_samples - self.completed_samples


@dataclass(frozen=True)
class DecisionView:
    """One filter rule bound to a complete one-to-one sample universe.

    Input snapshots remain the semantic partition boundary. Large snapshots are
    divided into bounded immutable decision windows so readers can pin the
    longest committed prefix without observing the writer's private tail.
    """

    rule: FilterRule
    dataset_factory: Any
    labels: tuple[str, ...] = ("accept",)
    input_id: str | None = None
    metrics: bool = False
    max_new_samples: int = 100_000

    def __post_init__(self) -> None:
        if not isinstance(self.rule, FilterRule):
            raise TypeError("rule must be a FilterRule.")
        if not callable(self.dataset_factory):
            raise TypeError("dataset_factory must be callable.")
        if not isinstance(self.labels, tuple) or not self.labels:
            raise ValueError("decision labels must be a non-empty tuple.")
        if any(not isinstance(label, str) or not label for label in self.labels):
            raise ValueError("decision labels must contain non-empty strings.")
        if self.input_id is not None and (
            not isinstance(self.input_id, str) or not self.input_id
        ):
            raise ValueError("input_id must be non-empty or None.")
        if type(self.metrics) is not bool:
            raise TypeError("metrics must be a boolean.")
        if isinstance(self.max_new_samples, bool) or not isinstance(
            self.max_new_samples,
            int,
        ):
            raise TypeError("max_new_samples must be an integer.")
        if self.max_new_samples <= 0:
            raise ValueError("max_new_samples must be positive.")

    def select(self, *labels: FilterLabel) -> DecisionView:
        """Return another terminal label selection over the same decisions."""

        return replace(self, labels=normalized_labels(labels))

    def status(self) -> DecisionStatus:
        """Inspect published per-snapshot decisions without running the rule."""

        source, universe = _source(self.dataset_factory())
        try:
            segments = _segments(
                universe,
                input_id=self.input_id,
                max_new_samples=self.max_new_samples,
            )
            completed_samples = 0
            for segment in segments:
                ready = _ready(
                    segment,
                    self.rule,
                    input_id=self.input_id,
                    metrics=self.metrics,
                )
                if ready is None:
                    break
                ready.close()
                completed_samples += segment.sample_count
            return DecisionStatus(
                expected_samples=len(universe),
                completed_samples=completed_samples,
            )
        finally:
            _close(source)

    def produce(
        self,
        **kwargs: Unpack[FilterApplyKwargs],
    ) -> DecisionStatus:
        """Publish decisions for at most the next missing committed window."""

        source, universe = _source(self.dataset_factory())
        try:
            segments = _segments(
                universe,
                input_id=self.input_id,
                max_new_samples=self.max_new_samples,
            )
            target: SnapshotSegment | None = None
            for segment in segments:
                ready = _ready(
                    segment,
                    self.rule,
                    input_id=self.input_id,
                    metrics=self.metrics,
                )
                if ready is None:
                    target = segment
                    break
                ready.close()
            if target is not None:
                options = apply_options(kwargs)
                applied = apply_filter(
                    self.rule,
                    input_id=_segment_input_id(self.input_id, target),
                    metrics=self.metrics,
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
                    rebuild=options["rebuild"],
                    dataset_factory=_OpenedDatasetFactory(target.dataset),
                    allow_logical=True,
                )
                applied.cache.close()
            completed_samples = 0
            for segment in segments:
                ready = _ready(
                    segment,
                    self.rule,
                    input_id=self.input_id,
                    metrics=self.metrics,
                )
                if ready is None:
                    break
                ready.close()
                completed_samples += segment.sample_count
            return DecisionStatus(
                expected_samples=len(universe),
                completed_samples=completed_samples,
            )
        finally:
            _close(source)

    def load(self) -> SelectionView:
        """Open the fixed decision prefix and apply the configured labels."""

        source, universe = _source(self.dataset_factory())
        segments = _segments(
            universe,
            input_id=self.input_id,
            max_new_samples=self.max_new_samples,
        )
        selected: list[int] = []
        leases: list[GenerationLease] = []
        covered_samples = 0
        try:
            for segment in segments:
                ready = _ready(
                    segment,
                    self.rule,
                    input_id=self.input_id,
                    metrics=self.metrics,
                )
                if ready is None:
                    break
                partitions = read_partitions(ready.path)
                local_indexes = selected_index(partitions, self.labels)
                selected.extend(segment.start + index for index in local_indexes)
                covered_samples += segment.sample_count
                leases.append(ready)
            if segments and covered_samples == 0:
                raise FileNotFoundError(
                    f"No decision snapshot is published for filter {self.rule.name!r}."
                )
            decision_universe = (
                source.universe
                if isinstance(source, SelectionView)
                else DatasetUniverse(universe)
            )
            selection = _DecisionSelection(
                universe=decision_universe,
                indices=tuple(selected),
                covered_samples=covered_samples,
                leases=tuple(leases),
            )
            leases = []
            if isinstance(source, SelectionView):
                return source.select(selection)
            return SelectionView(selection.universe, (selection,))
        except BaseException:
            for lease in leases:
                lease.close()
            _close(source)
            raise


@dataclass(frozen=True)
class _OpenedDatasetFactory:
    dataset: MapStyleABC

    def __call__(self) -> MapStyleABC:
        return self.dataset


@dataclass(frozen=True)
class _SnapshotWindow(MapStyleABC):
    """Read one immutable local range without owning the underlying dataset."""

    dataset: MapStyleABC
    start: int
    sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, MapStyleABC):
            raise TypeError("snapshot window dataset must be a MapStyleABC.")
        if not isinstance(self.dataset, SampleIdentity):
            raise TypeError("snapshot window dataset must provide sample_id().")
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("snapshot window start must be an integer.")
        if self.start < 0:
            raise ValueError("snapshot window start must be non-negative.")
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count,
            int,
        ):
            raise TypeError("snapshot window sample_count must be an integer.")
        if self.sample_count <= 0:
            raise ValueError("snapshot window sample_count must be positive.")
        if self.start + self.sample_count > len(self.dataset):
            raise ValueError("snapshot window exceeds the dataset length.")

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> Sample:
        return self.dataset[self.start + _position(index, len(self))]

    def __getitems__(self, indexes: Sequence[int]) -> list[Sample]:
        local = tuple(self.start + _position(index, len(self)) for index in indexes)
        getitems = getattr(self.dataset, "__getitems__", None)
        if callable(getitems):
            values = getitems(local)
            if not isinstance(values, Sequence):
                raise TypeError("snapshot window batch read must return a sequence.")
            if len(values) != len(local):
                raise ValueError("snapshot window batch read returned the wrong count.")
            return list(cast(Sequence[Sample], values))
        return [self.dataset[index] for index in local]

    def sample_id(self, index: int) -> str:
        identity = cast(SampleIdentity, self.dataset)
        value = identity.sample_id(self.start + _position(index, len(self)))
        if not isinstance(value, str) or not value:
            raise ValueError("snapshot window sample_id() must be non-empty.")
        return value

    def global_index(self, index: int) -> int:
        return _position(index, len(self))

    def cost_row(self, index: int) -> Any:
        return self.dataset.cost_row(self.start + _position(index, len(self)))


@dataclass(frozen=True)
class _DecisionSelection:
    universe: DatasetUniverse
    indices: tuple[int, ...]
    covered_samples: int
    leases: tuple[GenerationLease, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.universe, DatasetUniverse):
            raise TypeError("decision universe must be a DatasetUniverse.")
        if not isinstance(self.indices, tuple):
            raise TypeError("decision indices must be a tuple.")
        if self.covered_samples < 0 or self.covered_samples > len(self.universe):
            raise ValueError("decision covered_samples is outside the universe.")
        previous = -1
        for index in self.indices:
            if type(index) is not int or index <= previous:
                raise ValueError("decision indices must be strictly increasing.")
            if index >= self.covered_samples:
                raise ValueError("decision index is outside published coverage.")
            previous = index

    def contains(self, universe_index: int) -> bool:
        position = _position(universe_index, len(self.universe))
        if position >= self.covered_samples:
            return False
        offset = bisect_left(self.indices, position)
        return offset < len(self.indices) and self.indices[offset] == position

    @property
    def sealed(self) -> bool:
        return self.covered_samples == len(self.universe)

    def selected_index(self, position: int) -> int:
        return self.indices[_position(position, len(self.indices))]

    def __len__(self) -> int:
        return len(self.indices)

    def rebase(self, universe: DatasetUniverse) -> Selection:
        return rebase_selection(self, universe)

    def close(self) -> None:
        for lease in self.leases:
            lease.close()


def _source(dataset: object) -> tuple[MapStyleABC, MapStyleABC]:
    if isinstance(dataset, SelectionView):
        return dataset, dataset.universe.dataset
    if not isinstance(dataset, MapStyleABC):
        raise TypeError("decision inputs require a map-style dataset.")
    return dataset, dataset


def _segments(
    dataset: MapStyleABC,
    *,
    input_id: str | None,
    max_new_samples: int,
) -> tuple[SnapshotSegment, ...]:
    identity = _dataset_snapshot_id(dataset, input_id=input_id)
    source = dataset_snapshot_segments(
        dataset,
        fallback_snapshot_id=identity,
    )
    output: list[SnapshotSegment] = []
    for segment in source:
        if segment.sample_count <= max_new_samples:
            output.append(segment)
            continue
        local_start = 0
        while local_start < segment.sample_count:
            count = min(max_new_samples, segment.sample_count - local_start)
            output.append(
                SnapshotSegment(
                    snapshot_id=_window_snapshot_id(segment, local_start, count),
                    start=segment.start + local_start,
                    sample_count=count,
                    dataset=_SnapshotWindow(segment.dataset, local_start, count),
                )
            )
            local_start += count
    return tuple(output)


def _window_snapshot_id(
    segment: SnapshotSegment,
    local_start: int,
    sample_count: int,
) -> str:
    payload = json.dumps(
        {
            "schema": "anydataset-decision-window-v1",
            "snapshot_id": segment.snapshot_id,
            "segment_samples": segment.sample_count,
            "start": local_start,
            "sample_count": sample_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"decision-window-v1:{hashlib.sha256(payload).hexdigest()}"


def _dataset_snapshot_id(dataset: MapStyleABC, *, input_id: str | None) -> str:
    identity = getattr(dataset, "universe_id", None)
    value = identity() if callable(identity) else None
    if value is None:
        value = input_id
    if value is None:
        payload = {
            "type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
            "sample_count": len(dataset),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        value = f"logical:{hashlib.sha256(encoded).hexdigest()}"
    if not isinstance(value, str) or not value:
        raise ValueError("decision input universe_id must be non-empty or None.")
    return value


def _segment_input_id(base: str | None, segment: SnapshotSegment) -> str:
    payload = json.dumps(
        {
            "schema": "anydataset-decision-segment-v1",
            "base": base,
            "snapshot_id": segment.snapshot_id,
            "sample_count": segment.sample_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"decision-segment-v1:{hashlib.sha256(payload).hexdigest()}"


def _ready(
    segment: SnapshotSegment,
    rule: FilterRule,
    *,
    input_id: str | None,
    metrics: bool,
) -> GenerationLease | None:
    resolved_input_id = _segment_input_id(input_id, segment)
    identity = filter_identity(segment.dataset, input_id=resolved_input_id)
    expected = metadata(identity, segment.sample_count, rule)
    generation, _reason = ready_filter_generation(
        filter_path(rule, identity),
        expected,
        metrics=metrics,
    )
    return None if generation is None else generation.lease


def _position(index: int, length: int) -> int:
    position = operator.index(index)
    if position < 0:
        position += length
    if position < 0 or position >= length:
        raise IndexError("decision selection index out of range")
    return position


def _close(dataset: object) -> None:
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


__all__ = ["DecisionStatus", "DecisionView"]
