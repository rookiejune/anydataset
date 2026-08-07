"""One-to-one filter decisions aligned to dataset-owned snapshot segments."""

from __future__ import annotations

import hashlib
import json
import operator
from bisect import bisect_left
from dataclasses import dataclass, replace
from typing import Any

from typing_extensions import Unpack

from ..dataset._snapshot import SnapshotSegment
from ..dataset._snapshot import snapshot_segments as dataset_snapshot_segments
from ..dataset.abc import MapStyleABC
from ..dataset.universe import DatasetUniverse
from ..dataset.view import Selection, SelectionView, rebase_selection
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
    """Decision coverage over the input prefix visible at this call."""

    expected_snapshots: int
    completed_snapshots: int
    expected_samples: int
    completed_samples: int

    @property
    def complete(self) -> bool:
        return self.completed_snapshots == self.expected_snapshots


@dataclass(frozen=True)
class DecisionView:
    """One filter rule bound to a complete one-to-one sample universe."""

    rule: FilterRule
    dataset_factory: Any
    labels: tuple[str, ...] = ("accept",)
    input_id: str | None = None
    metrics: bool = False

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

    def select(self, *labels: FilterLabel) -> DecisionView:
        """Return another terminal label selection over the same decisions."""

        return replace(self, labels=normalized_labels(labels))

    def status(self) -> DecisionStatus:
        """Inspect published per-snapshot decisions without running the rule."""

        source, universe = _source(self.dataset_factory())
        try:
            segments = _segments(universe, input_id=self.input_id)
            completed_snapshots = 0
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
                completed_snapshots += 1
                completed_samples += segment.sample_count
            return DecisionStatus(
                expected_snapshots=len(segments),
                completed_snapshots=completed_snapshots,
                expected_samples=len(universe),
                completed_samples=completed_samples,
            )
        finally:
            _close(source)

    def produce(
        self,
        **kwargs: Unpack[FilterApplyKwargs],
    ) -> DecisionStatus:
        """Publish decisions for at most the next missing input snapshot."""

        source, universe = _source(self.dataset_factory())
        try:
            segments = _segments(universe, input_id=self.input_id)
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
            completed_snapshots = 0
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
                completed_snapshots += 1
                completed_samples += segment.sample_count
            return DecisionStatus(
                expected_snapshots=len(segments),
                completed_snapshots=completed_snapshots,
                expected_samples=len(universe),
                completed_samples=completed_samples,
            )
        finally:
            _close(source)

    def load(self) -> SelectionView:
        """Open the fixed decision prefix and apply the configured labels."""

        source, universe = _source(self.dataset_factory())
        segments = _segments(universe, input_id=self.input_id)
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
) -> tuple[SnapshotSegment, ...]:
    identity = _dataset_snapshot_id(dataset, input_id=input_id)
    return dataset_snapshot_segments(
        dataset,
        fallback_snapshot_id=identity,
    )


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
