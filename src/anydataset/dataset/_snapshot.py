"""Internal routing for datasets assembled from incremental snapshots."""

from __future__ import annotations

import operator
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, cast, runtime_checkable

from ..types.item import Sample
from .abc import MapStyleABC
from .universe import SampleIdentity


@dataclass(frozen=True)
class SnapshotSegment:
    """One immutable dense increment in a loaded snapshot catalog."""

    snapshot_id: str
    start: int
    sample_count: int
    dataset: MapStyleABC

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id:
            raise ValueError("snapshot_id must be a non-empty string.")
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("snapshot start must be an integer.")
        if self.start < 0:
            raise ValueError("snapshot start must be non-negative.")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int):
            raise TypeError("snapshot sample_count must be an integer.")
        if self.sample_count <= 0:
            raise ValueError("snapshot sample_count must be positive.")
        if not isinstance(self.dataset, MapStyleABC):
            raise TypeError("snapshot dataset must be a MapStyleABC.")
        if not isinstance(self.dataset, SampleIdentity):
            raise TypeError("snapshot dataset must provide sample_id().")
        if len(self.dataset) != self.sample_count:
            raise ValueError("snapshot dataset length does not match sample_count.")

    @property
    def stop(self) -> int:
        return self.start + self.sample_count


@runtime_checkable
class SnapshotPartitioned(Protocol):
    """Expose immutable dense segments for one loaded logical dataset."""

    @property
    def snapshot_segments(self) -> Sequence[SnapshotSegment]: ...


class SnapshotCatalogDataset(MapStyleABC):
    """A map-style concatenation of all increments in one loaded catalog.

    The constructor captures one catalog state. It never polls for later
    entries; callers reconstruct the owning resource to load the catalog again
    from its first snapshot.
    """

    def __init__(
        self,
        segments: Sequence[SnapshotSegment],
        *,
        sealed: bool,
        universe_id: str | None = None,
    ) -> None:
        values, start = _validated_segments(segments)
        if type(sealed) is not bool:
            raise TypeError("snapshot catalog sealed must be a boolean.")
        if universe_id is not None and (
            not isinstance(universe_id, str) or not universe_id
        ):
            raise ValueError(
                "snapshot catalog universe_id must be non-empty or None."
            )
        self._segments = values
        self._stops = tuple(segment.stop for segment in values)
        self._sample_count = start
        self._sealed = sealed
        self._universe_id = universe_id
        self._closed = False

    @property
    def snapshot_segments(self) -> tuple[SnapshotSegment, ...]:
        self._ensure_open()
        return self._segments

    @property
    def snapshot_ids(self) -> tuple[str, ...]:
        return tuple(segment.snapshot_id for segment in self.snapshot_segments)

    @property
    def snapshot_count(self) -> int:
        return len(self.snapshot_segments)

    @property
    def sealed(self) -> bool:
        self._ensure_open()
        return self._sealed

    def __len__(self) -> int:
        self._ensure_open()
        return self._sample_count

    def __getitem__(self, index: int) -> Sample:
        segment, local_index = self._locate(index)
        return segment.dataset[local_index]

    def __getitems__(self, indexes: Sequence[int]) -> list[Sample]:
        self._ensure_open()
        normalized = tuple(_position(index, len(self)) for index in indexes)
        if not normalized:
            return []
        requests: dict[int, list[tuple[int, int]]] = {}
        for output_index, index in enumerate(normalized):
            segment_index = bisect_right(self._stops, index)
            segment = self._segments[segment_index]
            requests.setdefault(segment_index, []).append(
                (output_index, index - segment.start)
            )
        output: list[Sample | None] = [None] * len(normalized)
        for segment_index, selected in requests.items():
            dataset = self._segments[segment_index].dataset
            local_indexes = tuple(local for _output, local in selected)
            getitems = getattr(dataset, "__getitems__", None)
            if callable(getitems):
                resolved = getitems(local_indexes)
                if not isinstance(resolved, Sequence):
                    raise TypeError("snapshot batch read must return a sequence.")
                values = cast(Sequence[Sample], resolved)
            else:
                values = [dataset[index] for index in local_indexes]
            if len(values) != len(local_indexes):
                raise ValueError("snapshot batch read returned the wrong sample count.")
            for (output_index, _local), value in zip(selected, values):
                output[output_index] = value
        return [cast(Sample, value) for value in output]

    def sample_id(self, index: int) -> str:
        segment, local_index = self._locate(index)
        identity = cast(SampleIdentity, segment.dataset)
        value = identity.sample_id(local_index)
        if not isinstance(value, str) or not value:
            raise ValueError("snapshot sample_id() must return a non-empty string.")
        return value

    def global_index(self, index: int) -> int:
        return _position(index, len(self))

    def universe_id(self) -> str | None:
        self._ensure_open()
        return self._universe_id

    def cost_row(self, index: int) -> Any:
        segment, local_index = self._locate(index)
        return segment.dataset.cost_row(local_index)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        for segment in reversed(self._segments):
            close = getattr(segment.dataset, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def __enter__(self) -> SnapshotCatalogDataset:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _locate(self, index: int) -> tuple[SnapshotSegment, int]:
        self._ensure_open()
        normalized = _position(index, len(self))
        segment_index = bisect_right(self._stops, normalized)
        segment = self._segments[segment_index]
        return segment, normalized - segment.start

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SnapshotCatalogDataset is closed.")


def snapshot_segments(
    dataset: MapStyleABC,
    *,
    fallback_snapshot_id: str,
) -> tuple[SnapshotSegment, ...]:
    """Return dataset-owned segments or adapt one map dataset as one snapshot."""

    if not isinstance(fallback_snapshot_id, str) or not fallback_snapshot_id:
        raise ValueError("fallback_snapshot_id must be a non-empty string.")
    if isinstance(dataset, SnapshotPartitioned):
        segments, sample_count = _validated_segments(dataset.snapshot_segments)
        if sample_count != len(dataset):
            raise ValueError(
                "snapshot segments do not cover the complete logical dataset."
            )
        return segments
    if len(dataset) == 0:
        return ()
    return (SnapshotSegment(fallback_snapshot_id, 0, len(dataset), dataset),)


def _validated_segments(
    segments: Sequence[SnapshotSegment],
) -> tuple[tuple[SnapshotSegment, ...], int]:
    values = tuple(segments)
    start = 0
    snapshot_ids: set[str] = set()
    for segment in values:
        if not isinstance(segment, SnapshotSegment):
            raise TypeError("segments must contain SnapshotSegment values.")
        if segment.start != start:
            raise ValueError("snapshot segments must form a dense prefix.")
        if segment.snapshot_id in snapshot_ids:
            raise ValueError("snapshot ids must be unique within one prefix.")
        snapshot_ids.add(segment.snapshot_id)
        start = segment.stop
    return values, start


def _position(index: int, length: int) -> int:
    position = operator.index(index)
    if position < 0:
        position += length
    if position < 0 or position >= length:
        raise IndexError("snapshot index out of range")
    return position


__all__: list[str] = []
