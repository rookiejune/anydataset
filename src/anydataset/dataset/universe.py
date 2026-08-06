"""Stable complete sample spaces for derived dataset views.

``DatasetUniverse`` wraps a complete map-style dataset and exposes the two
narrow identity operations needed by filters and one-to-one transforms.  It
does not apply selection and does not infer sample identity from a dense row
position.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Iterator, Sequence
from types import TracebackType
from typing import Any, Protocol, cast, runtime_checkable

from ..types.item import Sample
from .abc import MapStyleABC


@runtime_checkable
class IndexIdentity(Protocol):
    """Expose a stable external index for a universe position."""

    def global_index(self, index: int) -> int: ...


@runtime_checkable
class SampleIdentity(Protocol):
    """Expose the stable sample lineage identifier at a universe position."""

    def sample_id(self, index: int) -> str: ...


@runtime_checkable
class UniverseIdentity(Protocol):
    """Expose a stable identity for one complete logical sample universe."""

    def universe_id(self) -> str | None: ...


class DatasetUniverse(MapStyleABC):
    """A complete, stable map-style sample space.

    The wrapped dataset remains responsible for payload access, costs, and
    shuffle locality.  A sample identity provider is mandatory because dense
    positions are local to one universe and cannot prove transform lineage.
    """

    __slots__ = ("_closed", "_dataset", "_index_identity", "_sample_identity")

    def __init__(
        self,
        dataset: MapStyleABC,
        *,
        sample_identity: SampleIdentity | None = None,
        index_identity: IndexIdentity | None = None,
    ) -> None:
        if not isinstance(dataset, MapStyleABC):
            raise TypeError("dataset must be a MapStyleABC.")
        resolved_sample_identity = sample_identity
        if resolved_sample_identity is None and isinstance(dataset, SampleIdentity):
            resolved_sample_identity = dataset
        if resolved_sample_identity is None:
            raise TypeError(
                "sample_identity is required when dataset does not provide sample_id()."
            )
        resolved_index_identity = index_identity
        if resolved_index_identity is None and isinstance(dataset, IndexIdentity):
            resolved_index_identity = dataset
        self._dataset = dataset
        self._sample_identity = resolved_sample_identity
        self._index_identity = resolved_index_identity
        self._closed = False

    @property
    def dataset(self) -> MapStyleABC:
        return self._dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        return self.dataset[_position(index, len(self))]

    def __getitems__(self, indexes: Sequence[int]) -> list[Sample]:
        positions = tuple(_position(index, len(self)) for index in indexes)
        getitems = getattr(self.dataset, "__getitems__", None)
        if callable(getitems):
            return list(cast(Iterable[Sample], getitems(positions)))
        return [self.dataset[position] for position in positions]

    def sample_id(self, index: int) -> str:
        position = _position(index, len(self))
        value = self._sample_identity.sample_id(position)
        if not isinstance(value, str) or not value:
            raise ValueError("sample_id() must return a non-empty string.")
        return value

    def global_index(self, index: int) -> int:
        position = _position(index, len(self))
        if self._index_identity is None:
            return position
        value = self._index_identity.global_index(position)
        if type(value) is not int:
            raise TypeError("global_index() must return an integer.")
        return value

    def universe_id(self) -> str | None:
        identity = getattr(self.dataset, "universe_id", None)
        if not callable(identity):
            return None
        value = identity()
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("universe_id() must return a non-empty string or None.")
        return value

    def cost_row(self, index: int) -> Any:
        return self.dataset.cost_row(_position(index, len(self)))

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        yield from self.dataset._shuffle(
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            num_replicas=num_replicas,
            rank=rank,
        )

    def close(self) -> None:
        if not self._claim_close():
            return
        close = getattr(self.dataset, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> DatasetUniverse:
        enter = getattr(self.dataset, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._claim_close():
            return
        exit_method = getattr(self.dataset, "__exit__", None)
        if callable(exit_method):
            exit_method(exc_type, exc, traceback)
            return
        close = getattr(self.dataset, "close", None)
        if callable(close):
            close()

    def _claim_close(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        return True


def _position(index: int, length: int) -> int:
    position = operator.index(index)
    if position < 0:
        position += length
    if position < 0 or position >= length:
        raise IndexError("dataset index out of range")
    return position


__all__ = [
    "DatasetUniverse",
    "IndexIdentity",
    "SampleIdentity",
    "UniverseIdentity",
]
