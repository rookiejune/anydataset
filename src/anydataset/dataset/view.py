"""Immutable selections over complete dataset universes.

Selections contain only membership decisions.  Payload transforms continue to
consume the complete ``DatasetUniverse``; ``SelectionView`` applies the
ordered intersection only when samples are returned to a caller.
"""

from __future__ import annotations

import operator
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

from ..types.item import Sample
from ._selection import selected_index_groups
from .abc import MapStyleABC
from .universe import DatasetUniverse


class UnknownDecisionError(RuntimeError):
    """A logical selection position depends on an unresolved decision."""

    def __init__(self, universe_index: int) -> None:
        self.universe_index = universe_index
        super().__init__(
            f"selection decision for universe index {universe_index} is unknown"
        )


@runtime_checkable
class Selection(Protocol):
    """Narrow membership/index contract shared with online selections."""

    @property
    def universe(self) -> DatasetUniverse: ...

    def contains(self, universe_index: int) -> bool | None: ...

    def selected_index(self, position: int) -> int: ...

    def __len__(self) -> int: ...

    def rebase(self, universe: DatasetUniverse) -> Selection: ...


@dataclass(frozen=True)
class DecisionSet:
    """One immutable label (or unresolved ``None``) per universe sample."""

    universe: DatasetUniverse
    decisions: tuple[Hashable | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.universe, DatasetUniverse):
            raise TypeError("universe must be a DatasetUniverse.")
        if not isinstance(self.decisions, tuple):
            raise TypeError("decisions must be a tuple.")
        if len(self.decisions) != len(self.universe):
            raise ValueError("decisions must contain one entry per universe sample.")
        for index, decision in enumerate(self.decisions):
            if decision is not None and not isinstance(decision, Hashable):
                raise TypeError(f"decisions[{index}] must be hashable or None.")

    def decision(self, universe_index: int) -> Hashable | None:
        return self.decisions[_position(universe_index, len(self.universe))]

    @property
    def unknown_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index, decision in enumerate(self.decisions) if decision is None
        )

    def select(self, *labels: Hashable) -> StaticSelection:
        return StaticSelection(self, frozenset(labels))

    def rebase(self, universe: DatasetUniverse) -> DecisionSet:
        if universe is self.universe:
            return self
        target_positions = _sample_positions(universe)
        rebased: list[Hashable | None] = [None] * len(universe)
        seen: set[int] = set()
        for source_index, decision in enumerate(self.decisions):
            sample_id = self.universe.sample_id(source_index)
            try:
                target_index = target_positions[sample_id]
            except KeyError as exc:
                raise ValueError(
                    f"target universe is missing sample_id {sample_id!r}."
                ) from exc
            rebased[target_index] = decision
            seen.add(target_index)
        if len(seen) != len(universe):
            raise ValueError("universes do not have one-to-one sample lineage.")
        return DecisionSet(universe, tuple(rebased))


@dataclass(frozen=True)
class StaticSelection:
    """Select a fixed set of labels from a static decision set."""

    decision_set: DecisionSet
    labels: frozenset[Hashable]

    def __post_init__(self) -> None:
        if not isinstance(self.decision_set, DecisionSet):
            raise TypeError("decision_set must be a DecisionSet.")
        if not isinstance(self.labels, frozenset):
            raise TypeError("labels must be a frozenset.")

    @property
    def universe(self) -> DatasetUniverse:
        return self.decision_set.universe

    def contains(self, universe_index: int) -> bool | None:
        decision = self.decision_set.decision(universe_index)
        if decision is None:
            return None
        return decision in self.labels

    def selected_index(self, position: int) -> int:
        return _selected_index(self, position)

    @property
    def indices(self) -> tuple[int, ...]:
        return _selected_indices(self)

    def __len__(self) -> int:
        return _selected_length(self)

    def rebase(self, universe: DatasetUniverse) -> StaticSelection:
        return StaticSelection(self.decision_set.rebase(universe), self.labels)


@dataclass(frozen=True)
class _LineageSelection:
    """Lazy one-to-one selection mapping that preserves unresolved decisions."""

    source: Selection
    universe: DatasetUniverse
    source_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, Selection):
            raise TypeError("source must implement the Selection protocol.")
        if not isinstance(self.universe, DatasetUniverse):
            raise TypeError("universe must be a DatasetUniverse.")
        if len(self.source_indices) != len(self.universe):
            raise ValueError("source_indices must cover the target universe.")

    def contains(self, universe_index: int) -> bool | None:
        position = _position(universe_index, len(self.universe))
        return self.source.contains(self.source_indices[position])

    def selected_index(self, position: int) -> int:
        return _selected_index(self, position)

    def __len__(self) -> int:
        return _selected_length(self)

    def rebase(self, universe: DatasetUniverse) -> Selection:
        return rebase_selection(self, universe)

    def wait_decision(self, universe_index: int) -> bool:
        position = _position(universe_index, len(self.universe))
        source_index = self.source_indices[position]
        wait = getattr(self.source, "wait_decision", None)
        if not callable(wait):
            state = self.source.contains(source_index)
            if state is None:
                raise UnknownDecisionError(position)
            return state
        return bool(wait(source_index))

    def wait_complete(self) -> None:
        wait = getattr(self.source, "wait_complete", None)
        if callable(wait):
            wait()

    def close(self) -> None:
        close = getattr(self.source, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class _BorrowedSelection:
    """Delegate selection behavior without claiming ownership of its source."""

    selection: Selection

    @property
    def universe(self) -> DatasetUniverse:
        return self.selection.universe

    def contains(self, universe_index: int) -> bool | None:
        return self.selection.contains(universe_index)

    def selected_index(self, position: int) -> int:
        return self.selection.selected_index(position)

    def __len__(self) -> int:
        return len(self.selection)

    def rebase(self, universe: DatasetUniverse) -> Selection:
        return _BorrowedSelection(self.selection.rebase(universe))

    def wait_decision(self, universe_index: int) -> bool:
        wait = getattr(self.selection, "wait_decision", None)
        if not callable(wait):
            state = self.selection.contains(universe_index)
            if state is None:
                raise UnknownDecisionError(universe_index)
            return state
        return bool(wait(universe_index))

    def wait_complete(self) -> None:
        wait = getattr(self.selection, "wait_complete", None)
        if callable(wait):
            wait()


class _SelectionViewState:
    __slots__ = ("closed",)

    def __init__(self) -> None:
        self.closed = False

    def claim_close(self) -> bool:
        if self.closed:
            return False
        self.closed = True
        return True


@dataclass(frozen=True)
class SelectionView(MapStyleABC):
    """A map-style view applying an ordered intersection at its boundary."""

    universe: DatasetUniverse
    selections: tuple[Selection, ...] = ()
    resources: tuple[Any, ...] = field(default=(), repr=False, compare=False)
    _resource_state: _SelectionViewState = field(
        default_factory=_SelectionViewState,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.universe, DatasetUniverse):
            raise TypeError("universe must be a DatasetUniverse.")
        if not isinstance(self.selections, tuple):
            raise TypeError("selections must be a tuple.")
        if not isinstance(self.resources, tuple):
            raise TypeError("resources must be a tuple.")
        for selection in self.selections:
            if not isinstance(selection, Selection):
                raise TypeError("selections must implement the Selection protocol.")
            if selection.universe is not self.universe:
                raise ValueError("every selection must belong to the view universe.")

    def contains(self, universe_index: int) -> bool | None:
        position = _position(universe_index, len(self.universe))
        unknown = False
        for selection in self.selections:
            selected = selection.contains(position)
            if selected is False:
                return False
            if selected is None:
                unknown = True
        return None if unknown else True

    def wait_decision(self, universe_index: int) -> bool:
        """Wait only for unresolved decisions needed by one universe row."""

        position = _position(universe_index, len(self.universe))
        for selection in self.selections:
            selected = selection.contains(position)
            if selected is False:
                return False
            if selected is True:
                continue
            wait = getattr(selection, "wait_decision", None)
            if not callable(wait):
                raise UnknownDecisionError(position)
            if not bool(wait(position)):
                return False
        return True

    def wait_complete(self) -> None:
        """Wait for every live selection while leaving static unknowns unchanged."""

        for selection in self.selections:
            wait = getattr(selection, "wait_complete", None)
            if callable(wait):
                wait()

    def selected_index(self, position: int) -> int:
        return _selected_index(self, position)

    def universe_index(self, logical_index: int) -> int:
        """Map a returned logical position to the complete universe."""

        return self.selected_index(logical_index)

    @property
    def indices(self) -> tuple[int, ...]:
        return _selected_indices(self)

    def __len__(self) -> int:
        return _selected_length(self)

    def __getitem__(self, index: int) -> Sample:
        return self.universe[self.universe_index(index)]

    def __getitems__(self, indexes: Sequence[int]) -> list[Sample]:
        universe_indexes = tuple(self.universe_index(index) for index in indexes)
        return self.universe.__getitems__(universe_indexes)

    def sample_id(self, index: int) -> str:
        return self.universe.sample_id(self.universe_index(index))

    def universe_id(self) -> str | None:
        return self.universe.universe_id()

    def global_index(self, index: int) -> int:
        return self.universe.global_index(self.universe_index(index))

    def cost_row(self, index: int) -> Any:
        return self.universe.cost_row(self.universe_index(index))

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        yield from selected_index_groups(
            self.universe,
            self.indices,
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            num_replicas=num_replicas,
            rank=rank,
        )

    def select(self, selection: Selection) -> SelectionView:
        if selection.universe is not self.universe:
            raise ValueError("selection must belong to the view universe.")
        return SelectionView(
            self.universe,
            self.selections + (selection,),
            self.resources,
        )

    def rebase(self, universe: DatasetUniverse) -> SelectionView:
        return SelectionView(
            universe,
            tuple(
                _BorrowedSelection(selection.rebase(universe))
                for selection in self.selections
            ),
            (self,),
        )

    def close(self) -> None:
        if not self._resource_state.claim_close():
            return
        error: BaseException | None = None
        try:
            self.universe.close()
        except BaseException as exc:
            error = exc
        for selection in reversed(self.selections):
            close = getattr(selection, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as exc:
                if error is None:
                    error = exc
        for resource in reversed(self.resources):
            close = getattr(resource, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def __enter__(self) -> SelectionView:
        self.universe.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _selected_index(selection: Selection, position: int) -> int:
    logical_position = operator.index(position)
    if logical_position < 0:
        logical_position += _selected_length(selection)
    if logical_position < 0:
        raise IndexError("selection index out of range")
    selected_position = 0
    for universe_index in range(len(selection.universe)):
        state = _blocking_contains(selection, universe_index)
        if state:
            if selected_position == logical_position:
                return universe_index
            selected_position += 1
    raise IndexError("selection index out of range")


def _selected_indices(selection: Selection) -> tuple[int, ...]:
    _wait_complete(selection)
    output: list[int] = []
    for universe_index in range(len(selection.universe)):
        state = selection.contains(universe_index)
        if state is None:
            raise UnknownDecisionError(universe_index)
        if state:
            output.append(universe_index)
    return tuple(output)


def _selected_length(selection: Selection) -> int:
    _wait_complete(selection)
    count = 0
    for universe_index in range(len(selection.universe)):
        state = selection.contains(universe_index)
        if state is None:
            raise UnknownDecisionError(universe_index)
        count += state
    return count


def _blocking_contains(selection: Selection, universe_index: int) -> bool:
    state = selection.contains(universe_index)
    if state is not None:
        return state
    wait = getattr(selection, "wait_decision", None)
    if not callable(wait):
        raise UnknownDecisionError(universe_index)
    return bool(wait(universe_index))


def _wait_complete(selection: Selection) -> None:
    wait = getattr(selection, "wait_complete", None)
    if callable(wait):
        wait()


def _sample_positions(universe: DatasetUniverse) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index in range(len(universe)):
        sample_id = universe.sample_id(index)
        if sample_id in positions:
            raise ValueError(f"duplicate sample_id {sample_id!r} in target universe.")
        positions[sample_id] = index
    return positions


def rebase_selection(
    selection: Selection,
    universe: DatasetUniverse,
) -> Selection:
    if universe is selection.universe:
        return selection
    source_positions = _sample_positions(selection.universe)
    source_indices: list[int] = []
    seen: set[int] = set()
    for target_index in range(len(universe)):
        sample_id = universe.sample_id(target_index)
        try:
            source_index = source_positions[sample_id]
        except KeyError as exc:
            raise ValueError(
                f"source universe is missing sample_id {sample_id!r}."
            ) from exc
        if source_index in seen:
            raise ValueError("universes do not have one-to-one sample lineage.")
        seen.add(source_index)
        source_indices.append(source_index)
    if len(seen) != len(selection.universe):
        raise ValueError("universes do not have one-to-one sample lineage.")
    return _LineageSelection(selection, universe, tuple(source_indices))


def _position(index: int, length: int) -> int:
    position = operator.index(index)
    if position < 0:
        position += length
    if position < 0 or position >= length:
        raise IndexError("universe index out of range")
    return position


__all__ = [
    "DecisionSet",
    "rebase_selection",
    "Selection",
    "SelectionView",
    "StaticSelection",
    "UnknownDecisionError",
]
