from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import TracebackType

import pytest

from anydataset.dataset.abc import MapStyleABC
from anydataset.dataset.universe import DatasetUniverse
from anydataset.dataset.view import (
    DecisionSet,
    rebase_selection,
    Selection,
    SelectionView,
    UnknownDecisionError,
)
from anydataset.types import AudioItem, AudioView, Modality, Role, Sample


class _Dataset(MapStyleABC):
    def __init__(
        self,
        values: Sequence[int],
        sample_ids: Sequence[str],
        *,
        global_offset: int = 0,
    ) -> None:
        self.values = tuple(values)
        self.sample_ids = tuple(sample_ids)
        self.global_offset = global_offset

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Sample:
        return {
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: self.values[index]}
            )
        }

    def sample_id(self, index: int) -> str:
        return self.sample_ids[index]

    def global_index(self, index: int) -> int:
        return self.global_offset + index

    def cost_row(self, index: int) -> tuple[str, int]:
        return "cost", self.values[index]

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        del seed, epoch
        if num_replicas != 1 or rank != 0:
            raise AssertionError("selection must request the complete ordering")
        groups = ((2, 3), (0, 1)) if shuffle else ((0, 1), (2, 3))
        yield from groups


class _ClosableDataset(_Dataset):
    def __init__(self, values: Sequence[int], sample_ids: Sequence[str]) -> None:
        super().__init__(values, sample_ids)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls > 1:
            raise AssertionError("dataset must be closed exactly once")


class _ContextDataset(_ClosableDataset):
    def __init__(self, values: Sequence[int], sample_ids: Sequence[str]) -> None:
        super().__init__(values, sample_ids)
        self.exit_calls: list[
            tuple[type[BaseException] | None, BaseException | None, TracebackType | None]
        ] = []

    def __enter__(self) -> _ContextDataset:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exit_calls.append((exc_type, exc, traceback))
        self.close()


class _DeferredSelection:
    def __init__(
        self,
        universe: DatasetUniverse,
        decisions: Sequence[bool],
    ) -> None:
        self._universe = universe
        self._decisions = tuple(decisions)
        self._ready: set[int] = set()
        self.decision_waits: list[int] = []
        self.complete_waits = 0

    @property
    def universe(self) -> DatasetUniverse:
        return self._universe

    def contains(self, universe_index: int) -> bool | None:
        if universe_index not in self._ready:
            return None
        return self._decisions[universe_index]

    def selected_index(self, position: int) -> int:
        return SelectionView(self.universe, (self,)).selected_index(position)

    def __len__(self) -> int:
        return len(SelectionView(self.universe, (self,)))

    def rebase(self, universe: DatasetUniverse) -> Selection:
        return rebase_selection(self, universe)

    def wait_decision(self, universe_index: int) -> bool:
        self.decision_waits.append(universe_index)
        self._ready.add(universe_index)
        return self._decisions[universe_index]

    def wait_complete(self) -> None:
        self.complete_waits += 1
        self._ready.update(range(len(self.universe)))


class _ClosableSelection(_DeferredSelection):
    def __init__(
        self,
        universe: DatasetUniverse,
        decisions: Sequence[bool],
    ) -> None:
        super().__init__(universe, decisions)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls > 1:
            raise AssertionError("selection must be closed exactly once")


def _value(sample: Sample) -> int:
    return sample[Role.DEFAULT, Modality.AUDIO].views[AudioView.WAVEFORM]


def _shuffled(view: SelectionView, rank: int) -> tuple[int, ...]:
    return tuple(
        index
        for group in view._shuffle(
            shuffle=True,
            seed=0,
            epoch=0,
            num_replicas=2,
            rank=rank,
        )
        for index in group
    )


def test_universe_delegates_payload_identity_cost_and_shuffle() -> None:
    universe = DatasetUniverse(
        _Dataset((10, 20, 30, 40), ("a", "b", "c", "d"), global_offset=100)
    )

    assert _value(universe[1]) == 20
    assert universe.sample_id(1) == "b"
    assert universe.global_index(1) == 101
    assert universe.cost_row(1) == ("cost", 20)
    assert tuple(
        tuple(group)
        for group in universe._shuffle(
            shuffle=True,
            seed=0,
            epoch=0,
            num_replicas=1,
            rank=0,
        )
    ) == ((2, 3), (0, 1))


def test_universe_context_exit_and_close_share_one_ownership_claim() -> None:
    dataset = _ContextDataset((10,), ("a",))
    universe = DatasetUniverse(dataset)
    error = RuntimeError("context failed")

    with pytest.raises(RuntimeError, match="context failed"):
        with universe:
            raise error
    universe.close()

    assert dataset.close_calls == 1
    assert len(dataset.exit_calls) == 1
    exc_type, exc, traceback = dataset.exit_calls[0]
    assert exc_type is RuntimeError
    assert exc is error
    assert traceback is not None


def test_selection_view_intersects_ordered_static_selections() -> None:
    universe = DatasetUniverse(
        _Dataset((10, 20, 30, 40), ("a", "b", "c", "d"), global_offset=100)
    )
    language = DecisionSet(universe, ("ok", "ok", "bad", "ok")).select("ok")
    quality = DecisionSet(universe, ("bad", "ok", "ok", "ok")).select("ok")
    view = SelectionView(universe, (language, quality))

    assert view.selections == (language, quality)
    assert view.indices == (1, 3)
    assert view.universe_index(0) == 1
    assert view.universe_index(-1) == 3
    assert [_value(sample) for sample in view] == [20, 40]
    assert view.global_index(0) == 101
    assert view.cost_row(1) == ("cost", 40)

    first = _shuffled(view, 0)
    second = _shuffled(view, 1)
    assert set(first).isdisjoint(second)
    assert set(first) | set(second) == {0, 1}


def test_unknown_decision_is_not_rejection() -> None:
    universe = DatasetUniverse(_Dataset((10, 20, 30), ("a", "b", "c")))
    selection = DecisionSet(universe, ("ok", None, "bad")).select("ok")
    view = SelectionView(universe, (selection,))

    assert selection.contains(0) is True
    assert selection.contains(1) is None
    assert selection.contains(2) is False
    assert view.universe_index(0) == 0
    with pytest.raises(UnknownDecisionError, match="universe index 1"):
        view.universe_index(1)
    with pytest.raises(UnknownDecisionError, match="universe index 1"):
        len(view)


def test_false_in_intersection_dominates_unknown() -> None:
    universe = DatasetUniverse(_Dataset((10, 20), ("a", "b")))
    pending = DecisionSet(universe, (None, "ok")).select("ok")
    rejected = DecisionSet(universe, ("bad", "ok")).select("ok")
    view = SelectionView(universe, (pending, rejected))

    assert view.contains(0) is False
    assert view.indices == (1,)


def test_one_to_one_rebase_uses_sample_lineage_not_dense_position() -> None:
    source = DatasetUniverse(_Dataset((10, 20, 30, 40), ("a", "b", "c", "d")))
    target = DatasetUniverse(
        _Dataset((300, 100, 400, 200), ("c", "a", "d", "b"), global_offset=50)
    )
    selected = DecisionSet(source, ("bad", "ok", "bad", "ok")).select("ok")

    rebased = SelectionView(source, (selected,)).rebase(target)

    assert rebased.universe is target
    assert rebased.indices == (2, 3)
    assert [_value(sample) for sample in rebased] == [400, 200]
    assert rebased.global_index(0) == 52


def test_rebased_live_selection_waits_on_demand_and_for_complete_operations() -> None:
    source = DatasetUniverse(_Dataset((10, 20, 30), ("a", "b", "c")))
    target = DatasetUniverse(_Dataset((300, 100, 200), ("c", "a", "b")))
    selected = _DeferredSelection(source, (False, True, True))

    rebased = SelectionView(source, (selected,)).rebase(target)

    assert rebased.universe_index(0) == 0
    assert selected.decision_waits == [2]
    assert selected.complete_waits == 0
    assert len(rebased) == 2
    assert rebased.indices == (0, 2)
    assert rebased.universe_index(-1) == 2
    assert selected.complete_waits >= 1


def test_rebased_view_closes_each_universe_and_selection_once() -> None:
    source_dataset = _ClosableDataset((10, 20), ("a", "b"))
    target_dataset = _ClosableDataset((200, 100), ("b", "a"))
    source = DatasetUniverse(source_dataset)
    target = DatasetUniverse(target_dataset)
    selected = _ClosableSelection(source, (True, False))
    rebased = SelectionView(source, (selected,)).rebase(target)

    rebased.close()
    rebased.close()

    assert source_dataset.close_calls == 1
    assert target_dataset.close_calls == 1
    assert selected.close_calls == 1


def test_rebase_rejects_non_one_to_one_lineage() -> None:
    source = DatasetUniverse(_Dataset((10, 20), ("a", "b")))
    target = DatasetUniverse(_Dataset((30, 40), ("a", "c")))
    selected = DecisionSet(source, ("ok", "bad")).select("ok")

    with pytest.raises(ValueError, match="missing sample_id 'b'"):
        selected.rebase(target)
