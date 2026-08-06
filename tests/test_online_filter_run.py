from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

import pytest

from anydataset import AnyDataset, FilterRule, Spec, anydataset_home, register_source
from anydataset.dataset.source._registry import source_exists
from anydataset.dataset.abc import MapStyleABC
from anydataset.dataset.universe import DatasetUniverse
from anydataset.dataset.view import DecisionSet, SelectionView
from anydataset.filter import FilterRunStatus
from anydataset.types import AudioItem, AudioView, Modality, Role


class _RowsSource:
    def prepare(self, spec: Spec, cache_path: Path):
        del cache_path
        return [{"value": value} for value in spec.load_options["values"]]


class _Ids:
    def sample_id(self, index: int) -> str:
        return f"sample-{index}"


class _LogicalDataset(MapStyleABC):
    def __init__(self, values: list[int], reads: list[int]) -> None:
        self.values = tuple(values)
        self.reads = reads

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        value = self.values[index]
        self.reads.append(value)
        return _sample(value)

    def sample_id(self, index: int) -> str:
        return f"logical-{index}"


class _IdentifiedLogicalDataset(_LogicalDataset):
    def universe_id(self) -> str:
        return "materialized-v1:test-universe"


class _RecordingPredicate:
    def __init__(
        self,
        calls: list[int],
        lock: Lock,
        *,
        delay: float = 0.0,
        fail_at: int | None = None,
    ) -> None:
        self.calls = calls
        self.lock = lock
        self.delay = delay
        self.fail_at = fail_at

    def __call__(self, sample):
        value = _value(sample)
        with self.lock:
            self.calls.append(value)
        if self.delay:
            time.sleep(self.delay)
        if value == self.fail_at:
            raise RuntimeError("filter exploded")
        return value % 2 == 0


class _RecordingFactory:
    def __init__(
        self,
        calls: list[int],
        lock: Lock,
        *,
        delay: float = 0.0,
        fail_at: int | None = None,
    ) -> None:
        self.calls = calls
        self.lock = lock
        self.delay = delay
        self.fail_at = fail_at

    def __call__(self):
        return _RecordingPredicate(
            self.calls,
            self.lock,
            delay=self.delay,
            fail_at=self.fail_at,
        )


def test_open_returns_live_selection_and_finishes_full_universe() -> None:
    dataset = _dataset("online_filter_full", list(range(8)))
    calls: list[int] = []
    rule = FilterRule(
        "even-online",
        _RecordingFactory(calls, Lock(), delay=0.01),
    )

    run = rule.open(
        dataset_factory=lambda: dataset,
        labels=True,
        device="cpu",
        commit_samples=1,
        write_workers=0,
    )

    assert _value(run.dataset[0]) == 0
    assert [_value(sample) for sample in run.wait()] == [0, 2, 4, 6]
    assert run.status is FilterRunStatus.COMPLETE
    assert sorted(calls) == list(range(8))
    run.close()


def test_upstream_selection_only_intersects_return_boundary() -> None:
    dataset = _dataset("online_filter_upstream", list(range(6)))
    universe = DatasetUniverse(dataset, sample_identity=_Ids())
    upstream = DecisionSet(
        universe,
        ("keep", "drop", "keep", "drop", "keep", "drop"),
    ).select("keep")
    view = SelectionView(universe, (upstream,))
    calls: list[int] = []
    rule = FilterRule("full-scan", _RecordingFactory(calls, Lock()))

    run = rule.open(
        dataset_factory=lambda: view,
        labels=True,
        device="cpu",
        commit_samples=1,
        write_workers=0,
    )

    assert [_value(sample) for sample in run.wait()] == [0, 2, 4]
    assert sorted(calls) == list(range(6))
    run.close()


def test_len_negative_index_and_shuffle_wait_for_complete() -> None:
    dataset = _dataset("online_filter_complete_ops", list(range(5)))
    run = FilterRule(
        "complete-ops",
        _RecordingFactory([], Lock(), delay=0.005),
    ).open(
        dataset_factory=lambda: dataset,
        labels=True,
        device="cpu",
        commit_samples=1,
        write_workers=0,
    )

    assert len(run.dataset) == 3
    assert _value(run.dataset[-1]) == 4
    shuffled = tuple(
        index
        for group in run.dataset._shuffle(
            shuffle=True,
            seed=1,
            epoch=0,
            num_replicas=1,
            rank=0,
        )
        for index in group
    )
    assert set(shuffled) == {0, 1, 2}
    run.close()


def test_failure_is_sticky_and_keeps_arrow_decision_fragments() -> None:
    dataset = _dataset("online_filter_failure", list(range(5)))
    run = FilterRule(
        "failure",
        _RecordingFactory([], Lock(), fail_at=2),
    ).open(
        dataset_factory=lambda: dataset,
        labels=True,
        device="cpu",
        commit_samples=1,
        write_workers=0,
    )

    with pytest.raises(RuntimeError, match="filter exploded"):
        run.wait()
    assert run.status is FilterRunStatus.FAILED
    with pytest.raises(RuntimeError, match="filter exploded"):
        len(run.dataset)
    with pytest.raises(RuntimeError, match="filter exploded"):
        run.close()
    assert tuple(anydataset_home().glob("cache/filters/**/decisions/*.arrow"))


def test_ready_cache_is_opened_without_predicate_calls() -> None:
    dataset = _dataset("online_filter_ready", list(range(4)))
    first_calls: list[int] = []
    rule = FilterRule("ready", _RecordingFactory(first_calls, Lock()))
    first = rule.open(
        dataset_factory=lambda: dataset,
        labels=True,
        device="cpu",
        commit_samples=1,
        write_workers=0,
    )
    first.wait()
    first.close()

    second_calls: list[int] = []
    ready_rule = FilterRule("ready", _RecordingFactory(second_calls, Lock()))
    second = ready_rule.open(
        dataset_factory=lambda: dataset,
        labels=True,
        device="cpu",
    )

    assert second.status is FilterRunStatus.COMPLETE
    assert [_value(sample) for sample in second.dataset] == [0, 2]
    assert second_calls == []
    second.close()


def test_logical_selection_requires_explicit_input_id() -> None:
    logical = _LogicalDataset(list(range(4)), [])
    universe = DatasetUniverse(logical)
    view = SelectionView(
        universe,
        (DecisionSet(universe, (True, False, True, False)).select(True),),
    )

    with pytest.raises(ValueError, match="input_id is required"):
        FilterRule("logical-id", _RecordingFactory([], Lock())).open(
            dataset_factory=lambda: view,
            labels=True,
            device="cpu",
        )

    with pytest.raises(TypeError, match="AnyDataset"):
        FilterRule("logical-apply", _RecordingFactory([], Lock())).apply(
            dataset_factory=lambda: logical,
            input_id="logical-materialization-v1",
            device="cpu",
        )


def test_logical_materialized_selection_filters_complete_universe() -> None:
    reads: list[int] = []
    logical = _LogicalDataset(list(range(6)), reads)
    universe = DatasetUniverse(logical)
    upstream = DecisionSet(
        universe,
        (True, False, True, False, True, False),
    ).select(True)
    first_view = SelectionView(universe, (upstream,))
    calls: list[int] = []
    rule = FilterRule("logical-even", _RecordingFactory(calls, Lock()))

    first = rule.open(
        dataset_factory=lambda: first_view,
        labels=True,
        input_id="logical-materialization-v1",
        device="cpu",
        commit_samples=1,
        write_workers=0,
    )

    assert [_value(sample) for sample in first.wait()] == [0, 2, 4]
    assert sorted(calls) == list(range(6))
    first.close()

    other_selection = DecisionSet(
        universe,
        (False, True, False, True, False, True),
    ).select(True)
    second_view = SelectionView(universe, (other_selection,))
    second_calls: list[int] = []
    second = FilterRule(
        "logical-even",
        _RecordingFactory(second_calls, Lock()),
    ).open(
        dataset_factory=lambda: second_view,
        labels=True,
        input_id="logical-materialization-v1",
        device="cpu",
    )

    assert second.status is FilterRunStatus.COMPLETE
    assert list(second.dataset) == []
    assert second_calls == []
    second.close()


def test_universe_id_reuses_filter_cache_across_runtime_dataset_types() -> None:
    first_calls: list[int] = []
    first_dataset = _IdentifiedLogicalDataset(list(range(4)), [])
    first = FilterRule(
        "universe-id",
        _RecordingFactory(first_calls, Lock()),
    ).open(
        dataset_factory=lambda: first_dataset,
        labels=True,
        device="cpu",
        commit_samples=1,
        write_workers=0,
    )
    assert [_value(sample) for sample in first.wait()] == [0, 2]
    first.close()

    class _ReadyDataset(_IdentifiedLogicalDataset):
        pass

    second_calls: list[int] = []
    second_dataset = _ReadyDataset(list(range(4)), [])
    second = FilterRule(
        "universe-id",
        _RecordingFactory(second_calls, Lock()),
    ).open(
        dataset_factory=lambda: second_dataset,
        labels=True,
        device="cpu",
    )
    assert [_value(sample) for sample in second.dataset] == [0, 2]
    assert second_calls == []
    second.close()


def _dataset(source: str, values: list[int]) -> AnyDataset:
    if not source_exists(source):
        register_source(source, _RowsSource)
    return AnyDataset(
        Spec(source=source, path="/tmp/rows", load_options={"values": values}),
        parse_fn=_parse,
    )


def _parse(row):
    return _sample(row["value"])


def _sample(value: int):
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(views={AudioView.WAVEFORM: value})
    }


def _value(sample) -> int:
    return sample[Role.DEFAULT, Modality.AUDIO].views[AudioView.WAVEFORM]
