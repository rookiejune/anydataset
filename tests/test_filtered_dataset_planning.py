from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from anydataset import AnyDataset, FilterRule, Spec, register_source
from anydataset.dataset import MapStyleABC
from anydataset.dataset.source._registry import source_exists
from anydataset.types import AudioItem, AudioView, Modality, Role, Sample


_SOURCE = "unit_test_filtered_dataset_planning"


class _PlanningRows(MapStyleABC):
    def __init__(self, values: Sequence[int]) -> None:
        self.values = tuple(values)
        self.payload_calls: list[int] = []
        self.cost_calls: list[int] = []

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Sample:
        self.payload_calls.append(index)
        return {
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: self.values[index]},
            )
        }

    def cost_row(self, index: int) -> dict[str, int]:
        self.cost_calls.append(index)
        return {"frames": (self.values[index] + 1) * 10}

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        if num_replicas != 1 or rank != 0:
            raise AssertionError("selection must request the complete base ordering")
        groups = (
            ((4, 5, 6, 7), (0, 1, 2, 3)) if shuffle else ((0, 1, 2, 3), (4, 5, 6, 7))
        )
        yield from groups


class _PlanningSource:
    def prepare(self, spec: Spec, cache_path: Path) -> _PlanningRows:
        del cache_path
        return _PlanningRows(spec.load_options["values"])


def _dataset() -> AnyDataset:
    if not source_exists(_SOURCE):
        register_source(_SOURCE, _PlanningSource)
    return AnyDataset(
        Spec(
            source=_SOURCE,
            path="/tmp/filtered-planning",
            load_options={"values": list(range(8))},
        )
    )


def _value(sample: Sample) -> int:
    return sample[Role.DEFAULT, Modality.AUDIO].views[AudioView.WAVEFORM]


def _selection_factory():
    return lambda sample: _value(sample) not in {2, 5}


def _edge_factory():
    return lambda sample: _value(sample) in {0, 3, 4, 7}


def _filtered(dataset: AnyDataset):
    return (
        FilterRule("planning-selection", _selection_factory)
        .apply(
            dataset_factory=lambda: dataset,
            device="cpu",
        )
        .select_by("accept")
    )


def _groups(dataset, *, rank: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(group)
        for group in dataset._shuffle(
            shuffle=True,
            seed=0,
            epoch=0,
            num_replicas=2,
            rank=rank,
        )
    )


def test_cost_row_delegates_without_reading_selected_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "home"))
    dataset = _dataset()
    filtered = _filtered(dataset)
    rows = dataset.dataset
    assert isinstance(rows, _PlanningRows)
    rows.payload_calls.clear()
    rows.cost_calls.clear()

    cost = filtered.cost_row(2)

    assert cost == {"frames": 40}
    assert rows.payload_calls == []
    assert rows.cost_calls == [3]


def test_shuffle_preserves_groups_and_balances_selected_positions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "home"))
    filtered = _filtered(_dataset())

    first = _groups(filtered, rank=0)
    second = _groups(filtered, rank=1)

    assert first == ((3, 5), (1,))
    assert second == ((4,), (0, 2))
    assert sum(map(len, first)) == sum(map(len, second)) == 3
    assert {index for group in first for index in group}.isdisjoint(
        index for group in second for index in group
    )
    assert {
        index for groups in (first, second) for group in groups for index in group
    } == set(range(len(filtered)))


def test_chained_filter_shuffle_uses_physical_index_space(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "home"))
    upstream = _filtered(_dataset())
    filtered = (
        FilterRule("planning-edge", _edge_factory)
        .apply(
            dataset_factory=upstream.dataset_factory,
            device="cpu",
        )
        .select_by("accept")
    )

    first = _groups(filtered, rank=0)
    second = _groups(filtered, rank=1)

    assert filtered.indices == (0, 3, 4, 7)
    assert first == ((2,), (0,))
    assert second == ((3,), (1,))
    assert {
        index for groups in (first, second) for group in groups for index in group
    } == {
        0,
        1,
        2,
        3,
    }
