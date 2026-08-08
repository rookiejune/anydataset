from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from anydataset.dataset import MapStyleABC
from anydataset.filter import FilterRule
from anydataset.types import Modality, Role, Sample, TextItem, TextView


_TEXT = (Role.DEFAULT, Modality.TEXT)


class _Source(MapStyleABC):
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Sample:
        return {_TEXT: TextItem(views={TextView.TEXT: str(index)}, meta={})}

    def sample_id(self, index: int) -> str:
        return f"source-{index}"


def _value(sample: Sample) -> int:
    item = cast(TextItem, sample[_TEXT])
    return int(cast(str, item.views[TextView.TEXT]))


def test_decision_windows_publish_and_pin_committed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "home"))
    calls: list[int] = []
    decisions = FilterRule(
        "committed-prefix",
        lambda: lambda sample: calls.append(_value(sample)) or "accept",
    ).bind(
        dataset_factory=lambda: _Source(5),
        input_id="source-v1",
        max_new_samples=2,
    )

    first_status = decisions.produce(
        device="cpu",
        batch_size=2,
        commit_samples=1,
        write_workers=0,
    )
    first = decisions.load()

    assert first_status.expected_samples == 5
    assert first_status.completed_samples == 2
    assert first_status.pending_samples == 3
    assert [_value(first[index]) for index in range(len(first))] == [0, 1]
    assert not first.sealed
    assert calls == [0, 1]

    second_status = decisions.produce(
        device="cpu",
        batch_size=2,
        commit_samples=1,
        write_workers=0,
    )
    second = decisions.load()

    assert len(first) == 2
    assert second_status.completed_samples == 4
    assert second_status.pending_samples == 1
    assert [_value(second[index]) for index in range(len(second))] == [0, 1, 2, 3]
    assert not second.sealed
    assert calls == [0, 1, 2, 3]

    final_status = decisions.produce(
        device="cpu",
        batch_size=2,
        commit_samples=1,
        write_workers=0,
    )
    final = decisions.load()

    assert final_status.complete
    assert final_status.completed_samples == 5
    assert [_value(final[index]) for index in range(len(final))] == [0, 1, 2, 3, 4]
    assert final.sealed
    assert calls == [0, 1, 2, 3, 4]

    final.close()
    second.close()
    first.close()


@pytest.mark.parametrize("value", [True, 0, -1])
def test_decision_max_new_samples_validation(value: object) -> None:
    rule = FilterRule("all", lambda: lambda _sample: "accept")

    with pytest.raises((TypeError, ValueError)):
        rule.bind(
            dataset_factory=lambda: _Source(1),
            input_id="source-v1",
            max_new_samples=value,  # type: ignore[arg-type]
        )
