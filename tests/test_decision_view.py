from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from anydataset.dataset import MapStyleABC
from anydataset.dataset._snapshot import SnapshotCatalogDataset, SnapshotSegment
from anydataset.filter import FilterRule
from anydataset.types import Modality, Role, Sample, TextItem, TextView


class _Rows(MapStyleABC):
    def __init__(self, snapshot_id: str, values: Sequence[int]) -> None:
        self.snapshot_id = snapshot_id
        self.values = tuple(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Sample:
        return cast(
            Sample,
            {
            (Role.DEFAULT, Modality.TEXT): TextItem(
                views={TextView.TEXT: str(self.values[index])}
            )
            },
        )

    def sample_id(self, index: int) -> str:
        return f"sample-{self.values[index]}"

    def universe_id(self) -> str:
        return f"rows:{self.snapshot_id}"


def _source() -> SnapshotCatalogDataset:
    first = _Rows("first", (0, 1, 2))
    second = _Rows("second", (3, 4))
    return SnapshotCatalogDataset(
        (
            SnapshotSegment("first", 0, len(first), first),
            SnapshotSegment("second", len(first), len(second), second),
        ),
        sealed=False,
        universe_id="rows-prefix",
    )


def _value(sample: Sample) -> int:
    item = cast(TextItem, sample[Role.DEFAULT, Modality.TEXT])
    return int(cast(str, item.views[TextView.TEXT]))


def _rule_factory():
    return lambda sample: "accept" if _value(sample) % 2 == 0 else "review"


def test_decisions_publish_one_internal_segment_per_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "home"))
    view = FilterRule("parity", _rule_factory).bind(dataset_factory=_source)

    assert view.status().completed_samples == 0
    first = view.produce(device="cpu")

    assert first.expected_samples == 5
    assert first.completed_samples == 3
    first_prefix = view.load()
    try:
        assert [_value(sample) for sample in first_prefix] == [0, 2]

        second = view.produce(device="cpu")

        assert second.complete
        assert second.completed_samples == 5
        assert [_value(sample) for sample in first_prefix] == [0, 2]
        with view.load() as accepted:
            assert [_value(sample) for sample in accepted] == [0, 2, 4]
    finally:
        first_prefix.close()
    with view.select("review").load() as review:
        assert [_value(sample) for sample in review] == [1, 3]


def test_snapshot_id_separates_segments_with_the_same_universe_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "home"))
    first = _Rows("shared", (0, 1))
    second = _Rows("shared", (2, 3))

    def source() -> SnapshotCatalogDataset:
        return SnapshotCatalogDataset(
            (
                SnapshotSegment("first", 0, len(first), first),
                SnapshotSegment("second", len(first), len(second), second),
            ),
            sealed=True,
            universe_id="rows-prefix",
        )

    view = FilterRule("shared-parity", _rule_factory).bind(dataset_factory=source)

    first_status = view.produce(device="cpu", write_workers=0)
    second_status = view.produce(device="cpu", write_workers=0)

    assert first_status.completed_samples == 2
    assert second_status.complete
    with view.load() as accepted:
        assert [_value(sample) for sample in accepted] == [0, 2]


def test_decision_snapshot_completes_when_selected_partition_is_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "home"))
    view = FilterRule("reject-all", lambda: lambda _sample: "reject").bind(
        dataset_factory=lambda: SnapshotCatalogDataset(
            (SnapshotSegment("only", 0, 2, _Rows("only", (0, 1))),),
            sealed=True,
            universe_id="only-prefix",
        )
    )

    status = view.produce(device="cpu")

    assert status.complete
    assert status.completed_samples == 2
    with view.load() as accepted:
        assert len(accepted) == 0


def test_plain_map_dataset_is_one_logical_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ANYDATASET_HOME", str(tmp_path / "home"))
    view = FilterRule("plain-parity", _rule_factory).bind(
        dataset_factory=lambda: _Rows("plain", (0, 1, 2)),
    )

    status = view.produce(device="cpu", write_workers=0)

    assert status.expected_samples == 3
    assert status.completed_samples == 3
    with view.load() as accepted:
        assert [_value(sample) for sample in accepted] == [0, 2]
