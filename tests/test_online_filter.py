from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator, Sequence

import pytest

from anydataset.dataset import MapStyleABC
from anydataset.filter import FilterDecision, RejectReplaceDataset
from anydataset.types import AudioItem, AudioView, Modality, Role, Sample


class _Dataset(MapStyleABC):
    def __init__(self, values: Sequence[int]) -> None:
        self.values = tuple(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Sample:
        return {
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: self.values[index]}
            )
        }

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        del shuffle, seed, epoch
        yield range(rank, len(self), num_replicas)


def _value(sample: Sample) -> int:
    return sample[Role.DEFAULT, Modality.AUDIO].views[AudioView.WAVEFORM]


def _even(sample: Sample) -> bool:
    return _value(sample) % 2 == 0


def test_keeps_accept_and_preserves_length() -> None:
    wrapped = RejectReplaceDataset(
        _Dataset(range(6)),
        _even,
        buffer_size=4,
        min_buffer=2,
        seed=0,
    )

    assert len(wrapped) == 6
    assert _value(wrapped[0]) == 0
    assert _value(wrapped[2]) == 2


def test_look_ahead_replaces_reject_with_neighbor_accept() -> None:
    wrapped = RejectReplaceDataset(
        _Dataset(range(6)),
        _even,
        buffer_size=4,
        min_buffer=2,
        max_probe=4,
        seed=0,
    )

    assert _value(wrapped[1]) == 2
    assert wrapped.global_index(1) == 2


def test_global_index_tracks_served_sample_after_replace() -> None:
    wrapped = RejectReplaceDataset(
        _Dataset(range(6)),
        _even,
        buffer_size=4,
        min_buffer=2,
        max_probe=4,
        seed=0,
    )

    assert wrapped.global_index(0) == 0
    _ = wrapped[0]
    assert wrapped.global_index(0) == 0
    _ = wrapped[1]
    assert wrapped.global_index(1) == 2


def test_buffer_fallback_after_probes_exhausted() -> None:
    values = [0, 1, 3, 5, 7, 9]
    wrapped = RejectReplaceDataset(
        _Dataset(values),
        _even,
        buffer_size=4,
        min_buffer=1,
        max_probe=1,
        seed=0,
        stats_window=100,
        max_reject_ratio=1.0,
        warn_reject_ratio=1.0,
    )
    assert _value(wrapped[0]) == 0
    assert wrapped.buffer_filled == 1
    assert _value(wrapped[1]) == 0


def test_cold_start_raises_when_probe_and_buffer_fail() -> None:
    wrapped = RejectReplaceDataset(
        _Dataset([1, 3, 5, 7]),
        _even,
        buffer_size=4,
        min_buffer=1,
        max_probe=2,
        seed=0,
        stats_window=100,
        max_reject_ratio=1.0,
        warn_reject_ratio=1.0,
    )

    with pytest.raises(RuntimeError, match="could not find an accept sample"):
        wrapped[0]


def test_high_reject_ratio_raises() -> None:
    wrapped = RejectReplaceDataset(
        _Dataset([0, 1, 0, 1, 0, 1, 0, 1]),
        _even,
        buffer_size=8,
        min_buffer=1,
        max_probe=8,
        stats_window=4,
        warn_reject_ratio=0.25,
        max_reject_ratio=0.4,
        seed=0,
    )
    wrapped[0]
    wrapped[1]
    wrapped[2]
    with pytest.raises(RuntimeError, match="exceeds max_reject_ratio"):
        wrapped[3]


def test_warns_on_elevated_reject_ratio(caplog: pytest.LogCaptureFixture) -> None:
    wrapped = RejectReplaceDataset(
        _Dataset([0, 1, 0, 1, 0, 1, 0, 1]),
        _even,
        name="unit_online",
        buffer_size=8,
        min_buffer=1,
        max_probe=8,
        stats_window=4,
        warn_reject_ratio=0.2,
        max_reject_ratio=1.0,
        warn_cooldown_s=0.0,
        seed=0,
    )
    with caplog.at_level(logging.WARNING, logger="anydataset.filter.online"):
        wrapped[0]
        wrapped[1]
        wrapped[2]
        wrapped[3]

    assert any("unit_online" in record.getMessage() for record in caplog.records)


def test_accepts_filter_decision_and_non_accept_labels() -> None:
    def predicate(sample: Sample) -> FilterDecision:
        value = _value(sample)
        if value % 2 == 0:
            return FilterDecision(label=True, metrics={})
        return FilterDecision(label="review", metrics={"reason": "odd"})

    wrapped = RejectReplaceDataset(
        _Dataset(range(4)),
        predicate,
        buffer_size=4,
        min_buffer=1,
        seed=0,
    )
    assert _value(wrapped[1]) == 2


def test_pickle_roundtrip_keeps_worker_local_state() -> None:
    wrapped = RejectReplaceDataset(
        _Dataset(range(6)),
        _even,
        buffer_size=4,
        min_buffer=2,
        seed=1,
    )
    wrapped[0]
    restored = pickle.loads(pickle.dumps(wrapped))
    assert len(restored) == 6
    assert restored.buffer_filled == 1
    assert _value(restored[2]) == 2


def test_delegates_shuffle_to_base() -> None:
    wrapped = RejectReplaceDataset(
        _Dataset(range(5)),
        _even,
        buffer_size=4,
        min_buffer=1,
        seed=0,
    )
    groups = list(
        wrapped._shuffle(
            shuffle=True,
            seed=0,
            epoch=0,
            num_replicas=2,
            rank=1,
        )
    )
    assert groups == [range(1, 5, 2)]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"buffer_size": 0}, "buffer_size"),
        ({"min_buffer": 8, "buffer_size": 4}, "min_buffer"),
        ({"update_prob": 1.5}, "update_prob"),
        ({"warn_reject_ratio": 0.5, "max_reject_ratio": 0.2}, "warn_reject_ratio"),
    ],
)
def test_rejects_invalid_options(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        RejectReplaceDataset(_Dataset(range(3)), _even, **kwargs)  # type: ignore[arg-type]
