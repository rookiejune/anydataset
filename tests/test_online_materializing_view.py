from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from anydataset.cache import FileLock, FileLockError
from anydataset.dataset.abc import MapStyleABC
from anydataset.dataset.universe import DatasetUniverse
from anydataset.dataset.view import DecisionSet, SelectionView
from anydataset.store import DatasetWriter, MaterializationStatus, ViewMaterializer
from anydataset.store.materialize.snapshots import (
    CATALOG_FILENAME,
    producer_lock_path,
)
from anydataset.types import AudioItem, AudioMeta, AudioReq, AudioView, Modality, Role


_REF = (Role.DEFAULT, Modality.AUDIO)
_SCHEMA = {
    _REF: AudioReq(
        views=frozenset({AudioView.LONGCAT}),
        meta=frozenset(),
    )
}


class _Source(MapStyleABC):
    def __init__(self, count: int, *, sealed: bool = True) -> None:
        self.count = count
        self.sealed = sealed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        return {
            _REF: AudioItem(
                views={
                    AudioView.WAVEFORM: (
                        torch.tensor([[float(index)]], dtype=torch.float32),
                        16_000,
                    )
                },
                meta={AudioMeta.DURATION: 1.0},
            )
        }

    def sample_id(self, index: int) -> str:
        return f"source-{index}"


@dataclass
class _SourceState:
    count: int
    sealed: bool = True
    calls: int = 0


@dataclass(frozen=True)
class _SourceFactory:
    state: _SourceState

    def __call__(self) -> _Source:
        self.state.calls += 1
        return _Source(self.state.count, sealed=self.state.sealed)


@dataclass
class _ProviderState:
    factories: int = 0
    samples: list[int] | None = None

    def __post_init__(self) -> None:
        if self.samples is None:
            self.samples = []


@dataclass(frozen=True)
class _ProviderFactory:
    state: _ProviderState
    fail_at: int | None = None

    def __call__(self, device: str) -> _Provider:
        self.state.factories += 1
        return _Provider(self.state, device, fail_at=self.fail_at)


@dataclass
class _Provider:
    state: _ProviderState
    device: str
    fail_at: int | None = None

    output = AudioView.LONGCAT

    def __call__(self, views: dict[Any, Any]) -> dict[str, torch.Tensor]:
        waveform, _sample_rate = views[AudioView.WAVEFORM]
        index = int(waveform.item())
        if index == self.fail_at:
            raise RuntimeError("provider failed")
        samples = self.state.samples
        if samples is None:
            raise RuntimeError("provider state was not initialized")
        samples.append(index)
        return {"semantic_codes": waveform.to(torch.int64) + 10}


def _materializer(output: Path, staging: Path, **options: Any) -> ViewMaterializer:
    return ViewMaterializer(
        output,
        staging_dir=staging,
        split="train",
        dataset_id="train",
        input_id="source-v1",
        provider_id="provider-v1",
        output=AudioView.LONGCAT,
        schema=_SCHEMA,
        commit_samples=1,
        write_workers=0,
        **options,
    )


def _codes(sample: Any) -> int:
    item = cast(AudioItem, sample[_REF])
    value = cast(dict[str, torch.Tensor], item.views[AudioView.LONGCAT])
    return int(value["semantic_codes"].item())


def _fail_factory() -> Any:
    raise AssertionError("consumer must not construct its source")


def test_consumer_open_is_read_only_before_first_snapshot(tmp_path: Path) -> None:
    materializer = _materializer(tmp_path / "output", tmp_path / "staging")

    dataset = materializer.open(dataset_factory=_fail_factory)
    try:
        assert len(dataset) == 0
    finally:
        dataset.close()

    assert not (tmp_path / "output" / CATALOG_FILENAME).exists()


def test_reopened_epoch_sees_new_prefix_while_old_epoch_is_fixed(
    tmp_path: Path,
) -> None:
    source_state = _SourceState(3, sealed=False)
    provider_state = _ProviderState()
    source = _SourceFactory(source_state)
    provider = _ProviderFactory(provider_state)
    materializer = _materializer(tmp_path / "output", tmp_path / "staging")

    first = materializer.produce(
        dataset_factory=source,
        provider_factory=provider,
        snapshot_samples=2,
    )
    assert first == MaterializationStatus(
        output_dir=tmp_path / "output",
        expected=3,
        completed=3,
        finalized=False,
    )
    epoch_zero = materializer.open(dataset_factory=_fail_factory)
    assert len(epoch_zero) == 3
    assert [_codes(epoch_zero[index]) for index in range(3)] == [10, 11, 12]

    source_state.count = 5
    source_state.sealed = True
    second = materializer.produce(
        dataset_factory=source,
        provider_factory=provider,
        snapshot_samples=2,
    )
    assert second.completed == 5
    assert second.finalized
    assert len(epoch_zero) == 3

    epoch_one = materializer.open(dataset_factory=_fail_factory)
    try:
        assert len(epoch_one) == 5
        assert [
            _codes(sample)
            for sample in epoch_one.__getitems__([4, 0, 3, 4])
        ] == [14, 10, 13, 14]
        assert [epoch_one.sample_id(index) for index in range(5)] == [
            f"source-{index}" for index in range(5)
        ]
    finally:
        epoch_zero.close()
        epoch_one.close()

    assert provider_state.factories == 2
    assert provider_state.samples == [0, 1, 2, 3, 4]


def test_provider_failure_does_not_expose_partial_fragment(tmp_path: Path) -> None:
    state = _SourceState(3)
    materializer = _materializer(tmp_path / "output", tmp_path / "staging")

    with pytest.raises(RuntimeError, match="provider failed"):
        materializer.produce(
            dataset_factory=_SourceFactory(state),
            provider_factory=_ProviderFactory(_ProviderState(), fail_at=1),
            snapshot_samples=2,
        )

    dataset = materializer.open(dataset_factory=_fail_factory)
    try:
        assert len(dataset) == 0
    finally:
        dataset.close()


def test_resume_reuses_durable_fragments_and_publication_is_idempotent(
    tmp_path: Path,
) -> None:
    source = _SourceFactory(_SourceState(3))
    materializer = _materializer(tmp_path / "output", tmp_path / "staging")
    failed_state = _ProviderState()
    with pytest.raises(RuntimeError, match="provider failed"):
        materializer.produce(
            dataset_factory=source,
            provider_factory=_ProviderFactory(failed_state, fail_at=1),
            snapshot_samples=2,
        )
    assert failed_state.samples == [0]

    resumed_state = _ProviderState()
    materializer.produce(
        dataset_factory=source,
        provider_factory=_ProviderFactory(resumed_state),
        snapshot_samples=2,
    )
    assert resumed_state.samples == [1, 2]

    unused = _ProviderState()
    materializer.produce(
        dataset_factory=source,
        provider_factory=_ProviderFactory(unused),
        snapshot_samples=2,
    )
    assert unused.factories == 0


def test_selection_projects_onto_published_prefix(tmp_path: Path) -> None:
    state = _SourceState(3, sealed=False)
    materializer = _materializer(tmp_path / "output", tmp_path / "staging")
    materializer.produce(
        dataset_factory=_SourceFactory(state),
        provider_factory=_ProviderFactory(_ProviderState()),
        snapshot_samples=2,
    )

    def selection_factory() -> SelectionView:
        universe = DatasetUniverse(_Source(5, sealed=False))
        selected = DecisionSet(
            universe,
            (True, False, True, False, True),
        ).select(True)
        return SelectionView(universe, (selected,))

    dataset = materializer.open(
        dataset_factory=_fail_factory,
        selection_factory=selection_factory,
    )
    try:
        assert len(dataset) == 2
        assert [dataset.sample_id(index) for index in range(2)] == [
            "source-0",
            "source-2",
        ]
    finally:
        dataset.close()


def test_shared_root_rejects_identity_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "output"
    source = _SourceFactory(_SourceState(1))
    materializer = _materializer(output, tmp_path / "staging")
    materializer.produce(
        dataset_factory=source,
        provider_factory=_ProviderFactory(_ProviderState()),
    )

    mismatch = ViewMaterializer(
        output,
        staging_dir=tmp_path / "other-staging",
        split="train",
        dataset_id="train",
        input_id="different-source",
        provider_id="provider-v1",
        output=AudioView.LONGCAT,
        schema=_SCHEMA,
    )
    with pytest.raises(ValueError, match="provenance does not match"):
        mismatch.open(dataset_factory=_fail_factory)


def test_second_producer_is_rejected_by_long_lived_lease(tmp_path: Path) -> None:
    output = tmp_path / "output"
    materializer = _materializer(output, tmp_path / "staging")
    with FileLock(producer_lock_path(output)):
        with pytest.raises(FileLockError, match="already held"):
            materializer.produce(
                dataset_factory=_SourceFactory(_SourceState(1)),
                provider_factory=_ProviderFactory(_ProviderState()),
            )


def test_growing_input_may_not_shrink(tmp_path: Path) -> None:
    state = _SourceState(3, sealed=False)
    source = _SourceFactory(state)
    materializer = _materializer(tmp_path / "output", tmp_path / "staging")
    materializer.produce(
        dataset_factory=source,
        provider_factory=_ProviderFactory(_ProviderState()),
    )

    state.count = 2
    with pytest.raises(ValueError, match="growing-input staging state"):
        materializer.produce(
            dataset_factory=source,
            provider_factory=_ProviderFactory(_ProviderState()),
        )


def test_legacy_ready_store_remains_readable_without_factories(tmp_path: Path) -> None:
    output = tmp_path / "output"
    materializer = _materializer(output, tmp_path / "staging")
    output_id = materializer._output_id
    assert output_id is not None
    DatasetWriter(
        output,
        dataset_id="train",
        split="train",
        provenance={
            "input_id": "source-v1",
            "provider_id": "provider-v1",
            "output_id": output_id,
        },
    ).write(
        [
            {
                _REF: AudioItem(
                    views={
                        AudioView.LONGCAT: {
                            "semantic_codes": torch.tensor([[17]])
                        }
                    }
                )
            }
        ]
    )

    dataset = materializer.open(dataset_factory=_fail_factory)
    try:
        assert len(dataset) == 1
        assert _codes(dataset[0]) == 17
    finally:
        dataset.close()


def test_status_reads_only_published_catalog(tmp_path: Path) -> None:
    materializer = _materializer(tmp_path / "output", tmp_path / "staging")
    empty = materializer.status()
    assert (empty.expected, empty.completed, empty.finalized) == (0, 0, False)

    materializer.produce(
        dataset_factory=_SourceFactory(_SourceState(2)),
        provider_factory=_ProviderFactory(_ProviderState()),
    )
    ready = materializer.status()
    assert (ready.expected, ready.completed, ready.finalized) == (2, 2, True)
