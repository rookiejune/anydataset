from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest
import torch
import anydataset.store.materialize.fragments as materialize_fragments

from anydataset.store import (
    MaterializingViewDataset,
    SampleMaterializer,
    ViewMaterializer,
)
from anydataset.dataset.abc import MapStyleABC
from anydataset.dataset.universe import DatasetUniverse
from anydataset.dataset.view import DecisionSet, SelectionView
from anydataset.filter import FilterRule
from anydataset.store.reader import (
    StoreDataset,
    read_store_dataset,
    read_store_manifest,
)
from anydataset.types import AudioItem, AudioMeta, AudioReq, AudioView, Modality, Role


_REF = (Role.DEFAULT, Modality.AUDIO)
_SCHEMA = {
    _REF: AudioReq(
        views=frozenset({AudioView.LONGCAT}),
        meta=frozenset(),
    )
}
_SOURCE_AUDIO_REF = (Role.SOURCE, Modality.AUDIO)
_TARGET_AUDIO_REF = (Role.TARGET, Modality.AUDIO)
_PAIRED_SCHEMA = {
    _SOURCE_AUDIO_REF: AudioReq(
        views=frozenset({AudioView.WAVEFORM}),
        meta=frozenset(),
    ),
    _TARGET_AUDIO_REF: AudioReq(
        views=frozenset({AudioView.WAVEFORM}),
        meta=frozenset(),
    ),
}


class _Source(MapStyleABC):
    def __init__(self, count: int) -> None:
        self.count = count

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
                meta={AudioMeta.DURATION: float(index + 1)},
            )
        }

    def cost_row(self, index: int) -> tuple[str, int]:
        return "cost", index

    def sample_id(self, index: int) -> str:
        return f"source-{index}"

    def _shuffle(self, **_options):
        yield (2, 0, 1)


class _ClosableSource(_Source):
    def __init__(self, count: int) -> None:
        super().__init__(count)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls > 1:
            raise AssertionError("source must be closed exactly once")


class _FailingCloseSource(_Source):
    def __init__(self, count: int) -> None:
        super().__init__(count)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("source close failed")


class _ClosableResource:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls > 1:
            raise AssertionError("resource must be closed exactly once")


@dataclass(frozen=True)
class _SourceFactory:
    count: int

    def __call__(self) -> _Source:
        return _Source(self.count)


@dataclass(frozen=True)
class _ExternalIds:
    def sample_id(self, index: int) -> str:
        return f"external-{index}"


@dataclass(frozen=True)
class _SelectedSourceFactory:
    count: int

    def __call__(self) -> SelectionView:
        universe = DatasetUniverse(
            _Source(self.count),
            sample_identity=_ExternalIds(),
        )
        selected = DecisionSet(
            universe,
            tuple(index % 2 == 0 for index in range(self.count)),
        ).select(True)
        return SelectionView(universe, (selected,))


@dataclass(frozen=True)
class _EvenSourceFactory:
    count: int

    def __call__(self) -> SelectionView:
        universe = DatasetUniverse(_Source(self.count))
        selected = DecisionSet(
            universe,
            tuple(index % 2 == 0 for index in range(self.count)),
        ).select(True)
        return SelectionView(universe, (selected,))


@dataclass(frozen=True)
class _ProviderFactory:
    calls: Path
    batch: bool = False

    def __call__(self, device: str):
        if self.batch:
            return _BatchProvider(self.calls, device)
        return _Provider(self.calls, device)


class _Provider:
    output = AudioView.LONGCAT

    def __init__(self, calls: Path, device: str) -> None:
        self.calls = calls
        self.device = device

    def __call__(self, views):
        waveform, _sample_rate = views[AudioView.WAVEFORM]
        _append(self.calls, f"scalar:{int(waveform.item())}:{self.device}")
        return {"semantic_codes": waveform.to(torch.int64) + 10}


class _BatchProvider(_Provider):
    def __call__(self, views):
        raise AssertionError("batched online access must use call_batch")

    def call_batch(self, batch):
        waveform, _sample_rates = batch.sample[_REF].views[AudioView.WAVEFORM]
        _append(self.calls, f"batch:{','.join(str(int(v)) for v in waveform[:, 0, 0])}")
        return [{"semantic_codes": sample.to(torch.int64) + 10} for sample in waveform]


class _PairedSampleProvider:
    output = AudioView.WAVEFORM

    def __init__(self, calls: Path) -> None:
        self.calls = calls

    def __call__(self, sample):
        return self.call_batch((sample,))[0]

    def call_batch(self, samples):
        outputs = []
        for sample in samples:
            waveform, sample_rate = sample[_REF].views[AudioView.WAVEFORM]
            index = int(waveform.item())
            _append(self.calls, str(index))
            outputs.append(
                {
                    _SOURCE_AUDIO_REF: AudioItem(
                        views={
                            AudioView.WAVEFORM: (waveform + 100, sample_rate),
                        }
                    ),
                    _TARGET_AUDIO_REF: AudioItem(
                        views={
                            AudioView.WAVEFORM: (waveform + 200, sample_rate),
                        }
                    ),
                }
            )
        return outputs


@dataclass(frozen=True)
class _PairedSampleProviderFactory:
    calls: Path

    def __call__(self, _device: str) -> _PairedSampleProvider:
        return _PairedSampleProvider(self.calls)


class _EvenPredicate:
    def __call__(self, sample) -> bool:
        waveform, _sample_rate = sample[_REF].views[AudioView.WAVEFORM]
        return int(waveform.item()) % 2 == 0


@dataclass(frozen=True)
class _EvenPredicateFactory:
    def __call__(self) -> _EvenPredicate:
        return _EvenPredicate()


def _materializer(output: Path, staging: Path, **options) -> ViewMaterializer:
    return ViewMaterializer(
        output,
        staging_dir=staging,
        split="train",
        input_id="source-v1",
        provider_id="codec-v1",
        output=AudioView.LONGCAT,
        schema=_SCHEMA,
        commit_samples=1,
        **options,
    )


def _append(path: Path, value: str) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(value + "\n")


def _codes(sample) -> torch.Tensor:
    return sample[_REF].views[AudioView.LONGCAT]["semantic_codes"]


def _fail(*_args, **_kwargs):
    raise AssertionError("ready canonical access must not call factories")


def test_online_access_prioritizes_reads_and_background_covers_full_universe(
    tmp_path: Path,
) -> None:
    output = tmp_path / "canonical"
    staging = tmp_path / "staging"
    calls = tmp_path / "provider.calls"
    source_factory = _SourceFactory(3)
    provider_factory = _ProviderFactory(calls)
    materializer = _materializer(output, staging)

    dataset = materializer.open(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
        device="cpu",
    )
    assert isinstance(dataset, MaterializingViewDataset)
    first = dataset[1]
    second = dataset[1]
    assert torch.equal(_codes(first), torch.tensor([[11]]))
    assert torch.equal(_codes(second), torch.tensor([[11]]))
    assert dict(first[_REF].meta) == {}
    assert dataset.cost_row(2) == ("cost", 2)
    assert tuple(
        dataset._shuffle(
            shuffle=True,
            seed=0,
            epoch=0,
            num_replicas=1,
            rank=0,
        )
    ) == ((2, 0, 1),)
    dataset.close()

    observed = calls.read_text(encoding="utf-8").splitlines()
    assert observed[0] == "scalar:1:cpu"
    assert observed.count("scalar:0:cpu") == 1
    assert observed.count("scalar:1:cpu") == 2
    assert observed.count("scalar:2:cpu") == 1
    status = materializer.status(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
    )
    assert (status.completed, status.expected, status.finalized) == (3, 3, False)

    result = materializer.write(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
        devices="cpu",
    )
    assert result == output
    assert not staging.exists()
    assert calls.read_text(encoding="utf-8").splitlines() == observed

    ready = materializer.open(
        dataset_factory=_fail,
        provider_factory=_fail,
        device="unused",
    )
    assert isinstance(ready, StoreDataset)
    assert torch.equal(_codes(ready[2]), torch.tensor([[12]]))
    assert dict(ready[2][_REF].meta) == {}
    ready.close()

    finalized = materializer.status(
        dataset_factory=_fail,
        provider_factory=_fail,
    )
    assert (finalized.completed, finalized.expected, finalized.finalized) == (
        3,
        3,
        True,
    )
    provenance = read_store_manifest(output).provenance
    assert provenance["input_id"] == "source-v1"
    assert provenance["provider_id"] == "codec-v1"
    assert provenance["output_id"].startswith("view-v1:")


def test_online_getitems_batches_global_indexes_and_deduplicates_persistence(
    tmp_path: Path,
) -> None:
    calls = tmp_path / "provider.calls"
    materializer = _materializer(
        tmp_path / "canonical",
        tmp_path / "staging",
    )
    source_factory = _SourceFactory(3)
    provider_factory = _ProviderFactory(calls, batch=True)

    with materializer.open(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
    ) as dataset:
        outputs = dataset.__getitems__((2, 0, 2))
        assert [int(_codes(sample).item()) for sample in outputs] == [12, 10, 12]
        assert all(dict(sample[_REF].meta) == {} for sample in outputs)

    assert calls.read_text(encoding="utf-8").splitlines() == [
        "batch:2,0",
        "batch:1",
    ]
    status = materializer.status(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
    )
    assert (status.completed, status.expected) == (3, 3)


def test_close_without_sample_access_still_covers_full_universe(
    tmp_path: Path,
) -> None:
    calls = tmp_path / "provider.calls"
    materializer = _materializer(tmp_path / "canonical", tmp_path / "staging")
    source_factory = _SourceFactory(3)
    provider_factory = _ProviderFactory(calls, batch=True)

    dataset = materializer.open(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
    )
    dataset.close()

    assert calls.read_text(encoding="utf-8").splitlines() == [
        "batch:0",
        "batch:1",
        "batch:2",
    ]
    status = materializer.status(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
    )
    assert (status.completed, status.expected, status.finalized) == (3, 3, False)


def test_selection_only_changes_returned_rows_not_materialization_coverage(
    tmp_path: Path,
) -> None:
    calls = tmp_path / "provider.calls"
    materializer = _materializer(tmp_path / "canonical", tmp_path / "staging")
    source_factory = _SelectedSourceFactory(4)
    provider_factory = _ProviderFactory(calls, batch=True)

    dataset = materializer.open(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
    )
    assert isinstance(dataset, SelectionView)
    assert len(dataset) == 2
    assert dataset.sample_id(0) == "external-0"
    assert dataset.sample_id(1) == "external-2"
    online_universe_id = dataset.universe_id()
    assert online_universe_id is not None
    assert [int(_codes(sample).item()) for sample in dataset.__getitems__((1, 0))] == [
        12,
        10,
    ]
    dataset.close()

    observed = calls.read_text(encoding="utf-8").splitlines()
    assert sorted(
        int(value)
        for line in observed
        for value in line.removeprefix("batch:").split(",")
    ) == [0, 1, 2, 3]
    status = materializer.status(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
    )
    assert (status.completed, status.expected) == (4, 4)

    result = materializer.write(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
        devices="cpu",
    )
    assert result == tmp_path / "canonical"
    assert calls.read_text(encoding="utf-8").splitlines() == observed

    ready = read_store_dataset(result)
    try:
        assert len(ready) == 4
        assert ready.universe_id() == online_universe_id
        assert [ready.sample_id(index) for index in range(4)] == [
            "external-0",
            "external-1",
            "external-2",
            "external-3",
        ]
    finally:
        ready.close()

    selected_ready = materializer.open(
        dataset_factory=_fail,
        selection_factory=source_factory,
        provider_factory=_fail,
        device="unused",
    )
    try:
        assert isinstance(selected_ready, SelectionView)
        assert len(selected_ready) == 2
        assert [selected_ready.sample_id(index) for index in range(2)] == [
            "external-0",
            "external-2",
        ]
    finally:
        selected_ready.close()


def test_ready_input_id_factory_validates_and_reuses_open_selection(
    tmp_path: Path,
) -> None:
    output = tmp_path / "canonical"
    staging = tmp_path / "staging"
    calls = tmp_path / "provider.calls"
    source_factory = _SourceFactory(4)
    _materializer(output, staging).write(
        dataset_factory=source_factory,
        provider_factory=_ProviderFactory(calls),
        devices="cpu",
    )

    source = _ClosableSource(4)
    universe = DatasetUniverse(source)
    selected = DecisionSet(
        universe,
        tuple(index % 2 == 0 for index in range(4)),
    ).select(True)
    source_view = SelectionView(universe, (selected,))
    selection_factory = mock.Mock(return_value=source_view)
    input_id_factory = mock.Mock(return_value="source-v1")
    materializer = ViewMaterializer(
        output,
        staging_dir=staging,
        split="train",
        input_id_factory=input_id_factory,
        provider_id="codec-v1",
        output=AudioView.LONGCAT,
        schema=_SCHEMA,
    )

    ready = materializer.open(
        dataset_factory=_fail,
        selection_factory=selection_factory,
        provider_factory=_fail,
        device="unused",
    )
    assert isinstance(ready, SelectionView)
    assert selection_factory.call_count == 1
    input_id_factory.assert_called_once_with(source)
    assert [ready.sample_id(index) for index in range(len(ready))] == [
        "source-0",
        "source-2",
    ]
    assert source.close_calls == 0

    ready.close()
    assert source.close_calls == 1


def test_ready_input_id_factory_rejects_mismatched_canonical_store(
    tmp_path: Path,
) -> None:
    output = tmp_path / "canonical"
    staging = tmp_path / "staging"
    calls = tmp_path / "provider.calls"
    _materializer(output, staging).write(
        dataset_factory=_SourceFactory(1),
        provider_factory=_ProviderFactory(calls),
        devices="cpu",
    )

    source = _ClosableSource(1)
    source_view = SelectionView(DatasetUniverse(source))
    selection_factory = mock.Mock(return_value=source_view)
    input_id_factory = mock.Mock(return_value="source-v2")
    materializer = ViewMaterializer(
        output,
        staging_dir=staging,
        split="train",
        input_id_factory=input_id_factory,
        provider_id="codec-v1",
        output=AudioView.LONGCAT,
        schema=_SCHEMA,
    )

    with pytest.raises(ValueError, match="input_id"):
        materializer.open(
            dataset_factory=_fail,
            selection_factory=selection_factory,
            provider_factory=_fail,
            device="unused",
        )

    assert selection_factory.call_count == 1
    input_id_factory.assert_called_once_with(source)
    assert source.close_calls == 1


def test_ready_rebase_cleanup_closes_ready_after_source_close_failure(
    tmp_path: Path,
) -> None:
    ready = _ClosableSource(1)
    source = _FailingCloseSource(2)
    source_universe = DatasetUniverse(source)
    selected = DecisionSet(source_universe, (True, True)).select(True)
    source_view = SelectionView(source_universe, (selected,))
    materializer = _materializer(tmp_path / "canonical", tmp_path / "staging")

    with (
        mock.patch.object(materializer, "_ready_dataset", return_value=ready),
        pytest.raises(ValueError, match="missing sample_id 'source-1'"),
    ):
        materializer.open(
            dataset_factory=_fail,
            selection_factory=mock.Mock(return_value=source_view),
            provider_factory=_fail,
            device="unused",
        )

    assert source.close_calls == 1
    assert ready.close_calls == 1


def test_ready_status_rejects_input_id_factory_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "canonical"
    staging = tmp_path / "staging"
    calls = tmp_path / "provider.calls"
    _materializer(output, staging).write(
        dataset_factory=_SourceFactory(1),
        provider_factory=_ProviderFactory(calls),
        devices="cpu",
    )

    source = _ClosableSource(1)
    dataset_factory = mock.Mock(return_value=source)
    input_id_factory = mock.Mock(return_value="source-v2")
    materializer = ViewMaterializer(
        output,
        staging_dir=staging,
        split="train",
        input_id_factory=input_id_factory,
        provider_id="codec-v1",
        output=AudioView.LONGCAT,
        schema=_SCHEMA,
    )

    with pytest.raises(ValueError, match="input_id"):
        materializer.status(
            dataset_factory=dataset_factory,
            provider_factory=_fail,
        )

    assert dataset_factory.call_count == 1
    input_id_factory.assert_called_once_with(source)
    assert source.close_calls == 1


def test_online_open_failure_closes_selected_source_resources(tmp_path: Path) -> None:
    source = _ClosableSource(1)
    resource = _ClosableResource()
    source_view = SelectionView(
        DatasetUniverse(source),
        resources=(resource,),
    )
    materializer = _materializer(tmp_path / "canonical", tmp_path / "staging")

    with pytest.raises(AssertionError, match="ready canonical"):
        materializer.open(
            dataset_factory=_fail,
            selection_factory=mock.Mock(return_value=source_view),
            provider_factory=_fail,
        )

    assert source.close_calls == 1
    assert resource.close_calls == 1

    reopened = materializer.open(
        dataset_factory=_SourceFactory(1),
        provider_factory=_ProviderFactory(tmp_path / "reopened.calls"),
    )
    reopened.close()


def test_online_selected_source_is_closed_once(tmp_path: Path) -> None:
    source = _ClosableSource(2)
    universe = DatasetUniverse(source)
    selected = DecisionSet(universe, (True, False)).select(True)
    source_view = SelectionView(universe, (selected,))
    materializer = _materializer(tmp_path / "canonical", tmp_path / "staging")

    dataset = materializer.open(
        dataset_factory=_fail,
        selection_factory=mock.Mock(return_value=source_view),
        provider_factory=_ProviderFactory(tmp_path / "provider.calls"),
    )
    assert isinstance(dataset, SelectionView)
    assert int(_codes(dataset[0]).item()) == 10
    dataset.close()
    dataset.close()

    assert source.close_calls == 1


def test_non_ready_status_closes_selected_source_resources(tmp_path: Path) -> None:
    source = _ClosableSource(1)
    resource = _ClosableResource()
    source_view = SelectionView(
        DatasetUniverse(source),
        resources=(resource,),
    )
    materializer = _materializer(tmp_path / "canonical", tmp_path / "staging")

    status = materializer.status(
        dataset_factory=mock.Mock(return_value=source_view),
        provider_factory=_fail,
    )

    assert (status.expected, status.completed, status.finalized) == (1, 0, False)
    assert source.close_calls == 1
    assert resource.close_calls == 1


def test_ready_rebase_closes_live_filter_owner_and_source_once(
    tmp_path: Path,
) -> None:
    output = tmp_path / "canonical"
    staging = tmp_path / "staging"
    _materializer(output, staging).write(
        dataset_factory=_SourceFactory(2),
        provider_factory=_ProviderFactory(tmp_path / "provider.calls"),
        devices="cpu",
    )
    source = _ClosableSource(2)
    run = FilterRule("materializer-live-owner", _EvenPredicateFactory()).open(
        dataset_factory=lambda: source,
        labels=True,
        input_id="materializer-live-source-v1",
        device="cpu",
        commit_samples=1,
        write_workers=0,
    )
    run.wait()
    materializer = _materializer(output, staging)

    try:
        with mock.patch.object(run, "close", wraps=run.close) as close:
            ready = materializer.open(
                dataset_factory=_fail,
                selection_factory=lambda: run.dataset,
                provider_factory=_fail,
                device="unused",
            )
            assert len(ready) == 1
            ready.close()
            ready.close()
            assert close.call_count == 1
    finally:
        run.close()

    assert source.close_calls == 1


def test_logical_dataset_id_makes_universe_identity_root_independent(
    tmp_path: Path,
) -> None:
    universe_ids: list[str] = []
    logical_dataset_id = "wmt19/moss_tts/longcat/train"

    for name in ("physical-a", "physical-b"):
        output = tmp_path / name
        materializer = ViewMaterializer(
            output,
            staging_dir=tmp_path / f"{name}-staging",
            split="train",
            input_id="source-v1",
            provider_id="codec-v1",
            output=AudioView.LONGCAT,
            schema=_SCHEMA,
            commit_samples=1,
            dataset_id=logical_dataset_id,
        )
        source_factory = _SourceFactory(1)
        provider_factory = _ProviderFactory(tmp_path / f"{name}.calls")

        online = materializer.open(
            dataset_factory=source_factory,
            provider_factory=provider_factory,
        )
        online_universe_id = online.universe_id()
        assert online_universe_id is not None
        universe_ids.append(online_universe_id)
        online.close()
        materializer.write(
            dataset_factory=source_factory,
            provider_factory=provider_factory,
            devices="cpu",
        )

        ready = materializer.open(
            dataset_factory=_fail,
            provider_factory=_fail,
            device="unused",
        )
        try:
            ready_universe_id = ready.universe_id()
            assert ready_universe_id == online_universe_id
            assert read_store_manifest(output).dataset_id == logical_dataset_id
        finally:
            ready.close()

    assert universe_ids[0] == universe_ids[1]


def test_sample_materializer_online_selection_covers_full_universe(
    tmp_path: Path,
) -> None:
    output = tmp_path / "canonical"
    staging = tmp_path / "staging"
    calls = tmp_path / "provider.calls"
    source_factory = _SourceFactory(4)
    selection_factory = _EvenSourceFactory(4)
    provider_factory = _PairedSampleProviderFactory(calls)
    materializer = SampleMaterializer(
        output,
        staging_dir=staging,
        split="train",
        input_id="source-v1",
        provider_id="paired-speech-v1",
        output=AudioView.WAVEFORM,
        schema=_PAIRED_SCHEMA,
        batch_size=2,
        commit_samples=1,
    )

    dataset = materializer.open(
        dataset_factory=source_factory,
        selection_factory=selection_factory,
        provider_factory=provider_factory,
    )
    assert isinstance(dataset, SelectionView)
    assert len(dataset) == 2
    assert [dataset.sample_id(index) for index in range(2)] == [
        "source-0",
        "source-2",
    ]
    selected = dataset[1]
    source_waveform, source_rate = selected[_SOURCE_AUDIO_REF].views[AudioView.WAVEFORM]
    target_waveform, target_rate = selected[_TARGET_AUDIO_REF].views[AudioView.WAVEFORM]
    assert torch.equal(source_waveform, torch.tensor([[102.0]]))
    assert torch.equal(target_waveform, torch.tensor([[202.0]]))
    assert (source_rate, target_rate) == (16_000, 16_000)
    dataset.close()

    assert sorted(int(value) for value in calls.read_text().splitlines()) == [
        0,
        1,
        2,
        3,
    ]
    status = materializer.status(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
    )
    assert (status.completed, status.expected, status.finalized) == (4, 4, False)

    materializer.write(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
        devices="cpu",
    )
    assert sorted(int(value) for value in calls.read_text().splitlines()) == [
        0,
        1,
        2,
        3,
    ]

    ready = materializer.open(
        dataset_factory=_fail,
        selection_factory=selection_factory,
        provider_factory=_fail,
        device="unused",
    )
    try:
        assert isinstance(ready, SelectionView)
        assert len(ready) == 2
        source_waveform, _ = ready[1][_SOURCE_AUDIO_REF].views[AudioView.WAVEFORM]
        target_waveform, _ = ready[1][_TARGET_AUDIO_REF].views[AudioView.WAVEFORM]
        assert torch.equal(source_waveform, torch.tensor([[102.0]]))
        assert torch.equal(target_waveform, torch.tensor([[202.0]]))
    finally:
        ready.close()


def test_online_owner_blocks_second_open_finalize_and_forked_access(
    tmp_path: Path,
) -> None:
    calls = tmp_path / "provider.calls"
    materializer = _materializer(tmp_path / "canonical", tmp_path / "staging")
    source_factory = _SourceFactory(1)
    provider_factory = _ProviderFactory(calls)
    dataset = materializer.open(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
    )

    with pytest.raises(RuntimeError, match="already held"):
        materializer.open(
            dataset_factory=source_factory,
            provider_factory=provider_factory,
        )
    with pytest.raises(RuntimeError, match="already held"):
        materializer.write(
            dataset_factory=source_factory,
            provider_factory=provider_factory,
            devices="cpu",
        )
    with mock.patch(
        "anydataset.store.materialize.materializer.os.getpid",
        return_value=dataset._owner_pid + 1,
    ):
        with pytest.raises(RuntimeError, match="forked process"):
            dataset[0]

    dataset.close()


def test_nonready_or_incompatible_canonical_store_never_falls_back_online(
    tmp_path: Path,
) -> None:
    output = tmp_path / "canonical"
    staging = tmp_path / "staging"
    output.mkdir()
    (output / "partial").write_text("incomplete", encoding="utf-8")
    materializer = _materializer(output, staging)

    with pytest.raises(ValueError, match="exists but is not ready"):
        materializer.open(dataset_factory=_fail, provider_factory=_fail)

    (output / "partial").unlink()
    output.rmdir()
    source_factory = _SourceFactory(1)
    provider_factory = _ProviderFactory(tmp_path / "provider.calls")
    materializer.write(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
        devices="cpu",
    )

    incompatible = ViewMaterializer(
        output,
        staging_dir=staging,
        split="train",
        input_id="source-v1",
        provider_id="codec-v2",
        output=AudioView.LONGCAT,
        schema=_SCHEMA,
    )
    with pytest.raises(ValueError, match="provider_id"):
        incompatible.open(dataset_factory=_fail, provider_factory=_fail)


def test_staging_status_rejects_changed_semantic_ids(tmp_path: Path) -> None:
    output = tmp_path / "canonical"
    staging = tmp_path / "staging"
    source_factory = _SourceFactory(2)
    provider_factory = _ProviderFactory(tmp_path / "provider.calls")
    materializer = _materializer(output, staging)

    materializer.open(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
    ).close()

    for input_id, provider_id in (
        ("source-v2", "codec-v1"),
        ("source-v1", "codec-v2"),
    ):
        changed = ViewMaterializer(
            output,
            staging_dir=staging,
            split="train",
            input_id=input_id,
            provider_id=provider_id,
            output=AudioView.LONGCAT,
            schema=_SCHEMA,
            commit_samples=1,
        )
        with pytest.raises(ValueError, match="identity does not match"):
            changed.status(
                dataset_factory=source_factory,
                provider_factory=provider_factory,
            )


def test_background_write_failure_is_raised_on_close_and_releases_owner_lock(
    tmp_path: Path,
) -> None:
    failure = RuntimeError("fragment write failed")
    calls = tmp_path / "provider.calls"
    materializer = _materializer(tmp_path / "canonical", tmp_path / "staging")
    source_factory = _SourceFactory(1)
    provider_factory = _ProviderFactory(calls)

    with mock.patch.object(
        materialize_fragments,
        "write_fragment",
        side_effect=failure,
    ):
        dataset = materializer.open(
            dataset_factory=source_factory,
            provider_factory=provider_factory,
        )
        assert torch.equal(_codes(dataset[0]), torch.tensor([[10]]))
        with pytest.raises(RuntimeError) as raised:
            dataset.close()

    assert raised.value is failure
    assert dataset.closed
    reopened = materializer.open(
        dataset_factory=source_factory,
        provider_factory=provider_factory,
    )
    reopened.close()
