from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from anydataset.dataset.abc import MapStyleABC
from anydataset.dataset.universe import DatasetUniverse
from anydataset.dataset.view import SelectionView
from anydataset.store import DatasetWriter, ViewMaterializer
from anydataset.store.manifest.io import read_samples_manifest
from anydataset.store.reader import read_store_dataset
from anydataset.types import AudioItem, AudioView, Modality, Role


_REF = (Role.DEFAULT, Modality.AUDIO)


def _sample(index: int):
    return {
        _REF: AudioItem(
            views={
                AudioView.WAVEFORM: (
                    torch.tensor([[float(index)]], dtype=torch.float32),
                    16_000,
                )
            }
        )
    }


class _StableDataset(MapStyleABC):
    def __init__(self, sample_ids: tuple[str, ...]) -> None:
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        return _sample(index)

    def sample_id(self, index: int) -> str:
        return self.sample_ids[index]


class _PlainDataset(MapStyleABC):
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        return _sample(index)


@dataclass(frozen=True)
class _Identity:
    sample_ids: tuple[str, ...]

    def sample_id(self, index: int) -> str:
        return self.sample_ids[index]


@dataclass(frozen=True)
class _StableFactory:
    sample_ids: tuple[str, ...]

    def __call__(self) -> _StableDataset:
        return _StableDataset(self.sample_ids)


@dataclass(frozen=True)
class _StoreFactory:
    path: Path

    def __call__(self):
        return read_store_dataset(self.path)


@dataclass(frozen=True)
class _SelectionFactory:
    sample_ids: tuple[str, ...]

    def __call__(self) -> SelectionView:
        universe = DatasetUniverse(
            _PlainDataset(len(self.sample_ids)),
            sample_identity=_Identity(self.sample_ids),
        )
        return SelectionView(universe)


class _Provider:
    output = AudioView.LONGCAT

    def __call__(self, views):
        waveform, _sample_rate = views[AudioView.WAVEFORM]
        return {"semantic_codes": waveform.to(torch.int64)}


@dataclass(frozen=True)
class _ProviderFactory:
    def __call__(self, _device: str) -> _Provider:
        return _Provider()


def test_store_writer_preserves_stable_ids_and_dense_indexes(tmp_path: Path) -> None:
    output = tmp_path / "store"
    DatasetWriter(output, dataset_id="derived").write(
        _StableDataset(("source-a", "source-b", "source-c"))
    )

    entries = tuple(read_samples_manifest(output))
    assert [entry.sample_index for entry in entries] == [0, 1, 2]
    assert [entry.sample_id for entry in entries] == [
        "source-a",
        "source-b",
        "source-c",
    ]

    dataset = read_store_dataset(output)
    try:
        assert dataset.sample_id(0) == "source-a"
        assert dataset.sample_id(-1) == "source-c"
    finally:
        dataset.close()


def test_store_writer_rejects_duplicate_inherited_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate sample_id 'duplicate'"):
        DatasetWriter(tmp_path / "store", dataset_id="derived").write(
            _StableDataset(("duplicate", "duplicate"))
        )


def test_parallel_store_writer_preserves_stable_ids(tmp_path: Path) -> None:
    output = tmp_path / "store"
    sample_ids = ("source-a", "source-b", "source-c", "source-d")
    DatasetWriter(
        output,
        dataset_id="derived",
        num_shards=2,
    ).write(dataset_factory=_StableFactory(sample_ids))

    entries = tuple(read_samples_manifest(output))
    assert [entry.sample_index for entry in entries] == [0, 1, 2, 3]
    assert [entry.sample_id for entry in entries] == list(sample_ids)


def test_parallel_store_writer_rejects_cross_part_duplicate_ids(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Duplicate sample_id 'duplicate'"):
        DatasetWriter(
            tmp_path / "store",
            dataset_id="derived",
            num_shards=2,
        ).write(
            dataset_factory=_StableFactory(
                ("duplicate", "duplicate", "source-c", "source-d")
            )
        )


def test_materializer_preserves_source_sample_ids(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    DatasetWriter(source, dataset_id="source-store").write(
        _StableDataset(("source-a", "source-b", "source-c"))
    )

    ViewMaterializer(target, split="train", commit_samples=1).write(
        dataset_factory=_StoreFactory(source),
        provider_factory=_ProviderFactory(),
        devices=("cpu:0", "cpu:1"),
    )

    entries = tuple(read_samples_manifest(target))
    assert [entry.sample_index for entry in entries] == [0, 1, 2]
    assert [entry.sample_id for entry in entries] == [
        "source-a",
        "source-b",
        "source-c",
    ]


def test_materializer_preserves_universe_identity_outside_payload_dataset(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    sample_ids = ("source-a", "source-b", "source-c")

    ViewMaterializer(
        target,
        split="train",
        input_id="source-v1",
        commit_samples=1,
    ).write(
        dataset_factory=_SelectionFactory(sample_ids),
        provider_factory=_ProviderFactory(),
        devices=("cpu",),
    )

    entries = tuple(read_samples_manifest(target))
    assert [entry.sample_index for entry in entries] == [0, 1, 2]
    assert [entry.sample_id for entry in entries] == list(sample_ids)
