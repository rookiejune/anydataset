import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioView,
    Modality,
    Role,
    Source,
    Spec,
    TextItem,
    TextMeta,
    TextView,
)
from anydataset.store import DatasetWriter, ViewMaterializer
from anydataset.store.jsonio import read_json
from anydataset.store.materializer import iter_indexed_shard
from anydataset.store.manifestio import read_samples_manifest, read_view_manifest
from anydataset.store.paths import view_dir
from anydataset.store.reader import read_store_dataset


class ViewMaterializerTest(unittest.TestCase):
    def test_materializer_uses_output_dir_name_as_dataset_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "longcat-delta"
            sample = _audio_sample(torch.tensor([[1.0, 2.0]]))

            ViewMaterializer(target, split="train").write(
                dataset_factory=_DatasetFactory((sample,)),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )

            self.assertEqual(
                read_json(target / "dataset.json")["dataset_id"],
                "longcat-delta",
            )

    def test_materializer_writes_only_provider_output_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            dataset = _source_dataset(source, root, [_audio_sample(waveform)])

            ViewMaterializer(target, split="train").write(
                dataset_factory=_DatasetFactory(dataset),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            sample = _read_sample(target, root)

            self.assertEqual(
                set(stored.views),
                {(Role.DEFAULT, Modality.AUDIO, AudioView.LONGCAT)},
            )
            self.assertFalse(
                view_dir(
                    target,
                    (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),
                ).exists()
            )
            self.assertEqual(set(sample[Role.DEFAULT, Modality.AUDIO].views), {AudioView.LONGCAT})
            self.assertTrue(
                torch.equal(
                    sample[Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[11, 12, 13]]),
                )
            )

    def test_materializer_processes_multiple_roles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            dataset = _source_dataset(
                source,
                root,
                [
                    {
                        (Role.SOURCE, Modality.AUDIO): AudioItem(
                            views={AudioView.WAVEFORM: (torch.tensor([[1.0, 2.0]]), 4)}
                        ),
                        (Role.TARGET, Modality.AUDIO): AudioItem(
                            views={AudioView.WAVEFORM: (torch.tensor([[3.0, 4.0]]), 4)}
                        ),
                    }
                ],
            )

            ViewMaterializer(target).write(
                dataset_factory=_DatasetFactory(dataset),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            sample = _read_sample(target, root)

            self.assertEqual(
                set(stored.views),
                {
                    (Role.SOURCE, Modality.AUDIO, AudioView.LONGCAT),
                    (Role.TARGET, Modality.AUDIO, AudioView.LONGCAT),
                },
            )
            self.assertTrue(
                torch.equal(
                    sample[Role.SOURCE, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[11, 12]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    sample[Role.TARGET, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[13, 14]]),
                )
            )

    def test_materializer_skips_items_from_other_modalities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            dataset = _source_dataset(
                source,
                root,
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={AudioView.WAVEFORM: (torch.tensor([[1.0, 2.0]]), 4)}
                        ),
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello"},
                            meta={TextMeta.LANG: "en_us"},
                        ),
                    }
                ],
            )

            ViewMaterializer(target).write(
                dataset_factory=_DatasetFactory(dataset),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            sample = _read_sample(target, root)

            self.assertEqual(
                set(stored.views),
                {(Role.DEFAULT, Modality.AUDIO, AudioView.LONGCAT)},
            )
            self.assertEqual(set(sample), {(Role.DEFAULT, Modality.AUDIO)})
            self.assertTrue(
                torch.equal(
                    sample[Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[11, 12]]),
                )
            )

    def test_materialized_delta_merges_into_base_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            waveform = torch.tensor([[1.0, 2.0]])
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={AudioView.WAVEFORM: (waveform, 4)}
                        ),
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello"},
                            meta={TextMeta.LANG: "en_us"},
                        ),
                    }
                ]
            )

            ViewMaterializer(target, split="train").write(
                dataset_factory=_StoreDatasetFactory(source, root),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )
            merged = _store_dataset(source, root).merge(_store_dataset(target, root))
            sample = merged[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        text = sample[Role.DEFAULT, Modality.TEXT]
        self.assertEqual(set(audio.views), {AudioView.WAVEFORM, AudioView.LONGCAT})
        self.assertTrue(torch.equal(audio.views[AudioView.WAVEFORM][0], waveform))
        self.assertTrue(
            torch.equal(
                audio.views[AudioView.LONGCAT]["semantic_codes"],
                torch.tensor([[11, 12]]),
            )
        )
        self.assertEqual(text.views[TextView.TEXT], "hello")
        self.assertEqual(text.meta[TextMeta.LANG], "en_us")

    def test_iter_indexed_shard_uses_map_style_indexes(self):
        dataset = [_audio_sample(torch.tensor([[float(index)]])) for index in range(5)]

        self.assertEqual(
            [index for index, _ in iter_indexed_shard(dataset, 2, 1)],
            [1, 3],
        )

    def test_iter_indexed_shard_falls_back_to_iterable_modulo(self):
        dataset = (
            _audio_sample(torch.tensor([[float(index)]]))
            for index in range(5)
        )

        self.assertEqual(
            [index for index, _ in iter_indexed_shard(dataset, 2, 0)],
            [0, 2, 4],
        )

    def test_materializer_parts_commit_readable_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            parts = root / "parts"
            samples = [
                _audio_sample(torch.tensor([[float(index)]]))
                for index in range(4)
            ]
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(samples)
            dataset = _store_dataset(source, root)
            materializer = ViewMaterializer(
                target,
                split="train",
            )

            materializer.write_part(
                dataset,
                _Provider(offset=10),
                parts_dir=parts,
                num_shards=2,
                shard_id=0,
            )
            dataset = _store_dataset(source, root)
            materializer.write_part(
                dataset,
                _Provider(offset=10),
                parts_dir=parts,
                num_shards=2,
                shard_id=1,
            )
            materializer.commit_parts(parts)

            stored = read_store_dataset(target)
            indexes = [entry.sample_index for entry in read_samples_manifest(target)]
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.LONGCAT)
            shards = {entry.shard for entry in read_view_manifest(target, view)}

            self.assertEqual(len(stored), 4)
            self.assertEqual(indexes, [0, 1, 2, 3])
            self.assertEqual(shards, {"part-00000-000000.tar", "part-00001-000000.tar"})
            for index in range(4):
                sample = stored[index]
                self.assertTrue(
                    torch.equal(
                        sample[Role.DEFAULT, Modality.AUDIO]
                        .views[AudioView.LONGCAT]["semantic_codes"],
                        torch.tensor([[index + 10]]),
                    )
                )

    def test_materializer_write_parallel_uses_devices_and_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "target"
            samples = tuple(
                _audio_sample(torch.tensor([[float(index)]]))
                for index in range(4)
            )

            ViewMaterializer(target, split="train").write(
                dataset_factory=_DatasetFactory(samples),
                provider_factory=_ParallelProviderFactory(),
                devices=("cpu:0", "cpu:1"),
            )

            stored = read_store_dataset(target)
            logs = sorted((target / "logs").glob("part-*.log"))
            self.assertEqual(len(stored), 4)
            self.assertEqual([path.name for path in logs], ["part-00000.log", "part-00001.log"])
            self.assertIn("cpu:0", logs[0].read_text(encoding="utf-8"))
            self.assertIn("cpu:1", logs[1].read_text(encoding="utf-8"))
            self.assertTrue(
                torch.equal(
                    stored[0][Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[0]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    stored[1][Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[101]]),
                )
            )

    def test_resolve_devices_auto_falls_back_to_cpu(self):
        from anydataset.store.materializer import resolve_devices

        self.assertTrue(resolve_devices("auto"))

    def test_parallel_materializer_rejects_unpicklable_factory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with self.assertRaises(TypeError):
                ViewMaterializer(root / "target").write_parallel(
                    dataset_factory=lambda: (),
                    provider_factory=_ParallelProviderFactory(),
                    devices=("cpu:0", "cpu:1"),
                )


def _source_dataset(path: Path, root: Path, samples):
    DatasetWriter(path, dataset_id="toy-audio", split="train").write(samples)
    return _store_dataset(path, root)


def _store_dataset(path: Path, root: Path):
    return AnyDataset(
        Spec(source=Source.STORE, path=str(path), split="train"),
        cache_root=root / "cache-source",
    )


def _read_sample(path: Path, root: Path):
    dataset = AnyDataset(
        Spec(source=Source.STORE, path=str(path), split="train"),
        cache_root=root / "cache-target",
    )
    return dataset[0]


def _audio_sample(waveform: torch.Tensor):
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (waveform, 4)},
        )
    }


class _Provider:
    output = AudioView.LONGCAT

    def __init__(self, *, offset=0):
        self.offset = offset

    def __call__(self, views):
        waveform, _ = views[AudioView.WAVEFORM]
        return {"semantic_codes": waveform.to(torch.int64) + int(self.offset)}


@dataclass(frozen=True)
class _DatasetFactory:
    dataset: object

    def __call__(self):
        return self.dataset


@dataclass(frozen=True)
class _StoreDatasetFactory:
    path: Path
    root: Path

    def __call__(self):
        return _store_dataset(self.path, self.root)


@dataclass(frozen=True)
class _ProviderFactory:
    offset: int = 0

    def __call__(self, device: str):
        return _Provider(offset=self.offset)


@dataclass(frozen=True)
class _ParallelProviderFactory:
    def __call__(self, device: str):
        return _Provider(offset=100 if device.endswith(":1") else 0)


if __name__ == "__main__":
    unittest.main()
