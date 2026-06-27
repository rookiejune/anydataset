import tempfile
import unittest
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
from anydataset.store import DatasetWriter, ViewMaterializer, read_store_dataset
from anydataset.store.materializer import iter_indexed_shard
from anydataset.store.manifestio import read_samples_manifest, read_view_manifest
from anydataset.store.paths import view_dir


class ViewMaterializerTest(unittest.TestCase):
    def test_materializer_writes_only_provider_output_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            dataset = _source_dataset(source, root, [_audio_sample(waveform)])

            ViewMaterializer(target, dataset_id="toy-audio", split="train").write(
                dataset,
                _Provider(offset=10),
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

            ViewMaterializer(target, dataset_id="toy-audio").write(
                dataset,
                _Provider(offset=10),
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

            ViewMaterializer(target, dataset_id="toy-audio").write(
                dataset,
                _Provider(offset=10),
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

            ViewMaterializer(target, dataset_id="toy-audio", split="train").write(
                _store_dataset(source, root),
                _Provider(offset=10),
            )
            merged = read_store_dataset(source).merge(read_store_dataset(target))
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
                dataset_id="toy-audio",
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


if __name__ == "__main__":
    unittest.main()
