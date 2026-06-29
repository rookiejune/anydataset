from io import BytesIO
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from anydataset import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextView,
)
from anydataset.store import DatasetWriter
from anydataset.store.parts import (
    DatasetFragmentWriter,
    commit_store_fragments,
    completed_fragment_indexes,
)
from anydataset.store.writer import DEFAULT_MAX_SHARD_SAMPLES
from anydataset.store.jsonio import read_json
from anydataset.store.manifest import DatasetManifest
from anydataset.store.manifestio import read_samples_manifest, read_view_manifest
from anydataset.store.paths import (
    dataset_json_path,
    dataset_ready_path,
    samples_parquet_path,
    view_dir,
    view_manifest_parquet_path,
    view_ready_path,
    view_shard_path,
)


class DatasetWriterTest(unittest.TestCase):
    def test_writer_writes_waveform_view_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            sample = audio_sample(
                waveform=waveform,
                sample_rate=4,
                label="speech",
                text="hello",
            )
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)

            written = DatasetWriter(
                output,
                dataset_id="toy-audio",
                split="train",
            ).write([sample])

            dataset_json = read_json(dataset_json_path(output))
            dataset = DatasetManifest(**dataset_json)
            sample_entry = next(read_samples_manifest(output))
            view_entry = next(read_view_manifest(output, view))

            self.assertEqual(written, output)
            self.assertEqual(
                set(dataset_json),
                {"dataset_id", "sample_count", "split"},
            )
            self.assertTrue(dataset_ready_path(output).exists())
            self.assertTrue(view_ready_path(output, view).exists())
            self.assertTrue(samples_parquet_path(output).is_file())
            self.assertTrue(view_manifest_parquet_path(output, view).is_file())
            self.assertFalse((view_dir(output, view) / "view.json").exists())
            self.assertEqual(sample_entry.sample_id, view_entry.sample_id)
            audio_entry = sample_entry.item((Role.DEFAULT, Modality.AUDIO))
            self.assertEqual(audio_entry[1]["label"], "speech")
            self.assertEqual(
                set(view_entry.__dict__),
                {"role", "modality", "view", "sample_id", "shard", "key"},
            )

            with tarfile.open(view_shard_path(output, view, view_entry.shard), "r") as tar:
                payload = tar.extractfile(view_entry.key).read()
            loaded_waveform, loaded_sample_rate = torch.load(BytesIO(payload))
            self.assertTrue(torch.equal(loaded_waveform, waveform))
            self.assertEqual(loaded_sample_rate, 4)

    def test_writer_copies_file_view_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = Path(tmpdir) / "dataset"
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.FILE)

            DatasetWriter(output, dataset_id="file-audio").write(
                [audio_sample(file=str(source), sample_rate=16000)]
            )
            view_entry = next(read_view_manifest(output, view))

            self.assertEqual(Path(view_entry.key).suffix, ".wav")
            with tarfile.open(view_shard_path(output, view, view_entry.shard), "r") as tar:
                self.assertEqual(tar.extractfile(view_entry.key).read(), b"RIFF-data")

    def test_writer_rolls_shards_by_sample_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            samples = [
                audio_sample(waveform=torch.tensor([[float(index)]]), sample_rate=4)
                for index in range(3)
            ]

            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=1,
            ).write(samples)

            entries = list(read_view_manifest(output, view))

            self.assertEqual(
                [entry.shard for entry in entries],
                ["000000.tar", "000001.tar", "000002.tar"],
            )

    def test_writer_defaults_to_100k_samples_per_shard(self):
        writer = DatasetWriter("unused", dataset_id="toy")

        self.assertEqual(writer.max_shard_samples, 100_000)
        self.assertEqual(DEFAULT_MAX_SHARD_SAMPLES, 100_000)

    def test_explicit_view_must_exist_on_each_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            missing = Path(tmpdir) / "missing.wav"
            writer = DatasetWriter(
                output,
                dataset_id="toy",
                views=((Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),),
            )

            with self.assertRaises(KeyError):
                writer.write([audio_sample(file=str(missing), sample_rate=16000)])

            self.assertFalse(output.exists())

    def test_writer_rejects_sample_without_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"

            with self.assertRaises(ValueError):
                DatasetWriter(output, dataset_id="toy").write(
                    [audio_sample(sample_rate=16000)]
                )

            self.assertFalse(output.exists())

    def test_writer_rejects_inconsistent_item_view_sets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            source = Path(tmpdir) / "source.wav"
            source.write_bytes(b"RIFF-data")

            with self.assertRaises(ValueError):
                DatasetWriter(output, dataset_id="toy-audio").write(
                    [
                        audio_sample(
                            waveform=torch.tensor([[1.0]]),
                            file=str(source),
                            sample_rate=4,
                        ),
                        audio_sample(
                            waveform=torch.tensor([[2.0]]),
                            sample_rate=4,
                        ),
                    ]
                )

            self.assertFalse(output.exists())

    def test_fragment_writer_commits_batch_fragments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fragments = root / "fragments"
            output = root / "dataset"
            samples = [
                audio_sample(
                    waveform=torch.tensor([[float(index)]]),
                    sample_rate=4,
                )
                for index in range(3)
            ]

            DatasetFragmentWriter(
                fragments / "batch-000000000000-000000000001-a",
                dataset_id="toy-audio",
                split="train",
                fragment_id="batch-000000000000-000000000001-a",
            ).write([(0, samples[0]), (1, samples[1])])
            DatasetFragmentWriter(
                fragments / "batch-000000000002-000000000002-b",
                dataset_id="toy-audio",
                split="train",
                fragment_id="batch-000000000002-000000000002-b",
            ).write([(2, samples[2])])

            self.assertEqual(
                completed_fragment_indexes(
                    fragments,
                    dataset_id="toy-audio",
                    split="train",
                ),
                frozenset({0, 1, 2}),
            )

            commit_store_fragments(
                output,
                fragments,
                dataset_id="toy-audio",
                split="train",
                expected_sample_count=3,
            )

            indexes = [entry.sample_index for entry in read_samples_manifest(output)]
            self.assertEqual(indexes, [0, 1, 2])
            self.assertTrue(dataset_ready_path(output).exists())


def audio_sample(
    *,
    waveform=None,
    file=None,
    longcat=None,
    sample_rate: int,
    label=None,
    text: str | None = None,
):
    views = {}
    if waveform is not None:
        views[AudioView.WAVEFORM] = (waveform, sample_rate)
    if file is not None:
        views[AudioView.FILE] = file
    if longcat is not None:
        views[AudioView.LONGCAT] = longcat
    meta = {}
    if label is not None:
        meta[AudioMeta.LABEL] = label
    sample = {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views=views,
            meta=meta,
        )
    }
    if text is not None:
        sample[(Role.DEFAULT, Modality.TEXT)] = TextItem(
            views={TextView.TEXT: text}
        )
    return sample


if __name__ == "__main__":
    unittest.main()
