from io import BytesIO
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from anydataset import (
    AudioItem,
    AudioKey,
    AudioOptKey,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextView,
    ViewRef,
)
from anydataset.store import DatasetWriter
from anydataset.store.jsonio import read_json
from anydataset.store.manifest import DatasetManifest
from anydataset.store.manifestio import (
    read_samples_manifest,
    read_view_manifest,
)
from anydataset.store.paths import (
    dataset_json_path,
    dataset_ready_path,
    samples_jsonl_path,
    samples_parquet_path,
    view_manifest_path,
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
                duration=0.75,
                label="speech",
                text="hello",
            )

            written = DatasetWriter(
                output,
                dataset_id="toy-audio",
                split="train",
                config={"task": "audio_codec"},
                provenance={"input": "memory"},
            ).write([sample])

            ref = ViewRef(Modality.AUDIO, AudioView.WAVEFORM)
            dataset = DatasetManifest.from_dict(read_json(dataset_json_path(output)))
            sample_entry = next(read_samples_manifest(output))
            view_entry = next(read_view_manifest(output, ref, "raw"))

            self.assertEqual(written, output)
            self.assertTrue(dataset_ready_path(output).exists())
            self.assertTrue(view_ready_path(output, ref, "raw").exists())
            self.assertTrue(samples_parquet_path(output).is_file())
            self.assertTrue(view_manifest_parquet_path(output, ref, "raw").is_file())
            self.assertFalse(samples_jsonl_path(output).exists())
            self.assertFalse(view_manifest_path(output, ref, "raw").exists())
            self.assertEqual(dataset.dataset_id, "toy-audio")
            self.assertEqual(dataset.sample_count, 1)
            self.assertEqual(dataset.views[0].ref, ref)
            self.assertEqual(sample_entry.sample_id, view_entry.sample_id)
            audio_entry = sample_entry.item((Role.DEFAULT, Modality.AUDIO))
            self.assertIsNotNone(audio_entry)
            self.assertEqual(audio_entry.required[AudioKey.SAMPLE_RATE], 4)
            self.assertEqual(audio_entry.optional[AudioOptKey.DURATION], 0.75)
            self.assertEqual(audio_entry.optional[AudioOptKey.LABEL], "speech")
            text_entry = sample_entry.item((Role.DEFAULT, Modality.TEXT))
            self.assertIsNotNone(text_entry)
            self.assertEqual(view_entry.shape, (1, 3))
            self.assertEqual(view_entry.dtype, "torch.float32")
            self.assertTrue(view_entry.checksum.startswith("sha256:"))

            with tarfile.open(view_shard_path(output, ref, "raw", view_entry.shard), "r") as tar:
                payload = tar.extractfile(view_entry.key).read()
            loaded = torch.load(BytesIO(payload))
            self.assertTrue(torch.equal(loaded, waveform))

    def test_writer_copies_file_view_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = Path(tmpdir) / "dataset"
            sample = audio_sample(file=str(source), sample_rate=16000)

            DatasetWriter(output, dataset_id="file-audio").write([sample])

            ref = ViewRef(Modality.AUDIO, AudioView.FILE)
            view_entry = next(read_view_manifest(output, ref, "raw"))

            self.assertEqual(view_entry.shape, (len(b"RIFF-data"),))
            self.assertEqual(view_entry.dtype, "bytes")
            self.assertEqual(Path(view_entry.key).suffix, ".wav")
            with tarfile.open(view_shard_path(output, ref, "raw", view_entry.shard), "r") as tar:
                self.assertEqual(tar.extractfile(view_entry.key).read(), b"RIFF-data")

    def test_writer_can_write_jsonl_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            waveform = torch.tensor([[1.0, 2.0]])
            sample = audio_sample(waveform=waveform, sample_rate=4)

            DatasetWriter(
                output,
                dataset_id="toy-audio",
                manifest_format="jsonl",
            ).write([sample])

            ref = ViewRef(Modality.AUDIO, AudioView.WAVEFORM)
            sample_entry = next(read_samples_manifest(output))
            view_entry = next(read_view_manifest(output, ref, "raw"))

            self.assertTrue(samples_jsonl_path(output).is_file())
            self.assertTrue(view_manifest_path(output, ref, "raw").is_file())
            self.assertFalse(samples_parquet_path(output).exists())
            self.assertFalse(view_manifest_parquet_path(output, ref, "raw").exists())
            self.assertEqual(sample_entry.dataset_name, "toy-audio")
            self.assertEqual(view_entry.shape, (1, 2))

    def test_writer_rolls_shards_by_sample_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            samples = [
                audio_sample(
                    waveform=torch.tensor([[float(index)]]),
                    sample_rate=4,
                )
                for index in range(3)
            ]

            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=1,
            ).write(samples)

            ref = ViewRef(Modality.AUDIO, AudioView.WAVEFORM)
            entries = list(read_view_manifest(output, ref, "raw"))

            self.assertEqual([entry.shard for entry in entries], [
                "000000.tar",
                "000001.tar",
                "000002.tar",
            ])
            for index, entry in enumerate(entries):
                with tarfile.open(
                    view_shard_path(output, ref, "raw", entry.shard),
                    "r",
                ) as tar:
                    payload = tar.extractfile(entry.key).read()
                loaded = torch.load(BytesIO(payload))
                self.assertTrue(torch.equal(loaded, torch.tensor([[float(index)]])))

    def test_writer_rolls_shards_by_payload_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            samples = [
                audio_sample(
                    waveform=torch.full((1, 2), float(index)),
                    sample_rate=16000,
                )
                for index in range(2)
            ]

            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_bytes=1,
            ).write(samples)

            ref = ViewRef(Modality.AUDIO, AudioView.WAVEFORM)
            entries = list(read_view_manifest(output, ref, "raw"))

            self.assertEqual([entry.shard for entry in entries], ["000000.tar", "000001.tar"])

    def test_writer_rejects_invalid_shard_limits(self):
        with self.assertRaises(ValueError):
            DatasetWriter("/tmp/unused", dataset_id="toy", max_shard_samples=0)

        with self.assertRaises(TypeError):
            DatasetWriter("/tmp/unused", dataset_id="toy", max_shard_bytes=True)

        with self.assertRaises(ValueError):
            DatasetWriter("/tmp/unused", dataset_id="toy", manifest_format="bad")

    def test_writer_rejects_non_empty_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            output.mkdir()
            (output / "existing.txt").write_text("keep me", encoding="utf-8")

            writer = DatasetWriter(output, dataset_id="toy")

            with self.assertRaises(ValueError):
                writer.write([])

            self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "keep me")

    def test_explicit_view_must_exist_on_each_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            sample = audio_sample(
                file=str(Path(tmpdir) / "missing.wav"),
                sample_rate=16000,
            )
            writer = DatasetWriter(
                output,
                dataset_id="toy",
                views=(ViewRef(Modality.AUDIO, AudioView.WAVEFORM),),
            )

            with self.assertRaises(KeyError):
                writer.write([sample])

            self.assertFalse(output.exists())

    def test_writer_rejects_sample_without_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            sample = audio_sample(sample_rate=16000)

            with self.assertRaises(ValueError):
                DatasetWriter(output, dataset_id="toy").write([sample])

            self.assertFalse(output.exists())

    def test_writer_rejects_audio_sample_without_sample_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            sample = {
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: torch.zeros(1, 4)}
                )
            }

            with self.assertRaisesRegex(ValueError, "sample_rate"):
                DatasetWriter(output, dataset_id="toy").write([sample])

            self.assertFalse(output.exists())

    def test_writer_rejects_missing_sample_rate_with_explicit_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            sample = {
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: torch.zeros(1, 4)}
                )
            }

            with self.assertRaisesRegex(ValueError, "sample_rate"):
                DatasetWriter(
                    output,
                    dataset_id="toy",
                    views=(ViewRef(Modality.AUDIO, AudioView.WAVEFORM),),
                ).write([sample])

            self.assertFalse(output.exists())


def audio_sample(
    *,
    waveform=None,
    file=None,
    longcat=None,
    sample_rate: int,
    duration=None,
    label=None,
    text: str | None = None,
):
    views = {}
    if waveform is not None:
        views[AudioView.WAVEFORM] = waveform
    if file is not None:
        views[AudioView.FILE] = file
    if longcat is not None:
        views[AudioView.LONGCAT] = longcat
    optional = {}
    if duration is not None:
        optional[AudioOptKey.DURATION] = duration
    if label is not None:
        optional[AudioOptKey.LABEL] = label
    sample = {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views=views,
            required={AudioKey.SAMPLE_RATE: sample_rate},
            optional=optional,
        )
    }
    if text is not None:
        sample[(Role.DEFAULT, Modality.TEXT)] = TextItem(
            views={TextView.TEXT: text}
        )
    return sample


if __name__ == "__main__":
    unittest.main()
