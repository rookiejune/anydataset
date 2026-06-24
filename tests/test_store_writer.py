from io import BytesIO
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch

from anydataset import AudioKey, AudioOptKey, AudioView, ModalityKey, TextKey, ViewRef
from anydataset.samples import Sample
from anydataset.store import (
    DatasetManifest,
    DatasetWriter,
    SampleManifestEntry,
    ViewManifestEntry,
    dataset_json_path,
    dataset_ready_path,
    read_json,
    read_jsonl,
    samples_jsonl_path,
    view_manifest_path,
    view_ready_path,
    view_shard_path,
)


class DatasetWriterTest(unittest.TestCase):
    def test_writer_writes_waveform_view_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            sample = Sample(
                data={
                    ModalityKey.AUDIO: {
                        AudioKey.SAMPLE_RATE: 4,
                        AudioOptKey.DURATION: 0.75,
                        AudioOptKey.LABEL: "speech",
                        AudioKey.VIEWS: {
                            AudioView.WAVEFORM: waveform,
                        },
                    },
                    ModalityKey.TEXT: {
                        TextKey.CONTENT: "hello",
                    },
                },
                dataset_name="toy:train",
                sample_index=0,
            )

            written = DatasetWriter(
                output,
                dataset_id="toy-audio",
                split="train",
                config={"task": "audio_codec"},
                provenance={"input": "memory"},
            ).write([sample])

            ref = ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM)
            dataset = DatasetManifest.from_dict(read_json(dataset_json_path(output)))
            sample_entry = SampleManifestEntry.from_dict(next(read_jsonl(samples_jsonl_path(output))))
            view_entry = ViewManifestEntry.from_dict(
                next(read_jsonl(view_manifest_path(output, ref, "raw")))
            )

            self.assertEqual(written, output)
            self.assertTrue(dataset_ready_path(output).exists())
            self.assertTrue(view_ready_path(output, ref, "raw").exists())
            self.assertEqual(dataset.dataset_id, "toy-audio")
            self.assertEqual(dataset.sample_count, 1)
            self.assertEqual(dataset.views[0].ref, ref)
            self.assertEqual(sample_entry.sample_id, view_entry.sample_id)
            self.assertEqual(sample_entry.sample_rate, 4)
            self.assertEqual(sample_entry.duration, 0.75)
            self.assertEqual(sample_entry.label, "speech")
            self.assertEqual(sample_entry.text, "hello")
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
            sample = Sample(
                data={
                    ModalityKey.AUDIO: {
                        AudioKey.SAMPLE_RATE: 16000,
                        AudioKey.VIEWS: {
                            AudioView.FILE: str(source),
                        },
                    },
                },
                dataset_name="files:train",
                sample_index=5,
            )

            DatasetWriter(output, dataset_id="file-audio").write([sample])

            ref = ViewRef(ModalityKey.AUDIO, AudioView.FILE)
            view_entry = ViewManifestEntry.from_dict(
                next(read_jsonl(view_manifest_path(output, ref, "raw")))
            )

            self.assertEqual(view_entry.shape, (len(b"RIFF-data"),))
            self.assertEqual(view_entry.dtype, "bytes")
            self.assertEqual(Path(view_entry.key).suffix, ".wav")
            with tarfile.open(view_shard_path(output, ref, "raw", view_entry.shard), "r") as tar:
                self.assertEqual(tar.extractfile(view_entry.key).read(), b"RIFF-data")

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
            sample = Sample(
                data={
                    ModalityKey.AUDIO: {
                        AudioKey.SAMPLE_RATE: 16000,
                        AudioKey.VIEWS: {
                            AudioView.FILE: str(Path(tmpdir) / "missing.wav"),
                        },
                    },
                },
                dataset_name="files:train",
                sample_index=0,
            )
            writer = DatasetWriter(
                output,
                dataset_id="toy",
                views=(ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM),),
            )

            with self.assertRaises(KeyError):
                writer.write([sample])

            self.assertFalse(output.exists())

    def test_writer_rejects_sample_without_supported_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            sample = Sample(
                data={
                    ModalityKey.AUDIO: {
                        AudioKey.SAMPLE_RATE: 16000,
                        AudioKey.VIEWS: {
                            AudioView.LONGCAT: {"semantic_codes": [1, 2]},
                        },
                    },
                },
                dataset_name="longcat:train",
                sample_index=0,
            )

            with self.assertRaises(ValueError):
                DatasetWriter(output, dataset_id="toy").write([sample])

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
