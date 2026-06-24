import tempfile
import unittest
from pathlib import Path

import torch

from anydataset import (
    AnyDataset,
    AudioKey,
    AudioOptKey,
    AudioView,
    DatasetSpec,
    ModalityKey,
    Task,
    TextKey,
    UnifiedDatasetAdapter,
    ViewRef,
)
from anydataset.samples import Sample
from anydataset.store import DatasetWriter


class UnifiedDatasetAdapterTest(unittest.TestCase):
    def test_anydataset_reads_waveform_dataset_written_by_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(
                [
                    Sample(
                        data={
                            ModalityKey.AUDIO: {
                                AudioKey.SAMPLE_RATE: 4,
                                AudioOptKey.DURATION: 0.75,
                                AudioOptKey.LABEL: "speech",
                                AudioOptKey.LABELS: {"speaker": "alice"},
                                AudioKey.VIEWS: {
                                    AudioView.WAVEFORM: waveform,
                                },
                            },
                            ModalityKey.TEXT: {
                                TextKey.CONTENT: "hello",
                            },
                        },
                        dataset_name="source:train",
                        sample_index=7,
                    )
                ]
            )

            dataset = AnyDataset(
                datasets=DatasetSpec(
                    source="unified",
                    path=str(output),
                    name="toy_unified",
                    split="train",
                ),
                task=Task.AUDIO_CODEC,
                cache_dir=root / "cache",
            )
            sample = next(iter(dataset))

        audio = sample.data[ModalityKey.AUDIO]
        self.assertEqual(sample.dataset_name, "toy_unified:train")
        self.assertEqual(sample.sample_index, 0)
        self.assertTrue(torch.equal(audio[AudioKey.VIEWS][AudioView.WAVEFORM], waveform))
        self.assertEqual(audio[AudioKey.SAMPLE_RATE], 4)
        self.assertEqual(audio[AudioOptKey.DURATION], 0.75)
        self.assertEqual(audio[AudioOptKey.LABEL], "speech")
        self.assertEqual(audio[AudioOptKey.LABELS], {"speaker": "alice"})
        self.assertEqual(sample.data[ModalityKey.TEXT][TextKey.CONTENT], "hello")

    def test_file_view_is_extracted_to_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [
                    Sample(
                        data={
                            ModalityKey.AUDIO: {
                                AudioKey.SAMPLE_RATE: 16000,
                                AudioKey.VIEWS: {
                                    AudioView.FILE: str(source),
                                },
                            },
                        },
                        dataset_name="files:train",
                        sample_index=0,
                    )
                ]
            )

            dataset = AnyDataset(
                datasets=DatasetSpec(source="unified", path=str(output), name="files"),
                task=Task.AUDIO_CODEC,
                cache_dir=root / "cache",
            )
            sample = next(iter(dataset))

            file_view = Path(sample.data[ModalityKey.AUDIO][AudioKey.VIEWS][AudioView.FILE])

            self.assertTrue(file_view.is_file())
            self.assertEqual(file_view.read_bytes(), b"RIFF-data")
            self.assertTrue(str(file_view).startswith(str(root / "cache")))

    def test_explicit_missing_view_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [
                    Sample(
                        data={
                            ModalityKey.AUDIO: {
                                AudioKey.SAMPLE_RATE: 16000,
                                AudioKey.VIEWS: {
                                    AudioView.FILE: str(source),
                                },
                            },
                        },
                        dataset_name="files:train",
                        sample_index=0,
                    )
                ]
            )

            dataset = AnyDataset(
                datasets=DatasetSpec(source="unified", path=str(output), name="files"),
                task=Task.AUDIO_CODEC,
                adapter_map={
                    "files": UnifiedDatasetAdapter(
                        views=(ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM),)
                    ),
                },
                cache_dir=root / "cache",
            )

            with self.assertRaises(KeyError):
                list(dataset)

    def test_sharded_iteration_is_disjoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            samples = [
                Sample(
                    data={
                        ModalityKey.AUDIO: {
                            AudioKey.SAMPLE_RATE: 4,
                            AudioKey.VIEWS: {
                                AudioView.WAVEFORM: torch.tensor([[float(index)]]),
                            },
                        },
                    },
                    dataset_name="source:train",
                    sample_index=index,
                )
                for index in range(4)
            ]
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(samples)
            dataset = AnyDataset(
                datasets=DatasetSpec(
                    source="unified",
                    path=str(output),
                    name="toy_unified",
                    split="train",
                ),
                task=Task.AUDIO_CODEC,
                cache_dir=root / "cache",
            )

            shard_zero = [sample.sample_index for sample in dataset.shard(2, 0)]
            shard_one = [sample.sample_index for sample in dataset.shard(2, 1)]

        self.assertEqual(shard_zero, [0, 2])
        self.assertEqual(shard_one, [1, 3])
        self.assertEqual(sorted(shard_zero + shard_one), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
