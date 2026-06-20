import unittest

from anydataset.samples import Sample
from anydataset.tasks import (
    AudioCodecTask,
    ImageClassificationTask,
    Task,
)


class TaskTest(unittest.TestCase):
    def test_task_uses_auto_str_value(self):
        self.assertEqual(Task.IMAGE_CLASSIFICATION.value, "image_classification")
        self.assertEqual(Task.AUDIO_CODEC.value, "audio_codec")

    def test_image_classification_task_formats_one_sample(self):
        import numpy as np

        task = ImageClassificationTask()
        sample = task(
            Sample(
                data={
                    "image": np.zeros((2, 3, 1), dtype=np.uint8),
                    "label": "2",
                },
                dataset_name="array",
                sample_index=5,
            )
        )

        self.assertEqual(tuple(sample.data["image"].shape), (1, 2, 3))
        self.assertEqual(sample.data["label"], 2)

    def test_audio_codec_task_formats_one_sample_without_padding(self):
        import torch

        task = AudioCodecTask(sample_rate=4, channels=2, max_clip_seconds=0.5)
        sample = task(
            Sample(
                data={
                    "waveform": [1.0, 2.0, 3.0, 4.0],
                    "sample_rate": 4,
                    "text": "hello",
                },
                dataset_name="audio",
                sample_index=0,
            )
        )

        self.assertEqual(tuple(sample.data["waveform"].shape), (2, 2))
        self.assertEqual(sample.data["sample_rate"], 4)
        self.assertEqual(sample.data["text"], "hello")
        self.assertTrue(
            torch.equal(
                sample.data["waveform"],
                torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
            )
        )

    def test_audio_codec_task_keeps_short_samples_short(self):
        task = AudioCodecTask(sample_rate=4, channels=1, max_clip_seconds=1.0)
        sample = task(
            Sample(
                data={
                    "waveform": [1.0, 2.0],
                    "sample_rate": 4,
                },
                dataset_name="audio",
                sample_index=0,
            )
        )

        self.assertEqual(tuple(sample.data["waveform"].shape), (1, 2))
        self.assertNotIn("text", sample.data)

    def test_audio_codec_task_rejects_empty_waveform(self):
        task = AudioCodecTask(sample_rate=4, channels=1)

        with self.assertRaises(ValueError):
            task(
                Sample(
                    data={
                        "waveform": [],
                        "sample_rate": 4,
                    },
                    dataset_name="audio",
                    sample_index=0,
                )
            )


if __name__ == "__main__":
    unittest.main()
