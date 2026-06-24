import unittest

from anydataset import (
    AudioCodecKey,
    AudioKey,
    AudioOptKey,
    AudioView,
    ModalityKey,
    TextKey,
    TextOptKey,
    ViewRef,
)
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

    def test_audio_schema_keys_use_auto_str_values(self):
        from anydataset.tasks.audio_codec import AudioKey as AudioCodecAudioKey
        from anydataset.tasks.audio_codec import AudioView as AudioCodecAudioView

        self.assertIs(AudioCodecAudioKey, AudioKey)
        self.assertIs(AudioCodecAudioView, AudioView)
        self.assertIs(AudioCodecKey.TEXT, ModalityKey.TEXT)
        self.assertIsInstance(ModalityKey.AUDIO, str)
        self.assertEqual(ModalityKey.TEXT.value, "text")
        self.assertNotIn("AUDIO", AudioKey.__members__)
        self.assertNotIn("SOURCE_SAMPLE_RATE", AudioKey.__members__)
        self.assertEqual(AudioKey.VIEWS.value, "views")
        self.assertEqual(AudioKey.SAMPLE_RATE.value, "sample_rate")
        self.assertEqual(AudioOptKey.DURATION.value, "duration")
        self.assertEqual(AudioOptKey.LABEL.value, "label")
        self.assertEqual(AudioOptKey.LABELS.value, "labels")
        self.assertEqual(AudioView.WAVEFORM.value, "waveform")
        self.assertEqual(AudioView.FILE.value, "file")
        self.assertEqual(AudioView.LONGCAT.value, "longcat")
        self.assertEqual(TextKey.CONTENT.value, "content")
        self.assertEqual(TextOptKey.LANG.value, "lang")

    def test_view_ref_uses_empty_default_role(self):
        default = ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM)
        self.assertIsNone(default.role)
        self.assertEqual(default.view_key, "waveform")
        self.assertEqual(default.path_parts(), ("audio", "views", "waveform"))

        source = ViewRef(ModalityKey.AUDIO, AudioView.LONGCAT, role="source")
        self.assertEqual(source.path_parts(), ("audio", "source", "views", "longcat"))

    def test_view_ref_rejects_reserved_role(self):
        with self.assertRaises(ValueError):
            ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM, role="views")

        with self.assertRaises(ValueError):
            ViewRef(ModalityKey.AUDIO, "bad/key")

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
                    ModalityKey.AUDIO: {
                        AudioKey.SAMPLE_RATE: 4,
                        AudioKey.VIEWS: {
                            AudioView.WAVEFORM: [1.0, 2.0, 3.0, 4.0],
                        },
                    },
                    ModalityKey.TEXT: {
                        TextKey.CONTENT: "hello",
                        TextOptKey.LANG: "en",
                    },
                },
                dataset_name="audio",
                sample_index=0,
            )
        )

        self.assertEqual(set(sample.data), {"audio", "text"})
        audio = sample.data[ModalityKey.AUDIO]
        self.assertEqual(set(audio), {"views", "sample_rate", "duration"})
        self.assertEqual(tuple(audio[AudioKey.VIEWS][AudioView.WAVEFORM].shape), (2, 2))
        self.assertEqual(audio[AudioKey.SAMPLE_RATE], 4)
        self.assertEqual(audio[AudioOptKey.DURATION], 0.5)
        self.assertEqual(sample.data[ModalityKey.TEXT][TextKey.CONTENT], "hello")
        self.assertEqual(sample.data[ModalityKey.TEXT][TextOptKey.LANG], "en")
        self.assertTrue(
            torch.equal(
                audio[AudioKey.VIEWS][AudioView.WAVEFORM],
                torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
            )
        )

    def test_audio_codec_task_keeps_short_samples_short(self):
        task = AudioCodecTask(sample_rate=4, channels=1, max_clip_seconds=1.0)
        sample = task(
            Sample(
                data={
                    ModalityKey.AUDIO: {
                        AudioKey.SAMPLE_RATE: 4,
                        AudioKey.VIEWS: {
                            AudioView.WAVEFORM: [1.0, 2.0],
                        },
                    },
                },
                dataset_name="audio",
                sample_index=0,
            )
        )

        audio = sample.data[ModalityKey.AUDIO]
        self.assertEqual(tuple(audio[AudioKey.VIEWS][AudioView.WAVEFORM].shape), (1, 2))
        self.assertNotIn(ModalityKey.TEXT, sample.data)

    def test_audio_codec_task_accepts_plain_string_keys(self):
        task = AudioCodecTask(sample_rate=2, channels=1, max_clip_seconds=1.0)
        sample = task(
            Sample(
                data={
                    "audio": {
                        "sample_rate": 2,
                        "views": {
                            "waveform": [1.0, 2.0],
                        },
                    },
                    "text": {
                        "content": "hello",
                    },
                },
                dataset_name="audio",
                sample_index=0,
            )
        )

        waveform = sample.data[ModalityKey.AUDIO][AudioKey.VIEWS][AudioView.WAVEFORM]
        self.assertEqual(tuple(waveform.shape), (1, 2))
        self.assertEqual(sample.data[ModalityKey.TEXT][TextKey.CONTENT], "hello")

    def test_audio_codec_task_rejects_missing_sample_rate(self):
        task = AudioCodecTask(sample_rate=4, channels=1)

        with self.assertRaises(KeyError):
            task(
                Sample(
                    data={
                        ModalityKey.AUDIO: {
                            AudioKey.VIEWS: {
                                AudioView.WAVEFORM: [1.0],
                            },
                        },
                    },
                    dataset_name="audio",
                    sample_index=0,
                )
            )

    def test_audio_codec_task_rejects_text_without_content(self):
        task = AudioCodecTask(sample_rate=4, channels=1)

        with self.assertRaises(KeyError):
            task(
                Sample(
                    data={
                        ModalityKey.AUDIO: {
                            AudioKey.SAMPLE_RATE: 4,
                            AudioKey.VIEWS: {
                                AudioView.WAVEFORM: [1.0],
                            },
                        },
                        ModalityKey.TEXT: {
                            TextOptKey.LANG: "en",
                        },
                    },
                    dataset_name="audio",
                    sample_index=0,
                )
            )

    def test_audio_codec_task_rejects_empty_waveform(self):
        task = AudioCodecTask(sample_rate=4, channels=1)

        with self.assertRaises(ValueError):
            task(
                Sample(
                    data={
                        ModalityKey.AUDIO: {
                            AudioKey.SAMPLE_RATE: 4,
                            AudioKey.VIEWS: {
                                AudioView.WAVEFORM: [],
                            },
                        },
                    },
                    dataset_name="audio",
                    sample_index=0,
                )
            )


if __name__ == "__main__":
    unittest.main()
