import unittest

from anydataset import (
    AudioKey,
    AudioOptKey,
    AudioView,
    ModalityKey,
    TextKey,
    TextOptKey,
)
from anydataset.adapters import (
    ESC50Adapter,
    FSD50KAdapter,
    FleursAdapter,
    LibriSpeechASRAdapter,
    LocalFilesAdapter,
    MissingModalityError,
    NSynthAdapter,
)
from anydataset.tasks.audio_codec import AudioCodecAdapter


class ModalityAdapterTest(unittest.TestCase):
    def test_local_files_adapter_maps_dataset_fields_to_modalities(self):
        adapter = LocalFilesAdapter(
            audio_field="audio",
            text_field="sentence",
            duration_field="seconds",
            label_field="category",
            labels_fields={
                "target": "target",
                "subset": "subset",
            },
            file_field="path",
        )
        row = {
            "audio": {
                "array": [1.0, 2.0],
                "sampling_rate": 16000,
            },
            "sentence": "hello",
            "seconds": 0.5,
            "category": "speech",
            "target": 1,
            "subset": "toy",
            "path": "/tmp/a.wav",
        }

        audio = adapter.audio(row)
        text = adapter.text(row)

        self.assertEqual(audio[AudioKey.VIEWS][AudioView.WAVEFORM], [1.0, 2.0])
        self.assertEqual(audio[AudioKey.VIEWS][AudioView.FILE], "/tmp/a.wav")
        self.assertEqual(audio[AudioKey.SAMPLE_RATE], 16000)
        self.assertEqual(audio[AudioOptKey.DURATION], 0.5)
        self.assertEqual(audio[AudioOptKey.LABEL], "speech")
        self.assertEqual(audio[AudioOptKey.LABELS], {"target": 1, "subset": "toy"})
        self.assertEqual(text[TextKey.CONTENT], "hello")

    def test_local_files_adapter_decodes_torchcodec_audio(self):
        adapter = LocalFilesAdapter(audio_field="audio")

        audio = adapter.audio({"audio": _FakeAudioDecoder()})

        self.assertEqual(audio[AudioKey.VIEWS][AudioView.WAVEFORM], [0.1, 0.2])
        self.assertEqual(audio[AudioKey.SAMPLE_RATE], 16000)

    def test_text_roles_map_to_dataset_fields(self):
        adapter = LocalFilesAdapter(
            text_fields={
                "source": "en",
                "target": "zh",
            },
            lang_fields={
                "source": "source_lang",
                "target": "target_lang",
            },
        )
        row = {
            "en": "hello",
            "zh": "ni hao",
            "source_lang": "en",
            "target_lang": "zh",
        }

        source = adapter.text(row, role="source")
        target = adapter.text(row, role="target")

        self.assertEqual(source[TextKey.CONTENT], "hello")
        self.assertEqual(source[TextOptKey.LANG], "en")
        self.assertEqual(target[TextKey.CONTENT], "ni hao")
        self.assertEqual(target[TextOptKey.LANG], "zh")
        with self.assertRaises(MissingModalityError):
            adapter.text(row)

    def test_builtin_text_adapters_map_text(self):
        audio_row = {"array": [0.1, 0.2], "sampling_rate": 16000}
        cases = [
            (FleursAdapter(), {"audio": audio_row, "transcription": "bonjour"}, "en_us"),
            (LibriSpeechASRAdapter(), {"audio": audio_row, "text": "hello"}, "en"),
        ]

        for adapter, row, lang in cases:
            with self.subTest(adapter=type(adapter).__name__):
                audio = adapter.audio(row)
                text = adapter.text(row)

                self.assertEqual(audio[AudioKey.VIEWS][AudioView.WAVEFORM], [0.1, 0.2])
                self.assertEqual(audio[AudioKey.SAMPLE_RATE], 16000)
                self.assertIsInstance(text[TextKey.CONTENT], str)
                self.assertEqual(text[TextOptKey.LANG], lang)

    def test_builtin_audio_adapters_map_audio_opt_keys_and_reject_text(self):
        audio_row = {"array": [0.1, 0.2], "sampling_rate": 44100}
        cases = [
            (
                ESC50Adapter(),
                {
                    "audio": audio_row,
                    "category": "dog",
                    "target": 3,
                    "esc10": True,
                    "text": "ignored",
                },
                "dog",
                {"target": 3, "esc10": True},
                None,
            ),
            (
                FSD50KAdapter(),
                {
                    "audio": audio_row,
                    "audio_path": "/tmp/fsd.wav",
                    "text": "ignored",
                },
                None,
                None,
                "/tmp/fsd.wav",
            ),
            (
                NSynthAdapter(),
                {
                    "audio": audio_row,
                    "instrument_family_str": "guitar",
                    "instrument_source_str": "acoustic",
                    "pitch": 60,
                    "velocity": 80,
                    "text": "ignored",
                },
                None,
                {
                    "instrument_family": "guitar",
                    "instrument_source": "acoustic",
                    "pitch": 60,
                    "velocity": 80,
                },
                None,
            ),
        ]

        for adapter, row, label, labels, file_path in cases:
            with self.subTest(adapter=type(adapter).__name__):
                audio = adapter.audio(row)

                self.assertEqual(audio[AudioKey.VIEWS][AudioView.WAVEFORM], [0.1, 0.2])
                self.assertEqual(audio[AudioKey.SAMPLE_RATE], 44100)
                if label is not None:
                    self.assertEqual(audio[AudioOptKey.LABEL], label)
                if labels is not None:
                    self.assertEqual(audio[AudioOptKey.LABELS], labels)
                if file_path is not None:
                    self.assertEqual(audio[AudioKey.VIEWS][AudioView.FILE], file_path)
                with self.assertRaises(MissingModalityError):
                    adapter.text(row)


class AudioCodecTaskAdapterTest(unittest.TestCase):
    def test_audio_codec_adapter_assembles_task_sample(self):
        row = {
            "audio": {
                "array": [1.0, 2.0],
                "sampling_rate": 16000,
            },
            "transcription": "hello",
        }

        sample = AudioCodecAdapter().adapt(row, FleursAdapter())

        self.assertEqual(
            sample[ModalityKey.AUDIO][AudioKey.VIEWS][AudioView.WAVEFORM],
            [1.0, 2.0],
        )
        self.assertEqual(sample[ModalityKey.AUDIO][AudioKey.SAMPLE_RATE], 16000)
        self.assertEqual(sample[ModalityKey.TEXT][TextKey.CONTENT], "hello")
        self.assertEqual(sample[ModalityKey.TEXT][TextOptKey.LANG], "en_us")

    def test_audio_codec_adapter_omits_missing_optional_text(self):
        row = {
            "audio": {
                "array": [1.0, 2.0],
                "sampling_rate": 16000,
            },
            "category": "dog",
            "target": 3,
            "esc10": True,
        }

        sample = AudioCodecAdapter().adapt(row, ESC50Adapter())

        self.assertIn(ModalityKey.AUDIO, sample)
        self.assertNotIn(ModalityKey.TEXT, sample)


class _FakeAudioDecoder:
    def get_all_samples(self):
        return _FakeAudioSamples()


class _FakeAudioSamples:
    data = [0.1, 0.2]
    sample_rate = 16000


if __name__ == "__main__":
    unittest.main()
