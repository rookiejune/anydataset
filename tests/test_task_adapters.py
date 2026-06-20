import unittest

from anydataset import DatasetSpec, Task, TaskAdapterRegistry
from anydataset.datasets.esc50 import ESC50AudioCodecAdapter
from anydataset.datasets.fleurs import (
    FleursAudioCodecAdapter,
    register_task_adapters as register_fleurs_task_adapters,
)
from anydataset.datasets.fsd50k import FSD50KAudioCodecAdapter
from anydataset.datasets.librispeech_asr import LibriSpeechASRAudioCodecAdapter
from anydataset.datasets.local_files.adapters.audio_codec import AudioCodecSampleAdapter
from anydataset.datasets.nsynth import NSynthAudioCodecAdapter
from anydataset.datasets.task_adapters import default_task_adapter_registry


class TaskAdapterTest(unittest.TestCase):
    def test_audio_codec_adapter_maps_dataset_fields_to_canonical_sample(self):
        adapter = AudioCodecSampleAdapter(
            audio_key="audio",
            text_key="sentence",
        )

        sample = adapter.adapt(
            {
                "audio": {
                    "array": [1.0, 2.0],
                    "sampling_rate": 16000,
                },
                "sentence": "hello",
            }
        )

        self.assertEqual(sample["waveform"], [1.0, 2.0])
        self.assertEqual(sample["sample_rate"], 16000)
        self.assertEqual(sample["text"], "hello")
        self.assertNotIn("audio_source", sample)

    def test_audio_codec_adapter_decodes_torchcodec_audio(self):
        adapter = AudioCodecSampleAdapter(audio_key="audio")

        sample = adapter.adapt({"audio": _FakeAudioDecoder()})

        self.assertEqual(sample["waveform"], [0.1, 0.2])
        self.assertEqual(sample["sample_rate"], 16000)
        self.assertNotIn("audio_source", sample)

    def test_builtin_text_adapters_map_text(self):
        audio = {"array": [0.1, 0.2], "sampling_rate": 16000}
        cases = [
            (FleursAudioCodecAdapter(), {"audio": audio, "transcription": "bonjour"}),
            (LibriSpeechASRAudioCodecAdapter(), {"audio": audio, "text": "hello"}),
        ]

        for adapter, row in cases:
            with self.subTest(adapter=type(adapter).__name__):
                sample = adapter.adapt(row)

                self.assertEqual(sample["waveform"], [0.1, 0.2])
                self.assertEqual(sample["sample_rate"], 16000)
                self.assertIsInstance(sample["text"], str)
                self.assertNotIn("audio_source", sample)

    def test_builtin_audio_adapters_without_text_do_not_emit_text(self):
        audio = {"array": [0.1, 0.2], "sampling_rate": 44100}
        adapters = [
            ESC50AudioCodecAdapter(),
            FSD50KAudioCodecAdapter(),
            NSynthAudioCodecAdapter(),
        ]

        for adapter in adapters:
            with self.subTest(adapter=type(adapter).__name__):
                sample = adapter.adapt({"audio": audio, "text": "ignored"})

                self.assertEqual(sample["waveform"], [0.1, 0.2])
                self.assertEqual(sample["sample_rate"], 44100)
                self.assertNotIn("text", sample)
                self.assertNotIn("audio_source", sample)


class TaskAdapterRegistryTest(unittest.TestCase):
    def test_default_registry_resolves_builtin_dataset_adapter(self):
        spec = DatasetSpec(
            source="huggingface",
            path="google/fleurs",
            name="fleurs",
            split="train",
        )

        adapter = default_task_adapter_registry().resolve(spec, Task.AUDIO_CODEC)

        self.assertIsInstance(adapter, FleursAudioCodecAdapter)

    def test_dataset_module_registers_own_task_adapter(self):
        registry = TaskAdapterRegistry()
        register_fleurs_task_adapters(registry)
        spec = DatasetSpec(
            source="huggingface",
            path="google/fleurs",
            name="fleurs",
            split="train",
        )

        adapter = registry.resolve(spec, Task.AUDIO_CODEC)

        self.assertIsInstance(adapter, FleursAudioCodecAdapter)

    def test_registry_resolves_by_unique_dataset_name(self):
        registry = TaskAdapterRegistry()
        registry.register(
            "custom_audio",
            Task.AUDIO_CODEC,
            lambda spec: AudioCodecSampleAdapter(
                waveform_key="samples",
                sample_rate_key="sr",
            ),
        )
        spec = DatasetSpec(
            source="local_files",
            path="/tmp/audio.jsonl",
            name="custom_audio",
        )

        adapter = registry.resolve(spec, Task.AUDIO_CODEC)

        self.assertIsInstance(adapter, AudioCodecSampleAdapter)
        self.assertEqual(adapter.waveform_key, "samples")
        self.assertEqual(adapter.sample_rate_key, "sr")

    def test_registry_rejects_duplicate_dataset_task(self):
        registry = TaskAdapterRegistry()
        registry.register("custom_audio", Task.AUDIO_CODEC, lambda spec: AudioCodecSampleAdapter())

        with self.assertRaises(ValueError):
            registry.register(
                "custom_audio",
                Task.AUDIO_CODEC,
                lambda spec: AudioCodecSampleAdapter(),
            )

    def test_registry_rejects_factory_returning_wrong_type(self):
        registry = TaskAdapterRegistry()
        registry.register("custom_audio", Task.AUDIO_CODEC, lambda spec: object())
        spec = DatasetSpec(
            source="local_files",
            path="/tmp/audio.jsonl",
            name="custom_audio",
        )

        with self.assertRaises(TypeError):
            registry.resolve(spec, Task.AUDIO_CODEC)


class _FakeAudioDecoder:
    def get_all_samples(self):
        return _FakeAudioSamples()


class _FakeAudioSamples:
    data = [0.1, 0.2]
    sample_rate = 16000


if __name__ == "__main__":
    unittest.main()
