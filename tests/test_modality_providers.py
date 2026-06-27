import sys
import types
import unittest
from dataclasses import dataclass

import torch

from anydataset import AudioView, TextView
from anydataset.provider.moss_tts import MossTTSProvider
from anydataset.provider.whisper import WhisperASRProvider


class ModalityProviderTest(unittest.TestCase):
    def test_moss_tts_provider_loads_preset_and_synthesizes(self):
        FakeMossTTS.calls = []
        FakeMossTTS.loaded = None
        options = object()
        with _fake_anytrain_tts():
            provider = MossTTSProvider(
                "fake-moss",
                options=options,
                device="cpu",
                trust_remote_code=True,
                runtime_kwargs={"style": "clear"},
            )

        waveform, sample_rate = provider({TextView.TEXT: "hello"})

        self.assertEqual(
            FakeMossTTS.calls,
            [
                (
                    "fake-moss",
                    {
                        "device": "cpu",
                        "runtime_kwargs": {"style": "clear"},
                        "trust_remote_code": True,
                    },
                )
            ],
        )
        self.assertEqual(FakeMossTTS.loaded.synthesize_calls, [("hello", options)])
        self.assertTrue(torch.equal(waveform, torch.tensor([[1.0, 2.0]])))
        self.assertEqual(sample_rate, 16000)

    def test_moss_tts_provider_uses_anytrain_default_model(self):
        FakeMossTTS.calls = []
        FakeMossTTS.loaded = None
        with _fake_anytrain_tts():
            MossTTSProvider(device="cpu")

        self.assertEqual(
            FakeMossTTS.calls,
            [
                (
                    "__default__",
                    {
                        "device": "cpu",
                        "runtime_kwargs": None,
                    },
                )
            ],
        )

    def test_whisper_asr_provider_loads_preset_and_transcribes(self):
        FakeWhisperASREvaluator.calls = []
        FakeWhisperASREvaluator.loaded = None
        with _fake_anytrain_asr():
            provider = WhisperASRProvider(
                model_name="tiny",
                device="cpu",
                decode_options={"language": "en"},
                load_options={"in_memory": True},
            )

        text = provider(
            {AudioView.WAVEFORM: (torch.tensor([[1.0, 2.0]]), 16000)}
        )

        self.assertEqual(text, "hello")
        self.assertEqual(
            FakeWhisperASREvaluator.calls,
            [
                {
                    "decode_options": {"language": "en"},
                    "device": "cpu",
                    "load_options": {"in_memory": True},
                    "model_name": "tiny",
                }
            ],
        )
        self.assertEqual(len(FakeWhisperASREvaluator.loaded.transcribe_calls), 1)
        waveform, sample_rate = FakeWhisperASREvaluator.loaded.transcribe_calls[0]
        self.assertTrue(torch.equal(waveform, torch.tensor([[1.0, 2.0]])))
        self.assertEqual(sample_rate, 16000)


@dataclass
class _TTSOutput:
    waveform: torch.Tensor
    sample_rate: int


class FakeMossTTS:
    calls = []
    loaded = None

    def __init__(self) -> None:
        self.synthesize_calls = []

    @classmethod
    def from_pretrained(cls, model="__default__", **kwargs):
        cls.calls.append((model, kwargs))
        cls.loaded = cls()
        return cls.loaded

    def synthesize(self, text, options):
        self.synthesize_calls.append((text, options))
        return _TTSOutput(torch.tensor([[1.0, 2.0]]), 16000)


class FakeWhisperASREvaluator:
    calls = []
    loaded = None

    def __init__(self, **kwargs) -> None:
        self.transcribe_calls = []
        type(self).calls.append(kwargs)
        type(self).loaded = self

    def transcribe(self, waveform, sample_rate):
        self.transcribe_calls.append((waveform, sample_rate))
        return "hello"


class _fake_anytrain_tts:
    def __init__(self) -> None:
        self.previous = {}

    def __enter__(self):
        modules = {
            "anytrain": types.ModuleType("anytrain"),
            "anytrain.tts": types.ModuleType("anytrain.tts"),
            "anytrain.tts.moss": types.ModuleType("anytrain.tts.moss"),
        }
        modules["anytrain.tts.moss"].MossTTS = FakeMossTTS
        self.previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, module in self.previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _fake_anytrain_asr:
    def __init__(self) -> None:
        self.previous = {}

    def __enter__(self):
        modules = {
            "anytrain": types.ModuleType("anytrain"),
            "anytrain.evaluator": types.ModuleType("anytrain.evaluator"),
            "anytrain.evaluator.speech": types.ModuleType("anytrain.evaluator.speech"),
        }
        modules["anytrain.evaluator.speech"].WhisperASREvaluator = FakeWhisperASREvaluator
        self.previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, module in self.previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
