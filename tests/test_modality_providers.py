import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

from anydataset.dataset import collate_fn
from anydataset.types import (
    AudioItem,
    AudioReq,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextReq,
    TextView,
)
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
        self.assertEqual(FakeMossTTS.loaded.synthesize_calls, [("hello", options, None)])
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

    def test_moss_tts_provider_synthesizes_text_batch(self):
        FakeMossTTS.calls = []
        FakeMossTTS.loaded = None
        options = object()
        with _fake_anytrain_tts():
            provider = MossTTSProvider(options=options, device="cpu")

        outputs = provider.call_batch(
            collate_fn(
                {
                    (Role.DEFAULT, Modality.TEXT): TextReq(
                        views=frozenset({TextView.TEXT})
                    )
                }
            )(
                [
                    {
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello"}
                        )
                    },
                    {
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "world"}
                        )
                    },
                ]
            )
        )

        self.assertEqual(
            FakeMossTTS.loaded.synthesize_calls,
            [(["hello", "world"], options, None)],
        )
        self.assertEqual(len(outputs), 2)
        self.assertTrue(torch.equal(outputs[0][0], torch.tensor([[0.0, 1.0]])))
        self.assertTrue(torch.equal(outputs[1][0], torch.tensor([[2.0, 3.0]])))
        self.assertEqual([sample_rate for _, sample_rate in outputs], [16000, 16000])

    def test_moss_tts_provider_synthesizes_multiple_text_roles(self):
        FakeMossTTS.calls = []
        FakeMossTTS.loaded = None
        options = object()
        with _fake_anytrain_tts():
            provider = MossTTSProvider(options=options, device="cpu")

        outputs = provider.call_batch(
            collate_fn(
                {
                    (Role.SOURCE, Modality.TEXT): TextReq(
                        views=frozenset({TextView.TEXT})
                    ),
                    (Role.TARGET, Modality.TEXT): TextReq(
                        views=frozenset({TextView.TEXT})
                    ),
                }
            )(
                [
                    _text_pair("hello", "hi"),
                    _text_pair("world", "ok"),
                ]
            )
        )

        self.assertEqual(
            FakeMossTTS.loaded.synthesize_calls,
            [
                (["hello", "world"], options, None),
                (["hi", "ok"], options, None),
            ],
        )
        self.assertIsInstance(outputs, dict)
        source = outputs[(Role.SOURCE, Modality.TEXT)]
        target = outputs[(Role.TARGET, Modality.TEXT)]
        self.assertTrue(torch.equal(source[0][0], torch.tensor([[0.0, 1.0]])))
        self.assertTrue(torch.equal(target[0][0], torch.tensor([[0.0, 1.0]])))
        self.assertEqual([sample_rate for _, sample_rate in source], [16000, 16000])
        self.assertEqual([sample_rate for _, sample_rate in target], [16000, 16000])

    def test_moss_tts_provider_uses_reference_audio_role(self):
        FakeMossTTS.calls = []
        FakeMossTTS.loaded = None
        with _fake_anytrain_tts():
            provider = MossTTSProvider(
                reference_role=Role.SOURCE,
                max_reference_files=2,
                device="cpu",
            )

        batch = collate_fn(
            {
                (Role.TARGET, Modality.TEXT): TextReq(
                    views=frozenset({TextView.TEXT})
                ),
                (Role.SOURCE, Modality.AUDIO): AudioReq(
                    views=frozenset({AudioView.WAVEFORM})
                ),
            }
        )(
            [
                {
                    (Role.TARGET, Modality.TEXT): TextItem(
                        views={TextView.TEXT: "hello"}
                    ),
                    (Role.SOURCE, Modality.AUDIO): AudioItem(
                        views={
                            AudioView.WAVEFORM: (
                                torch.tensor([[1.0, 2.0, 3.0]]),
                                16000,
                            )
                        }
                    ),
                },
                {
                    (Role.TARGET, Modality.TEXT): TextItem(
                        views={TextView.TEXT: "world"}
                    ),
                    (Role.SOURCE, Modality.AUDIO): AudioItem(
                        views={
                            AudioView.WAVEFORM: (
                                torch.tensor([[4.0]]),
                                16000,
                            )
                        }
                    ),
                },
            ]
        )

        saved = []

        class FakeTorchAudio:
            @staticmethod
            def save(path, waveform, sample_rate):
                saved.append((Path(path).name, waveform.clone(), sample_rate))
                Path(path).write_bytes(b"wav")

        with patch("anydataset.provider.moss_tts.torchaudio", FakeTorchAudio()):
            outputs = provider.call_batch(batch)

        self.assertEqual(FakeMossTTS.loaded.synthesize_calls[0][0], ["hello", "world"])
        _, _, reference_audio_paths = FakeMossTTS.loaded.synthesize_calls[0]
        self.assertEqual(len(reference_audio_paths), 2)
        self.assertTrue(all(Path(path).is_file() for path in reference_audio_paths))
        self.assertEqual([name for name, _, _ in saved], ["ref-00000000.wav", "ref-00000001.wav"])
        self.assertEqual([tuple(wave.shape) for _, wave, _ in saved], [(1, 3), (1, 1)])
        self.assertEqual(len(outputs), 2)

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

    def test_whisper_asr_provider_transcribes_waveform_batch(self):
        FakeWhisperASREvaluator.calls = []
        FakeWhisperASREvaluator.loaded = None
        with _fake_anytrain_asr():
            provider = WhisperASRProvider(device="cpu")

        outputs = provider.call_batch(
            collate_fn(
                {
                    (Role.DEFAULT, Modality.AUDIO): AudioReq(
                        views=frozenset({AudioView.WAVEFORM})
                    )
                }
            )(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={
                                AudioView.WAVEFORM: (
                                    torch.tensor([[1.0, 2.0, 3.0]]),
                                    16000,
                                )
                            }
                        )
                    },
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={
                                AudioView.WAVEFORM: (
                                    torch.tensor([[4.0]]),
                                    16000,
                                )
                            }
                        )
                    },
                ]
            )
        )

        self.assertEqual(outputs, ["hello-0", "hello-1"])
        self.assertEqual(len(FakeWhisperASREvaluator.loaded.transcribe_calls), 1)
        waveform, sample_rate = FakeWhisperASREvaluator.loaded.transcribe_calls[0]
        self.assertEqual(tuple(waveform.shape), (2, 1, 3))
        self.assertTrue(torch.equal(waveform[1], torch.tensor([[4.0, 0.0, 0.0]])))
        self.assertEqual(sample_rate, 16000)

    def test_whisper_asr_provider_transcribes_file_batch(self):
        FakeWhisperASREvaluator.calls = []
        FakeWhisperASREvaluator.loaded = None
        with _fake_anytrain_asr():
            provider = WhisperASRProvider(device="cpu")

        batch = collate_fn(
            {
                (Role.DEFAULT, Modality.AUDIO): AudioReq(
                    views=frozenset({AudioView.FILE})
                )
            }
        )(
            [
                {
                    (Role.DEFAULT, Modality.AUDIO): AudioItem(
                        views={AudioView.FILE: b"first"}
                    )
                },
                {
                    (Role.DEFAULT, Modality.AUDIO): AudioItem(
                        views={AudioView.FILE: b"second"}
                    )
                },
            ]
        )

        class FakeTorchAudio:
            @staticmethod
            def load(source):
                payload = source.getvalue()
                if payload == b"first":
                    return torch.tensor([[1.0, 2.0, 3.0]]), 16000
                if payload == b"second":
                    return torch.tensor([[4.0]]), 16000
                raise AssertionError(source)

        with patch("anydataset.provider.abc.torchaudio", FakeTorchAudio()):
            outputs = provider.call_batch(batch)

        self.assertEqual(outputs, ["hello-0", "hello-1"])
        self.assertEqual(len(FakeWhisperASREvaluator.loaded.transcribe_calls), 1)
        waveform, sample_rate = FakeWhisperASREvaluator.loaded.transcribe_calls[0]
        self.assertEqual(tuple(waveform.shape), (2, 1, 3))
        self.assertTrue(torch.equal(waveform[1], torch.tensor([[4.0, 0.0, 0.0]])))
        self.assertEqual(sample_rate, 16000)

    def test_whisper_asr_provider_transcribes_multiple_audio_roles(self):
        FakeWhisperASREvaluator.calls = []
        FakeWhisperASREvaluator.loaded = None
        with _fake_anytrain_asr():
            provider = WhisperASRProvider(device="cpu")

        outputs = provider.call_batch(
            collate_fn(
                {
                    (Role.SOURCE, Modality.AUDIO): AudioReq(
                        views=frozenset({AudioView.WAVEFORM})
                    ),
                    (Role.TARGET, Modality.AUDIO): AudioReq(
                        views=frozenset({AudioView.WAVEFORM})
                    ),
                }
            )(
                [
                    _audio_pair(
                        source=torch.tensor([[1.0, 2.0, 3.0]]),
                        target=torch.tensor([[4.0]]),
                    ),
                    _audio_pair(
                        source=torch.tensor([[5.0]]),
                        target=torch.tensor([[6.0, 7.0]]),
                    ),
                ]
            )
        )

        self.assertEqual(
            outputs,
            {
                (Role.SOURCE, Modality.AUDIO): ["hello-0", "hello-1"],
                (Role.TARGET, Modality.AUDIO): ["hello-0", "hello-1"],
            },
        )
        self.assertEqual(len(FakeWhisperASREvaluator.loaded.transcribe_calls), 2)
        source_waveform, _ = FakeWhisperASREvaluator.loaded.transcribe_calls[0]
        target_waveform, _ = FakeWhisperASREvaluator.loaded.transcribe_calls[1]
        self.assertEqual(tuple(source_waveform.shape), (2, 1, 3))
        self.assertEqual(tuple(target_waveform.shape), (2, 1, 2))

    def test_whisper_asr_provider_requires_one_sample_rate_per_batch(self):
        with _fake_anytrain_asr():
            provider = WhisperASRProvider(device="cpu")

        batch = collate_fn(
            {
                (Role.DEFAULT, Modality.AUDIO): AudioReq(
                    views=frozenset({AudioView.WAVEFORM})
                )
            }
        )(
            [
                {
                    (Role.DEFAULT, Modality.AUDIO): AudioItem(
                        views={AudioView.WAVEFORM: (torch.tensor([[1.0]]), 16000)}
                    )
                },
                {
                    (Role.DEFAULT, Modality.AUDIO): AudioItem(
                        views={AudioView.WAVEFORM: (torch.tensor([[2.0]]), 24000)}
                    )
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "one sample rate"):
            provider.call_batch(batch)


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

    def synthesize(
        self,
        text,
        options,
        reference_audio_path=None,
        reference_audio_paths=None,
    ):
        references = reference_audio_path if isinstance(text, str) else reference_audio_paths
        self.synthesize_calls.append((text, options, references))
        if not isinstance(text, str):
            return [
                _TTSOutput(
                    torch.tensor([[float(index * 2), float(index * 2 + 1)]]),
                    16000,
                )
                for index, _ in enumerate(text)
            ]
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
        if isinstance(waveform, torch.Tensor) and waveform.ndim > 2:
            return [f"hello-{index}" for index in range(waveform.shape[0])]
        return "hello"


def _text_pair(source: str, target: str):
    return {
        (Role.SOURCE, Modality.TEXT): TextItem(views={TextView.TEXT: source}),
        (Role.TARGET, Modality.TEXT): TextItem(views={TextView.TEXT: target}),
    }


def _audio_pair(*, source: torch.Tensor, target: torch.Tensor):
    return {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (source, 16000)}
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (target, 16000)}
        ),
    }


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
