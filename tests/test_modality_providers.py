import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch

from anydataset.dataset.collate import collate_fn
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioReq,
    AudioView,
    FileBytes,
    Lang,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextReq,
    TextView,
)
from anydataset.provider.moss_tts import MossTTSProvider
from anydataset.provider.qwen_tts import QwenTTSProvider
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

        output = provider({TextView.TEXT: "hello"})

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
        self.assertIsInstance(output, AudioItem)
        waveform, sample_rate = output.views[AudioView.WAVEFORM]
        self.assertTrue(torch.equal(waveform, torch.tensor([[1.0, 2.0]])))
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(output.meta[AudioMeta.DURATION], 2.0 / 16000.0)

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
        self.assertTrue(
            torch.equal(outputs[0].views[AudioView.WAVEFORM][0], torch.tensor([[0.0, 1.0]]))
        )
        self.assertTrue(
            torch.equal(outputs[1].views[AudioView.WAVEFORM][0], torch.tensor([[2.0, 3.0]]))
        )
        self.assertEqual(
            [output.views[AudioView.WAVEFORM][1] for output in outputs],
            [16000, 16000],
        )
        self.assertEqual(
            [output.meta[AudioMeta.DURATION] for output in outputs],
            [2.0 / 16000.0, 2.0 / 16000.0],
        )

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
        self.assertTrue(
            torch.equal(source[0].views[AudioView.WAVEFORM][0], torch.tensor([[0.0, 1.0]]))
        )
        self.assertTrue(
            torch.equal(target[0].views[AudioView.WAVEFORM][0], torch.tensor([[0.0, 1.0]]))
        )
        self.assertEqual(
            [output.views[AudioView.WAVEFORM][1] for output in source],
            [16000, 16000],
        )
        self.assertEqual(
            [output.views[AudioView.WAVEFORM][1] for output in target],
            [16000, 16000],
        )

    def test_moss_tts_provider_uses_reference_audio_role(self):
        FakeMossTTS.calls = []
        FakeMossTTS.loaded = None
        with _fake_anytrain_tts():
            provider = MossTTSProvider(
                reference_role=Role.SOURCE,
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

        outputs = provider.call_batch(batch)

        self.assertEqual(FakeMossTTS.loaded.synthesize_calls[0][0], ["hello", "world"])
        _, _, references = FakeMossTTS.loaded.synthesize_calls[0]
        self.assertEqual(len(references), 2)
        self.assertTrue(
            torch.equal(references[0].waveform, torch.tensor([[1.0, 2.0, 3.0]]))
        )
        self.assertTrue(torch.equal(references[1].waveform, torch.tensor([[4.0]])))
        self.assertEqual([reference.sample_rate for reference in references], [16000, 16000])
        self.assertEqual(len(outputs), 2)

    def test_moss_tts_provider_preserves_file_bytes_reference_suffixes(self):
        FakeMossTTS.calls = []
        FakeMossTTS.loaded = None
        with _fake_anytrain_tts():
            provider = MossTTSProvider(
                reference_role=Role.SOURCE,
                device="cpu",
            )

        batch = collate_fn(
            {
                (Role.TARGET, Modality.TEXT): TextReq(
                    views=frozenset({TextView.TEXT})
                ),
                (Role.SOURCE, Modality.AUDIO): AudioReq(
                    views=frozenset({AudioView.FILE})
                ),
            }
        )(
            [
                {
                    (Role.TARGET, Modality.TEXT): TextItem(
                        views={TextView.TEXT: "hello"}
                    ),
                    (Role.SOURCE, Modality.AUDIO): AudioItem(
                        views={AudioView.FILE: FileBytes(b"flac", ".flac")}
                    ),
                },
                {
                    (Role.TARGET, Modality.TEXT): TextItem(
                        views={TextView.TEXT: "world"}
                    ),
                    (Role.SOURCE, Modality.AUDIO): AudioItem(
                        views={AudioView.FILE: FileBytes(b"ogg", ".ogg")}
                    ),
                },
            ]
        )

        provider.call_batch(batch)

        _, _, references = FakeMossTTS.loaded.synthesize_calls[0]
        self.assertEqual([reference.suffix for reference in references], [".flac", ".ogg"])
        self.assertEqual([reference.data for reference in references], [b"flac", b"ogg"])

    def test_qwen_tts_provider_synthesizes_text_batch_with_speaker_ids(self):
        FakeQwenCustomVoiceTTS.calls = []
        FakeQwenCustomVoiceTTS.loaded = None
        with _fake_anytrain_qwen_tts():
            provider = QwenTTSProvider(
                "fake-qwen",
                default_language="Auto",
                device="cpu",
                runtime_kwargs={"top_k": 5},
            )

        outputs = provider.call_batch(
            collate_fn(
                {
                    (Role.DEFAULT, Modality.TEXT): TextReq(
                        views=frozenset({TextView.TEXT, TextView.SPEAKERS}),
                    )
                }
            )(
                [
                    {
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello", TextView.SPEAKERS: "Vivian"},
                        )
                    },
                    {
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "world", TextView.SPEAKERS: "Ryan"},
                        )
                    },
                ]
            )
        )

        self.assertEqual(
            FakeQwenCustomVoiceTTS.calls,
            [
                (
                    "fake-qwen",
                    {
                        "device": "cpu",
                        "runtime_kwargs": {"top_k": 5},
                    },
                )
            ],
        )
        self.assertEqual(
            FakeQwenCustomVoiceTTS.loaded.calls,
            [
                (
                    ["hello", "world"],
                    ["Vivian", "Ryan"],
                    ["Auto", "Auto"],
                    None,
                    None,
                )
            ],
        )
        self.assertEqual(len(outputs), 2)
        self.assertTrue(torch.equal(outputs[0].views[AudioView.WAVEFORM][0], torch.tensor([[0.0, 1.0]])))
        self.assertEqual(outputs[0].meta[AudioMeta.SPEAKER_ID], "Vivian")
        self.assertEqual(outputs[1].meta[AudioMeta.SPEAKER_ID], "Ryan")

    def test_qwen_tts_provider_maps_text_language_meta(self):
        FakeQwenCustomVoiceTTS.calls = []
        FakeQwenCustomVoiceTTS.loaded = None
        with _fake_anytrain_qwen_tts():
            provider = QwenTTSProvider(default_language="Auto")

        output = provider(
            {
                TextView.TEXT: "你好",
                TextView.SPEAKERS: "Vivian",
                TextMeta.LANG: Lang.ZH,
            }
        )

        self.assertEqual(
            FakeQwenCustomVoiceTTS.loaded.calls,
            [("你好", "Vivian", "Chinese", None, None)],
        )
        self.assertEqual(output.meta[AudioMeta.SPEAKER_ID], "Vivian")

    def test_qwen_tts_provider_maps_batched_text_language_meta(self):
        FakeQwenCustomVoiceTTS.calls = []
        FakeQwenCustomVoiceTTS.loaded = None
        with _fake_anytrain_qwen_tts():
            provider = QwenTTSProvider(default_language="Auto")

        outputs = provider.call_batch(
            collate_fn(
                {
                    (Role.DEFAULT, Modality.TEXT): TextReq(
                        views=frozenset({TextView.TEXT, TextView.SPEAKERS}),
                        meta=frozenset({TextMeta.LANG}),
                    )
                }
            )(
                [
                    {
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "你好", TextView.SPEAKERS: "Vivian"},
                            meta={TextMeta.LANG: Lang.ZH},
                        )
                    },
                    {
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello", TextView.SPEAKERS: "Ryan"},
                            meta={TextMeta.LANG: Lang.EN},
                        )
                    },
                ]
            )
        )

        self.assertEqual(
            FakeQwenCustomVoiceTTS.loaded.calls,
            [
                (
                    ["你好", "hello"],
                    ["Vivian", "Ryan"],
                    ["Chinese", "English"],
                    None,
                    None,
                )
            ],
        )
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

        self.assertEqual(text, "hello-1")
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
        self.assertTrue(torch.equal(waveform, torch.tensor([[[1.0, 2.0]]])))
        self.assertEqual(sample_rate, 16000)

    def test_whisper_asr_provider_treats_stereo_as_one_sample(self):
        FakeWhisperASREvaluator.calls = []
        FakeWhisperASREvaluator.loaded = None
        with _fake_anytrain_asr():
            provider = WhisperASRProvider(device="cpu")

        text = provider(
            {
                AudioView.WAVEFORM: (
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    16000,
                )
            }
        )

        self.assertEqual(text, "hello-1")
        waveform, sample_rate = FakeWhisperASREvaluator.loaded.transcribe_calls[0]
        self.assertEqual(tuple(waveform.shape), (1, 2, 2))
        self.assertEqual(sample_rate, 16000)

    def test_whisper_asr_provider_groups_waveforms_and_restores_sample_order(self):
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
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={
                                AudioView.WAVEFORM: (
                                    torch.tensor([[5.0, 6.0, 7.0]]),
                                    16000,
                                )
                            }
                        )
                    },
                ]
            )
        )

        self.assertEqual(outputs, ["hello-1", "hello-4", "hello-5"])
        self.assertEqual(len(FakeWhisperASREvaluator.loaded.transcribe_calls), 2)
        first, sample_rate = FakeWhisperASREvaluator.loaded.transcribe_calls[0]
        second, _ = FakeWhisperASREvaluator.loaded.transcribe_calls[1]
        self.assertEqual(tuple(first.shape), (2, 1, 3))
        self.assertTrue(
            torch.equal(
                first,
                torch.tensor([[[1.0, 2.0, 3.0]], [[5.0, 6.0, 7.0]]]),
            )
        )
        self.assertTrue(torch.equal(second, torch.tensor([[[4.0]]])))
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
                {
                    (Role.DEFAULT, Modality.AUDIO): AudioItem(
                        views={AudioView.FILE: b"third"}
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
                if payload == b"third":
                    return torch.tensor([[5.0, 6.0, 7.0]]), 16000
                raise AssertionError(source)

        with patch("anydataset.provider.abc.torchaudio", FakeTorchAudio()):
            outputs = provider.call_batch(batch)

        self.assertEqual(outputs, ["hello-1", "hello-4", "hello-5"])
        self.assertEqual(len(FakeWhisperASREvaluator.loaded.transcribe_calls), 2)
        first, sample_rate = FakeWhisperASREvaluator.loaded.transcribe_calls[0]
        second, _ = FakeWhisperASREvaluator.loaded.transcribe_calls[1]
        self.assertEqual(tuple(first.shape), (2, 1, 3))
        self.assertTrue(
            torch.equal(
                first,
                torch.tensor([[[1.0, 2.0, 3.0]], [[5.0, 6.0, 7.0]]]),
            )
        )
        self.assertTrue(torch.equal(second, torch.tensor([[[4.0]]])))
        self.assertEqual(sample_rate, 16000)

    def test_whisper_asr_provider_transcribes_file_bytes_batch(self):
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
                        views={AudioView.FILE: FileBytes(b"first", ".flac")}
                    )
                },
                {
                    (Role.DEFAULT, Modality.AUDIO): AudioItem(
                        views={AudioView.FILE: FileBytes(b"second", ".ogg")}
                    )
                },
            ]
        )

        class FakeTorchAudio:
            @staticmethod
            def load(source):
                payload = source.getvalue()
                if payload == b"first":
                    return torch.tensor([[1.0, 2.0]]), 16000
                if payload == b"second":
                    return torch.tensor([[3.0]]), 16000
                raise AssertionError(source)

        with patch("anydataset.provider.abc.torchaudio", FakeTorchAudio()):
            outputs = provider.call_batch(batch)

        self.assertEqual(outputs, ["hello-1", "hello-3"])

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
                (Role.SOURCE, Modality.AUDIO): ["hello-1", "hello-5"],
                (Role.TARGET, Modality.AUDIO): ["hello-4", "hello-6"],
            },
        )
        self.assertEqual(len(FakeWhisperASREvaluator.loaded.transcribe_calls), 4)
        source_first, _ = FakeWhisperASREvaluator.loaded.transcribe_calls[0]
        source_second, _ = FakeWhisperASREvaluator.loaded.transcribe_calls[1]
        target_first, _ = FakeWhisperASREvaluator.loaded.transcribe_calls[2]
        target_second, _ = FakeWhisperASREvaluator.loaded.transcribe_calls[3]
        self.assertTrue(torch.equal(source_first, torch.tensor([[[1.0, 2.0, 3.0]]])))
        self.assertTrue(torch.equal(source_second, torch.tensor([[[5.0]]])))
        self.assertTrue(torch.equal(target_first, torch.tensor([[[4.0]]])))
        self.assertTrue(torch.equal(target_second, torch.tensor([[[6.0, 7.0]]])))

    def test_whisper_asr_provider_rejects_wrong_transcribe_batch_size(self):
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
                        views={AudioView.WAVEFORM: (torch.tensor([[2.0]]), 16000)}
                    )
                },
            ]
        )

        with patch.object(
            FakeWhisperASREvaluator.loaded,
            "transcribe",
            return_value=["one"],
        ):
            with self.assertRaisesRegex(ValueError, "one output per input waveform"):
                provider.call_batch(batch)

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


@dataclass(frozen=True)
class _WaveformReference:
    waveform: torch.Tensor
    sample_rate: int


@dataclass(frozen=True)
class _EncodedAudioReference:
    data: bytes
    suffix: str


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
        reference_audio=None,
        reference_audios=None,
    ):
        references = reference_audio if isinstance(text, str) else reference_audios
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


class FakeQwenCustomVoiceTTS:
    calls = []
    loaded = None

    def __init__(self) -> None:
        self.calls = []

    @classmethod
    def from_pretrained(cls, model="__default__", **kwargs):
        cls.calls.append((model, kwargs))
        cls.loaded = cls()
        return cls.loaded

    def synthesize_custom_voice(
        self,
        text,
        *,
        speakers,
        languages,
        instructs,
        options,
    ):
        self.calls.append((text, speakers, languages, instructs, options))
        if isinstance(text, str):
            return _TTSOutput(torch.tensor([[1.0, 2.0]]), 16000)
        return [
            _TTSOutput(
                torch.tensor([[float(index * 2), float(index * 2 + 1)]]),
                16000,
            )
            for index, _ in enumerate(text)
        ]


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
            return [f"hello-{int(sample.reshape(-1)[0].item())}" for sample in waveform]
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
        modules["anytrain.tts"].EncodedAudioReference = _EncodedAudioReference
        modules["anytrain.tts"].WaveformReference = _WaveformReference
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


class _fake_anytrain_qwen_tts:
    def __init__(self) -> None:
        self.previous = {}

    def __enter__(self):
        modules = {
            "anytrain": types.ModuleType("anytrain"),
            "anytrain.tts": types.ModuleType("anytrain.tts"),
            "anytrain.tts.qwen": types.ModuleType("anytrain.tts.qwen"),
        }
        modules["anytrain.tts.qwen"].QwenCustomVoiceTTS = FakeQwenCustomVoiceTTS
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
