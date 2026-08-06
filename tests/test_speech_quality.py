from __future__ import annotations

import unittest

import torch

from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextView,
)
from anydataset.quality.rules import QualityLabel
from anydataset.quality.speech import SpeechQuality, SpeechQualityProfile


class FakeSpeechEvaluator:
    def __init__(self, metrics):
        self.metrics = list(metrics)
        self.calls = []

    def __call__(self, audio, sample_rate, reference_text, **decode_options):
        self.calls.append((audio, sample_rate, reference_text, decode_options))
        index = len(self.calls) - 1
        return dict(self.metrics[index])


class FakeFrameCodec:
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def decode(self, codes):
        self.calls.append(codes.clone())
        values = codes[:, 0, 0].float().div(10.0)
        return values[:, None, None].expand(-1, 1, 4).clone()


class FakeCodecProvider:
    def __init__(self, codec, output=AudioView.LONGCAT):
        self.codec = codec
        self.output = output


class SpeechQualityTest(unittest.TestCase):
    def test_accepts_every_checked_audio_when_metrics_pass(self):
        source_wave = torch.full((1, 16000), 0.1)
        target_wave = torch.ones(1, 8000)
        evaluator = FakeSpeechEvaluator(
            [
                {"utmos": 3.5, "wer": 0.1, "chrf": 80.0, "bleu": 70.0},
                {"utmos": 4.0, "wer": 0.2, "chrf": 75.0, "bleu": 60.0},
            ]
        )
        predicate = SpeechQuality(
            evaluator=evaluator,
            decode_options={"language": "en", "temperature": 0.0},
        )

        decision = predicate(
            {
                (Role.SOURCE, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: (source_wave, 16000)},
                ),
                (Role.SOURCE, Modality.TEXT): TextItem(
                    views={TextView.TEXT: " hello   source "},
                ),
                (Role.TARGET, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: (target_wave, 8000)},
                ),
                (Role.TARGET, Modality.TEXT): TextItem(
                    views={TextView.TEXT: "hello target"},
                ),
            }
        )

        self.assertEqual(decision.label, QualityLabel.ACCEPT)
        self.assertEqual(decision.metrics["decision"], "accept")
        self.assertEqual(decision.metrics["flags"], [])
        self.assertEqual(decision.metrics["warnings"], [])
        self.assertEqual(decision.metrics["audio_count"], 2)
        self.assertEqual(decision.metrics["checked_count"], 2)
        self.assertEqual(
            decision.metrics["items"],
            [
                {
                    "role": "source",
                    "reference_text": "hello source",
                    "utmos": 3.5,
                    "wer": 0.1,
                    "chrf": 80.0,
                    "bleu": 70.0,
                    "duration_seconds": 1.0,
                    "peak_amplitude": 0.1,
                    "text_units": 2,
                    "seconds_per_text_unit": 0.5,
                    "flags": [],
                },
                {
                    "role": "target",
                    "reference_text": "hello target",
                    "utmos": 4.0,
                    "wer": 0.2,
                    "chrf": 75.0,
                    "bleu": 60.0,
                    "duration_seconds": 1.0,
                    "peak_amplitude": 1.0,
                    "text_units": 2,
                    "seconds_per_text_unit": 0.5,
                    "flags": [],
                },
            ],
        )
        self.assertEqual(
            evaluator.calls,
            [
                (
                    source_wave,
                    16000,
                    "hello source",
                    {"language": "en", "temperature": 0.0},
                ),
                (
                    target_wave,
                    8000,
                    "hello target",
                    {"language": "en", "temperature": 0.0},
                ),
            ],
        )

    def test_rejects_when_any_checked_audio_fails_thresholds(self):
        predicate = SpeechQuality(
            evaluator=FakeSpeechEvaluator(
                [
                    {"utmos": 2.79, "wer": 0.41, "chrf": 49.9, "bleu": 70.0},
                ]
            )
        )

        decision = predicate(_sample(torch.zeros(1, 16000), 16000, "hello"))

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertEqual(
            decision.metrics["flags"],
            ["default_utmos_low", "default_chrf_low", "default_peak_amplitude_low"],
        )
        self.assertEqual(
            decision.metrics["items"][0]["flags"],
            ["utmos_low", "chrf_low", "peak_amplitude_low"],
        )

    def test_wer_rejection_is_only_enabled_when_threshold_is_set(self):
        predicate = SpeechQuality(
            profile=SpeechQualityProfile(max_wer=0.4),
            evaluator=FakeSpeechEvaluator(
                [
                    {"utmos": 4.0, "wer": 0.41, "chrf": 80.0, "bleu": 70.0},
                ]
            ),
        )

        decision = predicate(_sample(torch.ones(1, 16000), 16000, "hello"))

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertEqual(decision.metrics["flags"], ["default_wer_high"])
        self.assertEqual(decision.metrics["items"][0]["flags"], ["wer_high"])

    def test_rejects_below_bleu_when_threshold_is_enabled(self):
        predicate = SpeechQuality(
            profile=SpeechQualityProfile(min_bleu=30.0),
            evaluator=FakeSpeechEvaluator(
                [
                    {"utmos": 4.0, "wer": 0.1, "chrf": 80.0, "bleu": 29.9},
                ]
            ),
        )

        decision = predicate(_sample(torch.ones(1, 16000), 16000, "hello"))

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertEqual(decision.metrics["flags"], ["default_bleu_low"])

    def test_rejects_above_absolute_duration_when_threshold_is_enabled(self):
        predicate = SpeechQuality(
            profile=SpeechQualityProfile(
                max_duration_seconds=0.5,
                max_seconds_per_text_unit=None,
            ),
            evaluator=FakeSpeechEvaluator(
                [
                    {"utmos": 4.0, "wer": 0.1, "chrf": 80.0, "bleu": 70.0},
                ]
            ),
        )

        decision = predicate(_sample(torch.ones(1, 16000), 16000, "hello"))

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertEqual(decision.metrics["flags"], ["default_duration_high"])
        self.assertEqual(decision.metrics["items"][0]["flags"], ["duration_high"])

    def test_rejects_invalid_absolute_duration_threshold(self):
        for value, error in (
            (True, TypeError),
            ("1.0", TypeError),
            (float("inf"), ValueError),
            (float("nan"), ValueError),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(error, "max_duration_seconds"):
                    SpeechQualityProfile(max_duration_seconds=value)  # type: ignore[arg-type]

    def test_rejects_long_audio_per_text_unit_and_low_peak(self):
        predicate = SpeechQuality(
            evaluator=FakeSpeechEvaluator(
                [
                    {"utmos": 4.0, "wer": 0.1, "chrf": 80.0, "bleu": 70.0},
                ]
            )
        )

        decision = predicate(_sample(torch.zeros(1, 80000), 16000, "啊"))

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertEqual(
            decision.metrics["flags"],
            [
                "default_duration_per_text_unit_high",
                "default_peak_amplitude_low",
            ],
        )
        self.assertEqual(
            decision.metrics["items"][0],
            {
                "role": "default",
                "reference_text": "啊",
                "utmos": 4.0,
                "wer": 0.1,
                "chrf": 80.0,
                "bleu": 70.0,
                "duration_seconds": 5.0,
                "peak_amplitude": 0.0,
                "text_units": 1,
                "seconds_per_text_unit": 5.0,
                "flags": ["duration_per_text_unit_high", "peak_amplitude_low"],
            },
        )

    def test_counts_cjk_characters_and_latin_words_as_text_units(self):
        predicate = SpeechQuality(
            evaluator=FakeSpeechEvaluator(
                [
                    {"utmos": 4.0, "wer": 0.1, "chrf": 80.0, "bleu": 70.0},
                ]
            )
        )

        decision = predicate(_sample(torch.ones(1, 16000), 16000, "你好 ABC 123"))

        self.assertEqual(decision.label, QualityLabel.ACCEPT)
        self.assertEqual(decision.metrics["items"][0]["text_units"], 4)
        self.assertEqual(decision.metrics["items"][0]["seconds_per_text_unit"], 0.25)

    def test_codec_provider_decodes_configured_view_instead_of_waveform(self):
        codec = FakeFrameCodec()
        evaluator = FakeSpeechEvaluator(
            [{"utmos": 4.0, "wer": 0.1, "chrf": 80.0, "bleu": 70.0}]
        )
        predicate = SpeechQuality(
            evaluator=evaluator,
            codec_provider=FakeCodecProvider(codec),
        )
        original = torch.full((1, 8), 0.9)
        sample = _codec_sample(7, frames=2, text="hello", waveform=original)

        decision = predicate(sample)

        self.assertEqual(decision.label, QualityLabel.ACCEPT)
        self.assertEqual(len(codec.calls), 1)
        self.assertEqual(tuple(codec.calls[0].shape), (1, 2, 1))
        self.assertTrue(
            torch.equal(
                codec.calls[0][0],
                sample[Role.DEFAULT, Modality.AUDIO].views[AudioView.LONGCAT],
            )
        )
        audio, sample_rate, reference_text, _options = evaluator.calls[0]
        self.assertTrue(torch.equal(audio, torch.full((1, 4), 0.7)))
        self.assertFalse(torch.equal(audio, original))
        self.assertEqual(sample_rate, 24000)
        self.assertEqual(reference_text, "hello")

    def test_codec_batch_groups_equal_lengths_and_restores_sample_order(self):
        codec = FakeFrameCodec()
        evaluator = FakeSpeechEvaluator(
            [
                {"utmos": 4.0, "wer": 0.1, "chrf": 80.0, "bleu": 70.0},
                {"utmos": 4.0, "wer": 0.1, "chrf": 80.0, "bleu": 70.0},
                {"utmos": 4.0, "wer": 0.1, "chrf": 80.0, "bleu": 70.0},
            ]
        )
        predicate = SpeechQuality(
            evaluator=evaluator,
            codec_provider=FakeCodecProvider(codec),
        )

        decisions = predicate.call_batch(
            (
                _codec_sample(1, frames=2, text="first"),
                _codec_sample(2, frames=3, text="second"),
                _codec_sample(3, frames=2, text="third"),
            )
        )

        self.assertEqual(
            [decision.label for decision in decisions],
            [QualityLabel.ACCEPT, QualityLabel.ACCEPT, QualityLabel.ACCEPT],
        )
        self.assertEqual(
            [tuple(codes.shape) for codes in codec.calls],
            [(2, 2, 1), (1, 3, 1)],
        )
        self.assertEqual(
            [call[2] for call in evaluator.calls],
            ["first", "second", "third"],
        )
        torch.testing.assert_close(
            torch.tensor([call[0][0, 0] for call in evaluator.calls]),
            torch.tensor([0.1, 0.2, 0.3]),
        )

    def test_codec_provider_does_not_fall_back_when_codec_view_is_missing(self):
        predicate = SpeechQuality(
            evaluator=FakeSpeechEvaluator([]),
            codec_provider=FakeCodecProvider(FakeFrameCodec()),
        )

        with self.assertRaisesRegex(ValueError, "expects audio view 'longcat'"):
            predicate(_sample(torch.ones(1, 4), 16000, "hello"))

    def test_skips_audio_without_waveform_and_records_warning(self):
        evaluator = FakeSpeechEvaluator([])
        predicate = SpeechQuality(evaluator=evaluator)
        decision = predicate(
            {
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.FILE: "speech.flac"},
                ),
                (Role.DEFAULT, Modality.TEXT): TextItem(
                    views={TextView.TEXT: "hello"},
                ),
            }
        )

        self.assertEqual(decision.label, QualityLabel.ACCEPT)
        self.assertEqual(decision.metrics["flags"], [])
        self.assertEqual(decision.metrics["warnings"], ["default_missing_waveform"])
        self.assertEqual(decision.metrics["audio_count"], 1)
        self.assertEqual(decision.metrics["checked_count"], 0)
        self.assertEqual(decision.metrics["items"], [])
        self.assertEqual(evaluator.calls, [])

    def test_skips_audio_without_same_role_text_and_records_warning(self):
        evaluator = FakeSpeechEvaluator([])
        predicate = SpeechQuality(evaluator=evaluator)
        decision = predicate(
            {
                (Role.SOURCE, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: (torch.zeros(1, 16000), 16000)},
                ),
                (Role.TARGET, Modality.TEXT): TextItem(
                    views={TextView.TEXT: "wrong role"},
                ),
            }
        )

        self.assertEqual(decision.label, QualityLabel.ACCEPT)
        self.assertEqual(decision.metrics["warnings"], ["source_missing_text"])
        self.assertEqual(decision.metrics["checked_count"], 0)
        self.assertEqual(evaluator.calls, [])

    def test_rejects_non_finite_waveform_before_evaluation(self):
        evaluator = FakeSpeechEvaluator([])
        predicate = SpeechQuality(evaluator=evaluator)

        decision = predicate(
            _sample(torch.tensor([[0.0, float("nan")]]), 16000, "hello")
        )

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertEqual(decision.metrics["flags"], ["default_non_finite_waveform"])
        self.assertEqual(decision.metrics["checked_count"], 0)
        self.assertEqual(evaluator.calls, [])

    def test_accepts_sample_without_audio_and_records_warning(self):
        evaluator = FakeSpeechEvaluator([])
        predicate = SpeechQuality(evaluator=evaluator)

        decision = predicate({})

        self.assertEqual(decision.label, QualityLabel.ACCEPT)
        self.assertEqual(decision.metrics["flags"], [])
        self.assertEqual(decision.metrics["warnings"], ["no_audio"])
        self.assertEqual(decision.metrics["audio_count"], 0)
        self.assertEqual(decision.metrics["checked_count"], 0)
        self.assertEqual(evaluator.calls, [])

    def test_requires_required_speech_metrics(self):
        predicate = SpeechQuality(
            evaluator=FakeSpeechEvaluator(
                [
                    {"utmos": 4.0, "wer": 0.1, "chrf": 80.0},
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "bleu"):
            predicate(_sample(torch.zeros(1, 16000), 16000, "hello"))

    def test_rejects_non_scalar_tensor_metric(self):
        predicate = SpeechQuality(
            evaluator=FakeSpeechEvaluator(
                [
                    {
                        "utmos": torch.tensor([4.0]),
                        "wer": 0.1,
                        "chrf": 80.0,
                        "bleu": 70.0,
                    },
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "0-d tensor"):
            predicate(_sample(torch.zeros(1, 16000), 16000, "hello"))


def _sample(waveform, sample_rate: int, text: str):
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (waveform, sample_rate)},
        ),
        (Role.DEFAULT, Modality.TEXT): TextItem(
            views={TextView.TEXT: text},
        ),
    }


def _codec_sample(
    value: int,
    *,
    frames: int,
    text: str,
    waveform=None,
):
    views = {
        AudioView.LONGCAT: torch.full((frames, 1), value, dtype=torch.int64),
    }
    if waveform is not None:
        views[AudioView.WAVEFORM] = (waveform, 16000)
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(views=views),
        (Role.DEFAULT, Modality.TEXT): TextItem(
            views={TextView.TEXT: text},
        ),
    }


if __name__ == "__main__":
    unittest.main()
