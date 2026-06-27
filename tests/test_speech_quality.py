from __future__ import annotations

import unittest

import torch

from anydataset import AudioItem, AudioView, Modality, Role, TextItem, TextView
from anydataset.quality.speech import Label, Predicate, Profile


class FakeSpeechEvaluator:
    def __init__(self, metrics):
        self.metrics = list(metrics)
        self.calls = []

    def __call__(self, audio, sample_rate, reference_text, **decode_options):
        self.calls.append((audio, sample_rate, reference_text, decode_options))
        index = len(self.calls) - 1
        return dict(self.metrics[index])


class SpeechQualityTest(unittest.TestCase):
    def test_accepts_every_checked_audio_when_metrics_pass(self):
        source_wave = torch.zeros(1, 16000)
        target_wave = torch.ones(1, 8000)
        evaluator = FakeSpeechEvaluator(
            [
                {"utmos": 3.5, "wer": 0.1, "chrf": 80.0, "bleu": 70.0},
                {"utmos": 4.0, "wer": 0.2, "chrf": 75.0, "bleu": 60.0},
            ]
        )
        predicate = Predicate(
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

        self.assertEqual(decision.label, Label.ACCEPT)
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
                    "flags": [],
                },
                {
                    "role": "target",
                    "reference_text": "hello target",
                    "utmos": 4.0,
                    "wer": 0.2,
                    "chrf": 75.0,
                    "bleu": 60.0,
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
        predicate = Predicate(
            evaluator=FakeSpeechEvaluator(
                [
                    {"utmos": 2.99, "wer": 0.41, "chrf": 49.9, "bleu": 70.0},
                ]
            )
        )

        decision = predicate(_sample(torch.zeros(1, 16000), 16000, "hello"))

        self.assertEqual(decision.label, Label.REJECT)
        self.assertEqual(
            decision.metrics["flags"],
            ["default_utmos_low", "default_wer_high", "default_chrf_low"],
        )
        self.assertEqual(
            decision.metrics["items"][0]["flags"],
            ["utmos_low", "wer_high", "chrf_low"],
        )

    def test_rejects_below_bleu_when_threshold_is_enabled(self):
        predicate = Predicate(
            profile=Profile(min_bleu=30.0),
            evaluator=FakeSpeechEvaluator(
                [
                    {"utmos": 4.0, "wer": 0.1, "chrf": 80.0, "bleu": 29.9},
                ]
            ),
        )

        decision = predicate(_sample(torch.zeros(1, 16000), 16000, "hello"))

        self.assertEqual(decision.label, Label.REJECT)
        self.assertEqual(decision.metrics["flags"], ["default_bleu_low"])

    def test_skips_audio_without_waveform_and_records_warning(self):
        evaluator = FakeSpeechEvaluator([])
        predicate = Predicate(evaluator=evaluator)
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

        self.assertEqual(decision.label, Label.ACCEPT)
        self.assertEqual(decision.metrics["flags"], [])
        self.assertEqual(decision.metrics["warnings"], ["default_missing_waveform"])
        self.assertEqual(decision.metrics["audio_count"], 1)
        self.assertEqual(decision.metrics["checked_count"], 0)
        self.assertEqual(decision.metrics["items"], [])
        self.assertEqual(evaluator.calls, [])

    def test_skips_audio_without_same_role_text_and_records_warning(self):
        evaluator = FakeSpeechEvaluator([])
        predicate = Predicate(evaluator=evaluator)
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

        self.assertEqual(decision.label, Label.ACCEPT)
        self.assertEqual(decision.metrics["warnings"], ["source_missing_text"])
        self.assertEqual(decision.metrics["checked_count"], 0)
        self.assertEqual(evaluator.calls, [])

    def test_accepts_sample_without_audio_and_records_warning(self):
        evaluator = FakeSpeechEvaluator([])
        predicate = Predicate(evaluator=evaluator)

        decision = predicate({})

        self.assertEqual(decision.label, Label.ACCEPT)
        self.assertEqual(decision.metrics["flags"], [])
        self.assertEqual(decision.metrics["warnings"], ["no_audio"])
        self.assertEqual(decision.metrics["audio_count"], 0)
        self.assertEqual(decision.metrics["checked_count"], 0)
        self.assertEqual(evaluator.calls, [])

    def test_requires_required_speech_metrics(self):
        predicate = Predicate(
            evaluator=FakeSpeechEvaluator(
                [
                    {"utmos": 4.0, "wer": 0.1, "chrf": 80.0},
                ]
            )
        )

        with self.assertRaisesRegex(ValueError, "bleu"):
            predicate(_sample(torch.zeros(1, 16000), 16000, "hello"))

    def test_rejects_non_scalar_tensor_metric(self):
        predicate = Predicate(
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


if __name__ == "__main__":
    unittest.main()
