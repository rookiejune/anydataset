from __future__ import annotations

import unittest
from unittest import mock

from anydataset.quality.rules import QualityChain, QualityLabel, Rule
from anydataset.quality.text import (
    ChineseGEC,
    TextAcceptability,
    TextQuality,
    TextQualityProfile,
)
from anydataset.quality.translation import (
    Bicleaner,
    TranslationQuality,
    TranslationQualityProfile,
)
from anydataset.types import Lang, Modality, Role, TextItem, TextView


class TranslationQualityTest(unittest.TestCase):
    def test_profile_rejects_invalid_thresholds(self):
        cases = (
            ({"source_lang": "", "target_lang": Lang.EN}, TypeError, "Lang"),
            (
                {"source_lang": Lang.UND, "target_lang": Lang.EN},
                ValueError,
                "explicit",
            ),
            (
                {
                    "source_lang": Lang.EN,
                    "target_lang": Lang.FR,
                    "min_identical_script_chars": 0,
                },
                ValueError,
                "min_identical_script_chars",
            ),
            (
                {
                    "source_lang": Lang.EN,
                    "target_lang": Lang.FR,
                    "review_min_ratio": float("nan"),
                },
                ValueError,
                "finite",
            ),
            (
                {
                    "source_lang": Lang.EN,
                    "target_lang": Lang.FR,
                    "reject_min_ratio": 0.3,
                    "review_min_ratio": 0.2,
                },
                ValueError,
                "length ratios",
            ),
        )

        for kwargs, error, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(error, message):
                    TranslationQualityProfile(**kwargs)

    def test_bicleaner_rejects_invalid_contract(self):
        with self.assertRaisesRegex(TypeError, "scorer must be callable"):
            Bicleaner(None, Lang.EN, Lang.FR)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            Bicleaner(lambda _source, _target: 0.5, Lang.EN, Lang.FR, min_score=1.1)

        predicate = Bicleaner(lambda _source, _target: float("nan"), Lang.EN, Lang.FR)
        with self.assertRaisesRegex(ValueError, "scorer output must be finite"):
            predicate(_text_pair("hello", "bonjour"))

    def test_pair_accepts_clean_pair(self):
        predicate = TranslationQuality(
            TranslationQualityProfile(source_lang=Lang.EN, target_lang=Lang.FR)
        )

        decision = predicate(_text_pair("hello world", "bonjour le monde"))

        self.assertEqual(decision.label, QualityLabel.ACCEPT)
        self.assertEqual(decision.metrics["decision"], "accept")
        self.assertEqual(decision.metrics["flags"], [])

    def test_pair_accepts_number_surface_mismatch_with_flag(self):
        predicate = TranslationQuality(
            TranslationQualityProfile(source_lang=Lang.EN, target_lang=Lang.FR)
        )

        decision = predicate(_text_pair("version 6.0 is ready", "version 6 is ready"))

        self.assertEqual(decision.label, QualityLabel.ACCEPT)
        self.assertIn("number_surface_mismatch", decision.metrics["flags"])

    def test_pair_rejects_short_target(self):
        predicate = TranslationQuality(
            TranslationQualityProfile(
                source_lang=Lang.EN,
                target_lang=Lang.FR,
                review_min_ratio=0.8,
            )
        )

        decision = predicate(_text_pair("hello world again", "bonjour"))

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertIn("target_short", decision.metrics["flags"])

    def test_text_rule_reports_per_role_reasons(self):
        predicate = TextQuality(
            role=Role.SOURCE,
            lang=Lang.EN,
            profile=TextQualityProfile(min_script_ratio=0.9, reject_script_ratio=0.5),
        )

        decision = predicate(_text_pair("你好 world", "bonjour"))

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertIn("source_script_low", decision.metrics["flags"])
        self.assertEqual(decision.metrics["items"][0]["role"], "source")

    def test_acceptability_rule_scores_single_role_language(self):
        classifier = _Classifier(
            [
                {"label": "LABEL_0", "score": 0.8},
                {"label": "LABEL_1", "score": 0.2},
            ]
        )
        predicate = TextAcceptability(role=Role.TARGET, lang=Lang.EN)

        with mock.patch(
            "anydataset.quality.text._classifier",
            return_value=classifier,
        ):
            decision = predicate(_text_pair("hello", "bonjour"))

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertIn("target_acceptability_low", decision.metrics["flags"])
        self.assertEqual(
            classifier.calls,
            [("bonjour", {"truncation": True, "top_k": None})],
        )
        self.assertEqual(
            decision.metrics["items"][0]["acceptability_model"],
            "textattack/roberta-base-CoLA",
        )

    def test_acceptability_rule_requires_default_language_model(self):
        with self.assertRaisesRegex(ValueError, "no default model"):
            TextAcceptability(role=Role.TARGET, lang=Lang.FR)

    def test_chinese_gec_rejects_large_correction(self):
        corrector = _Classifier([{"generated_text": "今天心情很好"}])
        predicate = ChineseGEC(role=Role.TARGET)

        with mock.patch(
            "anydataset.quality.text._corrector",
            return_value=corrector,
        ):
            decision = predicate(_text_pair("hello", "今天新情很好"))

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertIn("target_gec_edit_ratio_high", decision.metrics["flags"])
        self.assertEqual(
            corrector.calls,
            [("今天新情很好", {"truncation": True})],
        )
        item = decision.metrics["items"][0]
        self.assertEqual(item["corrected_text"], "今天心情很好")
        self.assertEqual(
            item["gec_model"],
            "shibing624/mengzi-t5-base-chinese-correction",
        )
        self.assertEqual(item["gec_edit_count"], 1)
        self.assertEqual(item["gec_edit_ratio"], 0.166667)

    def test_chinese_gec_accepts_unchanged_text(self):
        corrector = _Classifier([{"generated_text": "今天天气很好"}])
        predicate = ChineseGEC(role=Role.TARGET, max_edit_ratio=0.0)

        with mock.patch(
            "anydataset.quality.text._corrector",
            return_value=corrector,
        ):
            decision = predicate(_text_pair("hello", "今天天气很好"))

        self.assertEqual(decision.label, QualityLabel.ACCEPT)
        self.assertEqual(decision.metrics["flags"], [])
        self.assertEqual(decision.metrics["items"][0]["gec_edit_count"], 0)

    def test_chain_accept_lifts_previous_reject_to_review(self):
        predicate = QualityChain(
            (
                Rule(
                    "text",
                    TextQuality(
                        role=Role.SOURCE,
                        lang=Lang.EN,
                        profile=TextQualityProfile(
                            min_script_ratio=0.9,
                            reject_script_ratio=0.5,
                        ),
                    ),
                ),
                Rule(
                    "pair",
                    TranslationQuality(
                        TranslationQualityProfile(
                            source_lang=Lang.EN,
                            target_lang=Lang.FR,
                        )
                    ),
                ),
            )
        )

        decision = predicate(_text_pair("你好 world", "bonjour monde"))

        self.assertEqual(decision.label, QualityLabel.REVIEW)
        self.assertEqual(decision.metrics["rules"][0]["label"], "reject")
        self.assertEqual(decision.metrics["rules"][1]["label"], "accept")
        self.assertIn("text:source_script_low", decision.metrics["flags"])
        self.assertEqual(
            decision.metrics["transitions"],
            [
                {"rule": "text", "from": "accept", "to": "reject"},
                {"rule": "pair", "from": "reject", "to": "review"},
            ],
        )

    def test_chain_reject_overrides_review(self):
        predicate = QualityChain(
            (
                Rule(
                    "text",
                    TextQuality(
                        role=Role.SOURCE,
                        lang=Lang.EN,
                        profile=TextQualityProfile(
                            min_script_ratio=0.9,
                            reject_script_ratio=0.5,
                        ),
                    ),
                ),
                Rule(
                    "pair",
                    TranslationQuality(
                        TranslationQualityProfile(
                            source_lang=Lang.EN,
                            target_lang=Lang.FR,
                        )
                    ),
                ),
                Rule("model", Bicleaner(lambda _source, _target: 0.1, Lang.EN, Lang.FR)),
            )
        )

        decision = predicate(_text_pair("你好 world", "bonjour monde"))

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertEqual(decision.metrics["rules"][2]["label"], "reject")
        self.assertIn("model:bicleaner_reject", decision.metrics["flags"])
        self.assertEqual(decision.metrics["transitions"][-1]["to"], "reject")

    def test_rejects_html_tag_mismatch(self):
        predicate = TranslationQuality(
            TranslationQualityProfile(source_lang=Lang.EN, target_lang=Lang.FR)
        )

        decision = predicate(
            {
                (Role.SOURCE, Modality.TEXT): TextItem(
                    views={TextView.TEXT: "Read <b>this text</b>."}
                ),
                (Role.TARGET, Modality.TEXT): TextItem(
                    views={TextView.TEXT: "Lisez ce texte."}
                ),
            }
        )

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertIn("html_tag_mismatch", decision.metrics["flags"])


def _text_pair(source: str, target: str):
    return {
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: source},
        ),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: target},
        ),
    }


class _Classifier:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, text: str, **kwargs):
        self.calls.append((text, kwargs))
        return self.output


if __name__ == "__main__":
    unittest.main()
