from __future__ import annotations

import pickle
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

from anydataset.filter import FilterDecision
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
from anydataset.types import Lang, Modality, Preset, Role, TextItem, TextView


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

    def test_wmt19_translation_profile_accepts_both_zh_en_directions(self):
        for source_lang, target_lang in (
            (Lang.ZH, Lang.EN),
            (Lang.EN, Lang.ZH),
        ):
            with self.subTest(source_lang=source_lang, target_lang=target_lang):
                predicate = TranslationQuality.from_preset(
                    Preset.WMT19,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )

                self.assertEqual(predicate.profile.source_lang, source_lang)
                self.assertEqual(predicate.profile.target_lang, target_lang)

        with self.assertRaisesRegex(ValueError, "zh-en pair"):
            TranslationQuality.from_preset(
                Preset.WMT19,
                source_lang=Lang.EN,
                target_lang=Lang.FR,
            )

    def test_quality_configuration_is_immutable(self):
        text_profile = TextQualityProfile()
        translation_profile = TranslationQualityProfile(
            source_lang=Lang.EN,
            target_lang=Lang.FR,
        )
        text_quality = TextQuality(role=Role.SOURCE, lang=Lang.EN)
        acceptability = TextAcceptability(role=Role.TARGET, lang=Lang.EN)
        gec = ChineseGEC(role=Role.TARGET)
        bicleaner = Bicleaner(lambda _source, _target: 0.5, Lang.EN, Lang.FR)
        chain = QualityChain((Rule("text", text_quality),))

        cases = (
            (text_profile, "min_chars", 2),
            (translation_profile, "review_min_ratio", 0.3),
            (text_quality, "role", Role.TARGET),
            (acceptability, "min_score", 0.4),
            (gec, "max_edits", 2),
            (bicleaner, "min_score", 0.4),
            (chain, "rules", ()),
        )
        for config, name, value in cases:
            with self.subTest(config=type(config).__name__, field=name):
                with self.assertRaises(FrozenInstanceError):
                    setattr(config, name, value)

        self.assertIsInstance(chain.rules, tuple)
        self.assertEqual(pickle.loads(pickle.dumps(text_profile)), text_profile)
        self.assertEqual(hash(text_quality), hash(text_quality))

    def test_quality_configuration_restores_legacy_unsealed_pickle_state(self):
        current = TextQualityProfile()
        state = vars(current).copy()
        state.pop("_immutable_sealed")
        legacy = TextQualityProfile.__new__(TextQualityProfile)
        vars(legacy).update(state)

        restored = pickle.loads(pickle.dumps(legacy))

        self.assertEqual(restored, current)
        with self.assertRaises(FrozenInstanceError):
            restored.min_chars = 2
        invalid = TextQualityProfile.__new__(TextQualityProfile)
        with self.assertRaisesRegex(TypeError, "invalid immutable pickle state"):
            invalid.__setstate__(("not", "a", "dict"))

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

    def test_zh_en_numbers_accept_conservative_cross_lingual_equivalence(self):
        cases = (
            ("人口达到6.8亿。", "The population reached 680 million."),
            ("该国有16亿人口。", "The country has 1.6 billion people."),
            ("投资额为2.7亿元。", "The investment was 270 million yuan."),
            ("总额为1 000亿美元。", "The total was $100 billion."),
            ("全球经济产值为70万亿美元。", "The $70-trillion global economy."),
            ("流入超过1万多亿美元。", "More than $1 trillion flowed in."),
            ("市值为70 Trillion美元。", "The market value was $70 trillion."),
            ("统计期为2008—2009年。", "The period was 2008-2009."),
            ("统计期为2008至2009年。", "The period was 2008~2009."),
            ("这发生在20世纪90年代。", "This happened in the 1990s."),
            ("会议于2020年11月举行。", "The meeting was held in November 2020."),
            ("FOMC9月会议。", "The FOMC September meeting."),
            ("柏林－19年来一直如此。", "BERLIN – This has been true for 19 years."),
            ("会议于２０２０年１１月举行。", "The meeting was held in November 2020."),
        )

        for chinese, english in cases:
            for source, target, source_lang, target_lang in (
                (chinese, english, Lang.ZH, Lang.EN),
                (english, chinese, Lang.EN, Lang.ZH),
            ):
                with self.subTest(source=source, target=target):
                    predicate = TranslationQuality(
                        TranslationQualityProfile(
                            source_lang=source_lang,
                            target_lang=target_lang,
                        )
                    )
                    decision = predicate(_text_pair(source, target))

                    self.assertEqual(decision.label, QualityLabel.ACCEPT)
                    self.assertNotIn("complex_numbers", decision.metrics["flags"])
                    self.assertNotIn(
                        "number_value_mismatch",
                        decision.metrics["flags"],
                    )

    def test_zh_en_numbers_reject_real_value_mismatches(self):
        cases = (
            ("有500多万人。", "There were 500 million people."),
            ("价值1.5亿美元。", "It was worth $1.5 trillion."),
            ("价格为77美元。", "The price was $77 billion."),
            ("总额为700亿美元。", "The total was $700 billion."),
            ("共有37万人。", "There were 37 million people."),
            ("投资60亿美元。", "An investment of $60 billion."),
            ("2020年和2020年各举行一次。", "It was held once in 2020."),
        )

        for chinese, english in cases:
            for source, target, source_lang, target_lang in (
                (chinese, english, Lang.ZH, Lang.EN),
                (english, chinese, Lang.EN, Lang.ZH),
            ):
                with self.subTest(source=source, target=target):
                    predicate = TranslationQuality(
                        TranslationQualityProfile(
                            source_lang=source_lang,
                            target_lang=target_lang,
                        )
                    )
                    decision = predicate(_text_pair(source, target))

                    self.assertEqual(decision.label, QualityLabel.REJECT)
                    self.assertIn(
                        "number_value_mismatch",
                        decision.metrics["flags"],
                    )

    def test_zh_en_numbers_keep_unsupported_expressions_conservative(self):
        predicate = TranslationQuality(
            TranslationQualityProfile(source_lang=Lang.ZH, target_lang=Lang.EN)
        )
        cases = (
            ("会议于2020年5月举行。", "The meeting may happen in 2020."),
            ("会议于2020年5月举行。", "May happen in 2020."),
            ("资源为3兆美元。", "The resources totaled $3 trillion."),
            ("比例为¾。", "The ratio is 3/4."),
        )

        for source, target in cases:
            with self.subTest(source=source, target=target):
                decision = predicate(_text_pair(source, target))

                self.assertEqual(decision.label, QualityLabel.REJECT)
                self.assertIn("complex_numbers", decision.metrics["flags"])

    def test_zh_en_numbers_do_not_treat_modal_may_or_nfkc_fraction_as_numbers(self):
        predicate = TranslationQuality(
            TranslationQualityProfile(source_lang=Lang.ZH, target_lang=Lang.EN)
        )
        cases = (
            ("这也许有帮助。", "It may help."),
            (
                "这最终取决于判断¾这只是多种判断之一。",
                "It depends on a judgment – one of many judgments.",
            ),
        )

        for source, target in cases:
            with self.subTest(source=source, target=target):
                decision = predicate(_text_pair(source, target))

                self.assertEqual(decision.label, QualityLabel.ACCEPT)
                self.assertNotIn(
                    "number_value_mismatch",
                    decision.metrics["flags"],
                )

    def test_cross_lingual_number_equivalence_is_specific_to_zh_en(self):
        predicate = TranslationQuality(
            TranslationQualityProfile(source_lang=Lang.ZH, target_lang=Lang.FR)
        )

        decision = predicate(_text_pair("人口达到6.8亿。", "680 millions."))

        self.assertEqual(decision.label, QualityLabel.REJECT)
        self.assertIn("complex_numbers", decision.metrics["flags"])

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

    def test_acceptability_retries_only_unsupported_top_k(self):
        class LegacyClassifier(_Classifier):
            def __call__(self, text: str, **kwargs):
                self.calls.append((text, kwargs))
                if "top_k" in kwargs:
                    raise TypeError(
                        "__call__() got an unexpected keyword argument 'top_k'"
                    )
                return self.output

        classifier = LegacyClassifier(
            [
                {"label": "LABEL_0", "score": 0.2},
                {"label": "LABEL_1", "score": 0.8},
            ]
        )
        predicate = TextAcceptability(role=Role.TARGET, lang=Lang.EN)

        with mock.patch(
            "anydataset.quality.text._classifier",
            return_value=classifier,
        ):
            decision = predicate(_text_pair("hello", "bonjour"))

        self.assertEqual(decision.label, QualityLabel.ACCEPT)
        self.assertEqual(
            classifier.calls,
            [
                ("bonjour", {"truncation": True, "top_k": None}),
                ("bonjour", {"truncation": True, "return_all_scores": True}),
            ],
        )

    def test_acceptability_preserves_classifier_type_error(self):
        class BrokenClassifier:
            def __init__(self):
                self.calls = []

            def __call__(self, text: str, **kwargs):
                self.calls.append((text, kwargs))
                raise TypeError("classifier implementation failed")

        classifier = BrokenClassifier()
        predicate = TextAcceptability(role=Role.TARGET, lang=Lang.EN)

        with mock.patch(
            "anydataset.quality.text._classifier",
            return_value=classifier,
        ):
            with self.assertRaisesRegex(TypeError, "classifier implementation failed"):
                predicate(_text_pair("hello", "bonjour"))

        self.assertEqual(
            classifier.calls,
            [("bonjour", {"truncation": True, "top_k": None})],
        )

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

    def test_chain_dispatches_batch_rules_once_and_scalar_rules_per_sample(self):
        class BatchRule:
            def __init__(self):
                self.calls = []

            def __call__(self, _sample):
                raise AssertionError("batch rule should use call_batch()")

            def call_batch(self, samples):
                self.calls.append(samples)
                return (QualityLabel.REJECT, QualityLabel.ACCEPT)

        class ScalarRule:
            def __init__(self):
                self.calls = []

            def __call__(self, sample):
                self.calls.append(sample)
                return QualityLabel.ACCEPT

        samples = (
            _text_pair("first", "premier"),
            _text_pair("second", "deuxieme"),
        )
        batch_rule = BatchRule()
        scalar_rule = ScalarRule()
        predicate = QualityChain(
            (Rule("batch", batch_rule), Rule("scalar", scalar_rule))
        )

        decisions = predicate.call_batch(samples)

        self.assertEqual(len(batch_rule.calls), 1)
        self.assertEqual(batch_rule.calls[0], samples)
        self.assertEqual(scalar_rule.calls, list(samples))
        self.assertEqual(
            [decision.label for decision in decisions],
            [QualityLabel.REVIEW, QualityLabel.ACCEPT],
        )

    def test_chain_rejects_wrong_child_batch_size(self):
        class WrongSizeRule:
            def __call__(self, _sample):
                raise AssertionError("batch rule should use call_batch()")

            def call_batch(self, _samples):
                return (QualityLabel.ACCEPT,)

        predicate = QualityChain((Rule("wrong", WrongSizeRule()),))

        with self.assertRaisesRegex(ValueError, "one output per input sample"):
            predicate.call_batch(
                (
                    _text_pair("first", "premier"),
                    _text_pair("second", "deuxieme"),
                )
            )

    def test_chain_rejects_invalid_rule_flags(self):
        for flags in ("not-a-list", ["valid", 1]):
            with self.subTest(flags=flags):
                predicate = QualityChain(
                    (
                        Rule(
                            "invalid",
                            lambda _sample, flags=flags: FilterDecision(
                                label=True,
                                metrics={"flags": flags},
                            ),
                        ),
                    )
                )

                with self.assertRaisesRegex(TypeError, "flags.*list of strings"):
                    predicate(_text_pair("hello", "bonjour"))

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
