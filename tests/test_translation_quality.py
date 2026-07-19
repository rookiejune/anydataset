from __future__ import annotations

import unittest

from anydataset.quality.translation import Bicleaner, Label, Predicate, Profile
from anydataset.types import Modality, Role, TextItem, TextView


class TranslationQualityTest(unittest.TestCase):
    def test_profile_rejects_invalid_thresholds(self):
        cases = (
            ({"source_lang": "", "target_lang": "en"}, ValueError, "language"),
            ({"source_lang": None, "target_lang": "en"}, TypeError, "language"),
            (
                {"source_lang": "en", "target_lang": "fr", "min_chars": 0},
                ValueError,
                "min_chars",
            ),
            (
                {
                    "source_lang": "en",
                    "target_lang": "fr",
                    "review_min_ratio": float("nan"),
                },
                ValueError,
                "finite",
            ),
            (
                {
                    "source_lang": "en",
                    "target_lang": "fr",
                    "reject_min_ratio": 0.3,
                    "review_min_ratio": 0.2,
                },
                ValueError,
                "length ratios",
            ),
            (
                {
                    "source_lang": "en",
                    "target_lang": "fr",
                    "reject_script_ratio": 0.8,
                    "min_script_ratio": 0.5,
                },
                ValueError,
                "reject_script_ratio",
            ),
            (
                {
                    "source_lang": "en",
                    "target_lang": "fr",
                    "max_control_ratio": 1.1,
                },
                ValueError,
                "between 0 and 1",
            ),
        )

        for kwargs, error, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(error, message):
                    Profile(**kwargs)

    def test_bicleaner_rejects_invalid_contract(self):
        with self.assertRaisesRegex(TypeError, "scorer must be callable"):
            Bicleaner(None)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            Bicleaner(lambda _source, _target: 0.5, high_score=1.1)

        callback = Bicleaner(lambda _source, _target: float("nan"))
        predicate = Predicate(
            Profile(source_lang="en", target_lang="fr"),
            callbacks=(callback,),
        )
        with self.assertRaisesRegex(ValueError, "scorer output must be finite"):
            predicate(_text_pair("hello", "bonjour"))

    def test_rejects_html_tag_mismatch(self):
        predicate = Predicate(Profile(source_lang="en", target_lang="fr"))

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

        self.assertEqual(decision.label, Label.REJECT)
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


if __name__ == "__main__":
    unittest.main()
