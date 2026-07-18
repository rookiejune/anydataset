from __future__ import annotations

import unittest

from anydataset.quality.translation import Label, Predicate, Profile
from anydataset.types import Modality, Role, TextItem, TextView


class TranslationQualityTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
