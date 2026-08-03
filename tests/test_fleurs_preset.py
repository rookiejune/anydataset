from __future__ import annotations

import unittest
from typing import Any

from anydataset import Lang, Preset
from anydataset.presets.fleurs import Fleurs
from anydataset.types import Modality, Role, TextItem, TextMeta


class FleursPresetTest(unittest.TestCase):
    def test_default_config_drives_spec_and_sample_language(self):
        dataset = _dataset()

        self.assertEqual(dataset.spec.load_options["config_name"], "en_us")
        self.assertEqual(_sample_lang(dataset), Lang.EN)

    def test_explicit_config_drives_spec_and_sample_language(self):
        dataset = _dataset(config_name="fr_fr")
        spec = Preset.FLEURS.spec(config_name="fr_fr")

        self.assertEqual(dataset.spec.load_options["config_name"], "fr_fr")
        self.assertEqual(spec.load_options["config_name"], "fr_fr")
        self.assertEqual(_sample_lang(dataset), Lang.FR)

    def test_rejects_invalid_config_before_prepare(self):
        cases: tuple[tuple[object, type[Exception]], ...] = (
            (None, TypeError),
            (1, TypeError),
            (Lang.EN, TypeError),
            ("", ValueError),
            (" ", ValueError),
            (" en_us", ValueError),
            ("en_us ", ValueError),
        )

        for config_name, error in cases:
            with self.subTest(config_name=config_name, entry="dataset"):
                with self.assertRaisesRegex(error, "FLEURS config_name"):
                    Fleurs(config_name=config_name)
            with self.subTest(config_name=config_name, entry="spec"):
                with self.assertRaisesRegex(error, "FLEURS config_name"):
                    Preset.FLEURS.spec(config_name=config_name)


def _dataset(**load_options: Any) -> Fleurs:
    dataset = Fleurs(**load_options)
    dataset._dataset = [
        {
            "audio": {
                "array": [0.0],
                "sampling_rate": 16_000,
            },
            "transcription": "bonjour",
        }
    ]
    return dataset


def _sample_lang(dataset: Fleurs) -> Lang:
    text = dataset[0][Role.DEFAULT, Modality.TEXT]
    assert isinstance(text, TextItem)
    return text.meta[TextMeta.LANG]


if __name__ == "__main__":
    unittest.main()
