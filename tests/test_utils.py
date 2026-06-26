import unittest

from anydataset import (
    AudioKey,
    AudioOptKey,
    AudioView,
    ImageOptKey,
    ImageView,
    Modality,
    Role,
    TextOptKey,
    TextView,
)
from anydataset.utils import labels, sample_from_row, text_map


class PresetCommonTest(unittest.TestCase):
    def test_sample_from_row_maps_hf_audio_text_and_labels(self):
        row = {
            "audio": {
                "array": [0.1, 0.2],
                "sampling_rate": 16000,
            },
            "transcription": "hello",
            "category": "speech",
            "target": 3,
        }

        sample = sample_from_row(
            row,
            audio={
                "audio": AudioView.WAVEFORM,
                "category": AudioOptKey.LABEL,
                "target": labels("target"),
            },
            text={"transcription": TextView.TEXT},
            text_values={TextOptKey.LANG: "en"},
        )

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        text = sample[Role.DEFAULT, Modality.TEXT]
        self.assertEqual(audio.views[AudioView.WAVEFORM], [0.1, 0.2])
        self.assertEqual(audio.required[AudioKey.SAMPLE_RATE], 16000)
        self.assertEqual(audio.optional[AudioOptKey.LABEL], "speech")
        self.assertEqual(audio.optional[AudioOptKey.LABELS], {"target": 3})
        self.assertEqual(text.views[TextView.TEXT], "hello")
        self.assertEqual(text.optional[TextOptKey.LANG], "en")

    def test_sample_from_row_maps_image_classification_fields(self):
        sample = sample_from_row(
            {
                "image": [[1, 2], [3, 4]],
                "label": 7,
            },
            image={
                "image": ImageView.PIXEL,
                "label": ImageOptKey.LABEL,
            },
        )

        image = sample[Role.DEFAULT, Modality.IMAGE]
        self.assertEqual(image.views[ImageView.PIXEL], [[1, 2], [3, 4]])
        self.assertEqual(image.optional[ImageOptKey.LABEL], 7)

    def test_sample_from_row_requires_audio_sample_rate(self):
        with self.assertRaisesRegex(ValueError, "sample_rate"):
            sample_from_row(
                {"audio": [0.1, 0.2]},
                audio={"audio": AudioView.WAVEFORM},
            )

    def test_sample_from_row_maps_wmt19_style_translation_roles(self):
        sample = sample_from_row(
            {
                "translation": {
                    "en": "The tea is hot.",
                    "de": "Der Tee ist heiss.",
                }
            },
            items={
                (Role.SOURCE, Modality.TEXT): text_map(
                    {("translation", "en"): TextView.TEXT},
                    values={TextOptKey.LANG: "en"},
                ),
                (Role.TARGET, Modality.TEXT): text_map(
                    {("translation", "de"): TextView.TEXT},
                    values={TextOptKey.LANG: "de"},
                ),
            },
        )

        source = sample[Role.SOURCE, Modality.TEXT]
        target = sample[Role.TARGET, Modality.TEXT]
        self.assertEqual(source.views[TextView.TEXT], "The tea is hot.")
        self.assertEqual(source.optional[TextOptKey.LANG], "en")
        self.assertEqual(target.views[TextView.TEXT], "Der Tee ist heiss.")
        self.assertEqual(target.optional[TextOptKey.LANG], "de")

    def test_sample_from_row_rejects_duplicate_references(self):
        with self.assertRaises(ValueError):
            sample_from_row(
                {"text": "hello"},
                items={
                    (Role.DEFAULT, Modality.TEXT): text_map(
                        {"text": TextView.TEXT},
                    )
                },
                text={"text": TextView.TEXT},
            )


if __name__ == "__main__":
    unittest.main()
