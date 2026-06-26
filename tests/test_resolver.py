import unittest

from anydataset import Preset, Source, Spec, resolve_dataset


class ResolverTest(unittest.TestCase):
    def test_preset_uses_auto_str_value(self):
        self.assertEqual(Preset.FLEURS.value, "fleurs")
        self.assertEqual(Preset.LIBRISPEECH_ASR.value, "librispeech_asr")
        self.assertEqual(Preset.FSD50K.value, "fsd50k")

    def test_preset_spec_resolves_builtin(self):
        spec = Preset.FLEURS.spec(split="validation")

        self.assertEqual(spec.source, Source.HF)
        self.assertEqual(spec.path, "google/fleurs")
        self.assertEqual(spec.split, "validation")
        self.assertEqual(spec.load_options["config_name"], "en_us")
        self.assertEqual(spec.load_options["streaming"], True)

    def test_resolve_dataset_accepts_string_preset_and_spec(self):
        spec = resolve_dataset("mnist:train")

        self.assertEqual(spec.source, Source.HF)
        self.assertEqual(spec.path, "ylecun/mnist")
        self.assertEqual(spec.split, "train")

        preset = resolve_dataset(Preset.FSD50K)

        self.assertEqual(preset.source, Source.HF)
        self.assertEqual(preset.path, "Fhrozen/FSD50k")
        self.assertEqual(preset.split, "dev")

        explicit = Spec(source=Source.LOCAL, path="/tmp/data.jsonl")

        self.assertIs(resolve_dataset(explicit), explicit)

    def test_resolves_explicit_source_shorthands(self):
        cases = {
            "hf://org/name:train": (Source.HF, "org/name", "train"),
            "hf-disk:///tmp/saved_dataset:validation": (
                Source.HF_DISK,
                "/tmp/saved_dataset",
                "validation",
            ),
            "local:///tmp/custom.jsonl:validation": (
                Source.LOCAL,
                "/tmp/custom.jsonl",
                "validation",
            ),
            "unified:///tmp/unified_audio:train": (
                Source.UNIFIED,
                "/tmp/unified_audio",
                "train",
            ),
        }

        for shorthand, expected in cases.items():
            with self.subTest(shorthand=shorthand):
                spec = resolve_dataset(shorthand)
                self.assertEqual((spec.source, spec.path, spec.split), expected)

    def test_builtin_audio_dataset_split_can_be_overridden(self):
        spec = resolve_dataset("fleurs:validation")

        self.assertEqual(spec.split, "validation")
        self.assertEqual(spec.load_options["config_name"], "en_us")

    def test_unknown_dataset_requires_explicit_source(self):
        with self.assertRaises(KeyError):
            resolve_dataset("missing:train")


if __name__ == "__main__":
    unittest.main()
