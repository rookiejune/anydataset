import unittest

from anydataset import (
    Preset,
    Source,
    Spec,
    resolve_dataset,
)


class ResolverTest(unittest.TestCase):
    def test_preset_uses_auto_str_value(self):
        self.assertEqual(Preset.FLEURS.value, "fleurs")
        self.assertEqual(Preset.LIBRISPEECH_ASR.value, "librispeech_asr")
        self.assertEqual(Preset.FSD50K.value, "fsd50k")

    def test_source_uses_auto_str_value(self):
        self.assertEqual(Source.HF.value, "hf")
        self.assertEqual(Source.HF_DISK.value, "hf-disk")
        self.assertEqual(Source.STORE.value, "store")

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
        self.assertEqual(preset.load_options["revision"], "main")

        explicit = Spec(source=Source.STORE, path="/tmp/data")

        self.assertIs(resolve_dataset(explicit), explicit)

    def test_resolves_explicit_source_shorthands(self):
        cases = {
            "hf://org/name:train": (Source.HF, "org/name", "train"),
            "hf-disk:///tmp/saved_dataset:validation": (
                Source.HF_DISK,
                "/tmp/saved_dataset",
                "validation",
            ),
            "store:///tmp/store_audio:train": (
                Source.STORE,
                "/tmp/store_audio",
                "train",
            ),
        }

        for shorthand, expected in cases.items():
            with self.subTest(shorthand=shorthand):
                spec = resolve_dataset(shorthand)
                self.assertEqual((spec.source, spec.path, spec.split), expected)

    def test_resolves_huggingface_split_slices(self):
        cases = {
            "hf://org/name:train[:10%]": "train[:10%]",
            "hf://org/name:train[10%:20%]": "train[10%:20%]",
        }

        for shorthand, split in cases.items():
            with self.subTest(shorthand=shorthand):
                spec = resolve_dataset(shorthand)
                self.assertEqual(spec.path, "org/name")
                self.assertEqual(spec.split, split)

    def test_split_parser_keeps_colons_in_source_paths(self):
        spec = resolve_dataset("store://C:/datasets/audio:train")

        self.assertEqual(spec.path, "C:/datasets/audio")
        self.assertEqual(spec.split, "train")

    def test_split_parser_keeps_windows_drive_path_without_split(self):
        spec = resolve_dataset("store://C:/datasets/audio")

        self.assertEqual(spec.path, "C:/datasets/audio")
        self.assertIsNone(spec.split)

    def test_empty_explicit_splits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Spec.split"):
            Preset.MNIST.spec(split="")
        with self.assertRaisesRegex(ValueError, "Spec.split"):
            resolve_dataset("hf://org/data:")

    def test_builtin_audio_dataset_split_can_be_overridden(self):
        spec = resolve_dataset("fleurs:validation")

        self.assertEqual(spec.split, "validation")
        self.assertEqual(spec.load_options["config_name"], "en_us")

    def test_unknown_dataset_requires_explicit_source(self):
        with self.assertRaises(KeyError):
            resolve_dataset("missing:train")

    def test_unregistered_source_shorthand_is_not_parsed_as_preset(self):
        for shorthand in ("missing:///tmp/data", "mnist://anything"):
            with self.subTest(shorthand=shorthand):
                with self.assertRaisesRegex(KeyError, "Unknown dataset source"):
                    resolve_dataset(shorthand)


if __name__ == "__main__":
    unittest.main()
