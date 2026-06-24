import unittest

from anydataset.api.resolver import DatasetResolver, resolve_dataset_spec, resolve_dataset_specs
from anydataset.api.spec import DatasetSpec


class DatasetResolverTest(unittest.TestCase):
    def test_resolve_dataset_spec_accepts_string_and_spec(self):
        spec = resolve_dataset_spec("mnist:train")

        self.assertEqual(spec.source, "huggingface")
        self.assertEqual(spec.path, "ylecun/mnist")
        self.assertEqual(spec.name, "mnist")
        self.assertEqual(spec.split, "train")

        explicit = DatasetSpec(source="local_files", path="/tmp/data.jsonl", name="custom")

        self.assertIs(resolve_dataset_spec(explicit), explicit)

    def test_resolve_dataset_specs_treats_single_string_as_one_ref(self):
        specs = resolve_dataset_specs("mnist:train")

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].key, "mnist:train")

    def test_resolves_default_dataset_with_split(self):
        spec = DatasetResolver().resolve("mnist:train")

        self.assertEqual(spec.source, "huggingface")
        self.assertEqual(spec.path, "ylecun/mnist")
        self.assertEqual(spec.name, "mnist")
        self.assertEqual(spec.split, "train")
        self.assertEqual(spec.key, "mnist:train")

    def test_resolves_custom_dataset_map(self):
        resolver = DatasetResolver(
            {
                "custom": DatasetSpec(
                    source="local_files",
                    path="/tmp/custom.jsonl",
                    name="custom",
                )
            }
        )

        spec = resolver.resolve("custom:validation")

        self.assertEqual(spec.source, "local_files")
        self.assertEqual(spec.path, "/tmp/custom.jsonl")
        self.assertEqual(spec.name, "custom")
        self.assertEqual(spec.split, "validation")

    def test_rejects_dataset_map_key_that_does_not_match_spec_name(self):
        with self.assertRaises(ValueError):
            DatasetResolver(
                {
                    "custom": DatasetSpec(
                        source="local_files",
                        path="/tmp/custom.jsonl",
                        name="other",
                    )
                }
            )

    def test_resolves_explicit_huggingface_reference(self):
        spec = DatasetResolver().resolve("hf://org/name:train")

        self.assertEqual(spec.source, "huggingface")
        self.assertEqual(spec.path, "org/name")
        self.assertEqual(spec.name, "org/name")
        self.assertEqual(spec.split, "train")
        self.assertEqual(spec.load_options, {})

    def test_resolves_explicit_local_reference_with_path_as_name(self):
        spec = DatasetResolver().resolve("local:///tmp/custom.jsonl:validation")

        self.assertEqual(spec.source, "local_files")
        self.assertEqual(spec.path, "/tmp/custom.jsonl")
        self.assertEqual(spec.name, "/tmp/custom.jsonl")
        self.assertEqual(spec.split, "validation")

    def test_resolves_explicit_unified_reference_with_path_as_name(self):
        spec = DatasetResolver().resolve("unified:///tmp/unified_audio:train")

        self.assertEqual(spec.source, "unified")
        self.assertEqual(spec.path, "/tmp/unified_audio")
        self.assertEqual(spec.name, "/tmp/unified_audio")
        self.assertEqual(spec.split, "train")

    def test_resolves_builtin_audio_datasets(self):
        cases = {
            "fleurs": (
                "huggingface",
                "google/fleurs",
                "train",
                "en_us",
                True,
            ),
            "librispeech_asr": (
                "huggingface",
                "openslr/librispeech_asr",
                "train.100",
                "clean",
                True,
            ),
            "esc50": ("huggingface", "ashraq/esc50", "train", None, True),
            "nsynth": (
                "huggingface",
                "confit/nsynth-parquet",
                "train",
                "instrument",
                True,
            ),
            "fsd50k": (
                "huggingface_audio_files",
                "Fhrozen/FSD50k",
                "dev",
                None,
                False,
            ),
        }
        for (
            name,
            (source, path, split, config_name, uses_hf_streaming),
        ) in cases.items():
            with self.subTest(name=name):
                spec = DatasetResolver().resolve(name)

                self.assertEqual(spec.source, source)
                self.assertEqual(spec.path, path)
                self.assertEqual(spec.name, name)
                self.assertEqual(spec.split, split)
                if uses_hf_streaming:
                    self.assertTrue(spec.load_options["streaming"])
                else:
                    self.assertNotIn("streaming", spec.load_options)
                if config_name is None:
                    self.assertNotIn("config_name", spec.load_options)
                else:
                    self.assertEqual(spec.load_options["config_name"], config_name)

    def test_builtin_audio_dataset_split_can_be_overridden(self):
        spec = DatasetResolver().resolve("fleurs:validation")

        self.assertEqual(spec.split, "validation")
        self.assertEqual(spec.load_options["config_name"], "en_us")
        self.assertEqual(spec.key, "fleurs:validation")

    def test_unknown_dataset_requires_map(self):
        with self.assertRaises(KeyError):
            DatasetResolver().resolve("missing:train")


if __name__ == "__main__":
    unittest.main()
