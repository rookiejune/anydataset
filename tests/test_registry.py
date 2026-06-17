import unittest

from anydatasets.registry import DatasetRegistry, DatasetSpec


class DatasetRegistryTest(unittest.TestCase):
    def test_resolves_default_dataset_with_split(self):
        spec = DatasetRegistry().resolve("mnist:train")

        self.assertEqual(spec.source, "huggingface")
        self.assertEqual(spec.path, "ylecun/mnist")
        self.assertEqual(spec.name, "mnist")
        self.assertEqual(spec.split, "train")
        self.assertEqual(spec.key, "mnist:train")

    def test_resolves_custom_dataset_map(self):
        registry = DatasetRegistry(
            {
                "custom": DatasetSpec(
                    source="local_files",
                    path="/tmp/custom.jsonl",
                )
            }
        )

        spec = registry.resolve("custom:validation")

        self.assertEqual(spec.source, "local_files")
        self.assertEqual(spec.path, "/tmp/custom.jsonl")
        self.assertEqual(spec.name, "custom")
        self.assertEqual(spec.split, "validation")

    def test_resolves_explicit_huggingface_reference(self):
        spec = DatasetRegistry().resolve("hf://org/name:train")

        self.assertEqual(spec.source, "huggingface")
        self.assertEqual(spec.path, "org/name")
        self.assertEqual(spec.name, "name")
        self.assertEqual(spec.split, "train")

    def test_unknown_dataset_requires_map(self):
        with self.assertRaises(KeyError):
            DatasetRegistry().resolve("missing:train")


if __name__ == "__main__":
    unittest.main()
