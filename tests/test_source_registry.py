import json
from pathlib import Path
import tempfile
import unittest

from anydataset import (
    AnyDataset,
    Spec,
    for_source,
    has_source,
    register_source,
    resolve_dataset,
)


class ListSource:
    def prepare(self, spec: Spec, cache_path: Path):
        return [{"path": spec.path, "cache_path": str(cache_path)}]


class SourceRegistryTest(unittest.TestCase):
    def test_registers_custom_source_for_dataset_prepare(self):
        register_source("unit_test_list", ListSource)
        with tempfile.TemporaryDirectory():
            spec = Spec(source="unit_test_list", path="/tmp/custom")
            dataset = AnyDataset(spec)
            metadata_path = dataset.cache_manager.root / "sources" / spec.cache_relpath / "metadata.json"

            self.assertEqual(dataset[0]["path"], "/tmp/custom")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["source"], "unit_test_list")

    def test_resolves_registered_source_shorthand(self):
        register_source("unit_test_shorthand", ListSource)

        spec = resolve_dataset("unit_test_shorthand:///tmp/custom:train")

        self.assertEqual((spec.source, spec.path, spec.split), (
            "unit_test_shorthand",
            "/tmp/custom",
            "train",
        ))
        self.assertTrue(has_source(spec.source))

    def test_rejects_duplicate_source_registration(self):
        register_source("unit_test_duplicate", ListSource)

        with self.assertRaises(ValueError):
            register_source("unit_test_duplicate", ListSource)

    def test_unknown_source_fails_when_resolved(self):
        with self.assertRaises(KeyError):
            for_source("unit_test_missing")


if __name__ == "__main__":
    unittest.main()
