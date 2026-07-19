import json
from pathlib import Path
import tempfile
import unittest

from anydataset import (
    AnyDataset,
    Source,
    Spec,
    has_source,
    register_source,
    resolve_dataset,
)
from anydataset.dataset.source import for_source
from anydataset.dataset.source.store import StoreSource


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

    def test_rejects_non_callable_source_factory(self):
        with self.assertRaisesRegex(TypeError, "factory must be callable"):
            register_source("unit_test_invalid_factory", None)

    def test_unknown_source_fails_when_resolved(self):
        with self.assertRaises(KeyError):
            for_source("unit_test_missing")

    def test_store_source_rejects_unknown_load_options_before_io(self):
        spec = Spec(
            source=Source.STORE,
            path="missing-store",
            load_options={"unknown": True},
        )

        with self.assertRaisesRegex(TypeError, "Unexpected store load option: unknown"):
            StoreSource().prepare(spec, Path("unused-cache"))


if __name__ == "__main__":
    unittest.main()
