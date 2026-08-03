import json
import pickle
from pathlib import Path
import tempfile
import unittest

from anydataset import (
    AnyDataset,
    Source,
    Spec,
    register_source,
    resolve_dataset,
)
import anydataset.dataset.source as source_module
from anydataset.dataset.source._registry import create_source, source_exists
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
        self.assertTrue(source_exists(spec.source))

    def test_rejects_duplicate_source_registration(self):
        register_source("unit_test_duplicate", ListSource)

        with self.assertRaises(ValueError):
            register_source("unit_test_duplicate", ListSource)

    def test_rejects_non_callable_source_factory(self):
        with self.assertRaisesRegex(TypeError, "factory must be callable"):
            register_source("unit_test_invalid_factory", None)

    def test_custom_source_declares_operational_identity_options(self):
        register_source(
            "unit_test_operational_options",
            ListSource,
            operational_load_options=("worker_count",),
        )

        base = Spec(source="unit_test_operational_options", path="/tmp/custom")
        operational = Spec(
            source="unit_test_operational_options",
            path="/tmp/custom",
            load_options={"worker_count": 4},
        )
        physical = Spec(
            source="unit_test_operational_options",
            path="/tmp/custom",
            load_options={"format": "jsonl"},
        )

        self.assertEqual(base.id, operational.id)
        self.assertNotEqual(base.id, physical.id)
        self.assertNotIn("worker_count", operational.to_dict()["load_options"])
        self.assertEqual(physical.to_dict()["load_options"]["format"], "jsonl")

    def test_prepare_workers_remains_globally_operational(self):
        for source in (*Source, "unit_test_unregistered_identity"):
            with self.subTest(source=source):
                base = Spec(source=source, path="/tmp/custom")
                with_workers = Spec(
                    source=source,
                    path="/tmp/custom",
                    load_options={"prepare_workers": 4},
                )

                self.assertEqual(base.id, with_workers.id)
                self.assertNotIn(
                    "prepare_workers",
                    with_workers.to_dict()["load_options"],
                )

    def test_spec_identity_is_stable_across_late_source_registration(self):
        source = "unit_test_late_identity_registration"
        existing = Spec(
            source=source,
            path="/tmp/custom",
            load_options={"worker_count": 4},
        )
        before = existing.to_dict()

        register_source(
            source,
            ListSource,
            operational_load_options=("worker_count",),
        )
        later = Spec(
            source=source,
            path="/tmp/custom",
            load_options={"worker_count": 4},
        )
        restored = pickle.loads(pickle.dumps(existing))

        self.assertEqual(existing.to_dict(), before)
        self.assertEqual(restored, existing)
        self.assertEqual(restored.id, existing.id)
        self.assertNotEqual(later.id, existing.id)
        self.assertNotEqual(later, existing)

    def test_rejects_invalid_operational_identity_options(self):
        with self.assertRaisesRegex(TypeError, "collection of strings"):
            register_source(
                "unit_test_string_operational_options",
                ListSource,
                operational_load_options="worker_count",
            )
        with self.assertRaisesRegex(ValueError, "empty strings"):
            register_source(
                "unit_test_empty_operational_option",
                ListSource,
                operational_load_options=("",),
            )

    def test_unknown_source_fails_when_resolved(self):
        with self.assertRaises(KeyError):
            create_source("unit_test_missing")

    def test_source_exports_physical_categories_not_concrete_datasets(self):
        self.assertEqual(
            source_module.__all__,
            [
                "DatasetSource",
                "HuggingFaceDiskSource",
                "HuggingFaceFilesSource",
                "HuggingFaceSource",
                "ShardedCsvSource",
                "ShardingSource",
                "StoreSource",
                "TsvSource",
                "register_source",
            ],
        )
        self.assertIn("HuggingFaceFilesSource", source_module.__all__)
        self.assertIn("ShardingSource", source_module.__all__)
        self.assertNotIn("FSD50KSource", source_module.__all__)
        self.assertNotIn("FSD50KDataset", source_module.__all__)
        self.assertNotIn("ShardedCsvDataset", source_module.__all__)
        self.assertNotIn("TsvDataset", source_module.__all__)
        self.assertNotIn("SourceFactory", source_module.__all__)
        self.assertNotIn("prepare_hf", source_module.__all__)
        self.assertNotIn("prepare_hf_disk", source_module.__all__)
        self.assertNotIn("for_source", source_module.__all__)
        self.assertNotIn("has_source", source_module.__all__)
        self.assertFalse(hasattr(source_module, "FSD50KSource"))
        self.assertFalse(hasattr(source_module, "FSD50KDataset"))
        self.assertFalse(hasattr(source_module, "ShardedCsvDataset"))
        self.assertFalse(hasattr(source_module, "TsvDataset"))
        self.assertFalse(hasattr(source_module, "SourceFactory"))
        self.assertFalse(hasattr(source_module, "prepare_hf"))
        self.assertFalse(hasattr(source_module, "prepare_hf_disk"))
        self.assertFalse(hasattr(source_module, "for_source"))
        self.assertFalse(hasattr(source_module, "has_source"))

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
