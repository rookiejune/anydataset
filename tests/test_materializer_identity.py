from __future__ import annotations

import unittest
from pathlib import Path

from anydataset.store.materializer import ViewMaterializer, _callable_id


def _factory(prefix: str):
    def create(_device: str):
        return prefix

    return create


def _dataset_factory():
    return []


def _provider_factory(_device: str):
    return None


class MaterializerIdentityTest(unittest.TestCase):
    def test_callable_identity_includes_closure_values(self):
        first = _callable_id(_factory("old"))
        second = _callable_id(_factory("new"))
        repeated = _callable_id(_factory("old"))

        self.assertNotEqual(first, second)
        self.assertEqual(first, repeated)

    def test_resume_metadata_includes_explicit_semantic_ids(self):
        materializer = ViewMaterializer(
            Path("output"),
            input_id="input-v2",
            provider_id="provider-v3",
        )

        metadata = materializer._resume_metadata(
            [],
            dataset_factory=_dataset_factory,
            provider_factory=_provider_factory,
            expected=0,
            use_map_style_loader=True,
        )

        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["input"]["semantic_id"], "input-v2")
        self.assertEqual(metadata["provider"]["semantic_id"], "provider-v3")

    def test_rejects_empty_semantic_id(self):
        with self.assertRaisesRegex(ValueError, "provider_id"):
            ViewMaterializer("output", provider_id="")


if __name__ == "__main__":
    unittest.main()
