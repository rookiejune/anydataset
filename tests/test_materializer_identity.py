from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from anydataset.store.materializer import (
    ViewMaterializer,
    _callable_id,
    _metadata_value,
)


def _factory(prefix: str):
    def create(_device: str):
        return prefix

    return create


def _callback():
    def callback():
        return None

    return callback


def _callback_factory(callback):
    def create(_device: str):
        return callback()

    return create


class _StatefulFactory:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __call__(self, _device: str):
        return self.prefix


class _TensorFactory:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __call__(self, _device: str):
        return self.tensor


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

    def test_callable_identity_includes_plain_instance_state(self):
        first = _callable_id(_StatefulFactory("old"))
        second = _callable_id(_StatefulFactory("new"))
        repeated = _callable_id(_StatefulFactory("old"))

        self.assertNotEqual(first, second)
        self.assertEqual(first, repeated)

    def test_callable_identity_stabilizes_callable_closure_values(self):
        first = _callable_id(_callback_factory(_callback()))
        second = _callable_id(_callback_factory(_callback()))

        self.assertEqual(first, second)

    def test_metadata_identity_preserves_mapping_key_types(self):
        metadata = _metadata_value({1: "integer", "1": "string"})

        self.assertEqual(len(metadata["items"]), 2)

    def test_metadata_identity_distinguishes_tuple_and_list(self):
        self.assertNotEqual(_metadata_value((1, 2)), _metadata_value([1, 2]))

    def test_callable_identity_hashes_large_tensor_contents_compactly(self):
        first_tensor = torch.zeros(300_000)
        changed_tensor = first_tensor.clone()
        changed_tensor[150_000] = 1

        first = _callable_id(_TensorFactory(first_tensor))
        repeated = _callable_id(_TensorFactory(first_tensor.clone()))
        changed = _callable_id(_TensorFactory(changed_tensor))

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertLess(len(json.dumps(first)), 1000)

    def test_callable_identity_handles_contentless_meta_tensor(self):
        first = _callable_id(_TensorFactory(torch.empty(2, device="meta")))
        repeated = _callable_id(_TensorFactory(torch.empty(2, device="meta")))

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

        self.assertEqual(metadata["schema_version"], 3)
        self.assertEqual(metadata["input"]["semantic_id"], "input-v2")
        self.assertEqual(metadata["provider"]["semantic_id"], "provider-v3")

    def test_rejects_empty_semantic_id(self):
        with self.assertRaisesRegex(ValueError, "provider_id"):
            ViewMaterializer("output", provider_id="")


if __name__ == "__main__":
    unittest.main()
