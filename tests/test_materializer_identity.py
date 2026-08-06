from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from anydataset.store import ViewMaterializer
from anydataset.store.materialize.identity import callable_id, metadata_value


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


class _ExecutionDatasetFactory:
    def __init__(self, cache: Path) -> None:
        self.cache = cache

    def __call__(self):
        return []


class _ExecutionProviderFactory:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def __call__(self, _device: str):
        return None


def _dataset_factory():
    return []


def _provider_factory(_device: str):
    return None


class MaterializerIdentityTest(unittest.TestCase):
    def test_callable_identity_includes_closure_values(self):
        first = callable_id(_factory("old"))
        second = callable_id(_factory("new"))
        repeated = callable_id(_factory("old"))

        self.assertNotEqual(first, second)
        self.assertEqual(first, repeated)

    def test_callable_identity_includes_plain_instance_state(self):
        first = callable_id(_StatefulFactory("old"))
        second = callable_id(_StatefulFactory("new"))
        repeated = callable_id(_StatefulFactory("old"))

        self.assertNotEqual(first, second)
        self.assertEqual(first, repeated)

    def test_callable_identity_stabilizes_callable_closure_values(self):
        first = callable_id(_callback_factory(_callback()))
        second = callable_id(_callback_factory(_callback()))

        self.assertEqual(first, second)

    def test_metadata_identity_preserves_mapping_key_types(self):
        metadata = metadata_value({1: "integer", "1": "string"})

        self.assertEqual(len(metadata["items"]), 2)

    def test_metadata_identity_distinguishes_tuple_and_list(self):
        self.assertNotEqual(metadata_value((1, 2)), metadata_value([1, 2]))

    def test_callable_identity_hashes_large_tensor_contents_compactly(self):
        first_tensor = torch.zeros(300_000)
        changed_tensor = first_tensor.clone()
        changed_tensor[150_000] = 1

        first = callable_id(_TensorFactory(first_tensor))
        repeated = callable_id(_TensorFactory(first_tensor.clone()))
        changed = callable_id(_TensorFactory(changed_tensor))

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertLess(len(json.dumps(first)), 1000)

    def test_callable_identity_handles_contentless_meta_tensor(self):
        first = callable_id(_TensorFactory(torch.empty(2, device="meta")))
        repeated = callable_id(_TensorFactory(torch.empty(2, device="meta")))

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

        self.assertEqual(metadata["schema_version"], 7)
        self.assertIsNone(metadata["materializer"]["max_shard_bytes"])
        self.assertEqual(metadata["input"]["semantic_id"], "input-v2")
        self.assertEqual(metadata["provider"]["semantic_id"], "provider-v3")
        self.assertEqual(
            metadata["input"]["factory"],
            {"kind": "semantic", "id": "input-v2"},
        )
        self.assertEqual(
            metadata["provider"]["factory"],
            {"kind": "semantic", "id": "provider-v3"},
        )

    def test_explicit_semantic_ids_exclude_execution_only_factory_state(self):
        materializer = ViewMaterializer(
            Path("output"),
            input_id="input-v2",
            provider_id="provider-v3",
        )

        first = materializer._resume_metadata(
            [],
            dataset_factory=_ExecutionDatasetFactory(Path("cache-a")),
            provider_factory=_ExecutionProviderFactory("worker-a"),
            expected=0,
            use_map_style_loader=True,
        )
        second = materializer._resume_metadata(
            [],
            dataset_factory=_ExecutionDatasetFactory(Path("cache-b")),
            provider_factory=_ExecutionProviderFactory("worker-b"),
            expected=0,
            use_map_style_loader=True,
        )

        self.assertEqual(first, second)

    def test_resume_metadata_still_distinguishes_semantic_id_changes(self):
        dataset_factory = _ExecutionDatasetFactory(Path("cache"))
        provider_factory = _ExecutionProviderFactory("worker")

        first = ViewMaterializer(
            Path("output"),
            input_id="input-v1",
            provider_id="provider-v1",
        )._resume_metadata(
            [],
            dataset_factory=dataset_factory,
            provider_factory=provider_factory,
            expected=0,
            use_map_style_loader=True,
        )
        changed_input = ViewMaterializer(
            Path("output"),
            input_id="input-v2",
            provider_id="provider-v1",
        )._resume_metadata(
            [],
            dataset_factory=dataset_factory,
            provider_factory=provider_factory,
            expected=0,
            use_map_style_loader=True,
        )
        changed_provider = ViewMaterializer(
            Path("output"),
            input_id="input-v1",
            provider_id="provider-v2",
        )._resume_metadata(
            [],
            dataset_factory=dataset_factory,
            provider_factory=provider_factory,
            expected=0,
            use_map_style_loader=True,
        )

        self.assertNotEqual(first, changed_input)
        self.assertNotEqual(first, changed_provider)

    def test_rejects_empty_semantic_id(self):
        with self.assertRaisesRegex(ValueError, "provider_id"):
            ViewMaterializer("output", provider_id="")

        with self.assertRaisesRegex(ValueError, "dataset_id"):
            ViewMaterializer("output", dataset_id="")


if __name__ == "__main__":
    unittest.main()
