import json
import tempfile
import unittest
from pathlib import Path

from anydataset.api.cache import CacheManager
from anydataset.api.spec import DatasetSpec


class CacheManagerTest(unittest.TestCase):
    def test_prepare_creates_stable_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(tmpdir)
            spec = DatasetSpec(
                source="huggingface",
                path="ylecun/mnist",
                name="mnist",
                split="train",
            )

            first = manager.prepare(spec)
            second = manager.prepare(spec)

            self.assertEqual(first.cache_path, second.cache_path)
            self.assertTrue(first.metadata_path.exists())
            metadata = json.loads(Path(first.metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["source"], "huggingface")
            self.assertEqual(metadata["name"], "mnist")
            self.assertEqual(metadata["split"], "train")

    def test_cache_path_ignores_sample_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(tmpdir)
            base = DatasetSpec(
                source="huggingface",
                path="org/audio",
                name="audio",
                split="train",
            )
            with_sample_metadata = DatasetSpec(
                source="huggingface",
                path="org/audio",
                name="audio",
                split="train",
                sample_metadata={"unused": "metadata"},
            )

            self.assertEqual(
                manager.dataset_cache_path(base),
                manager.dataset_cache_path(with_sample_metadata),
            )

    def test_cache_path_ignores_dataset_ref(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(tmpdir)
            base = DatasetSpec(
                source="huggingface",
                path="google/fleurs",
                name="fleurs",
                split="train",
                load_options={"config_name": "en_us", "streaming": True},
                ref="fleurs",
            )
            explicit_split = DatasetSpec(
                source="huggingface",
                path="google/fleurs",
                name="fleurs",
                split="train",
                load_options={"config_name": "en_us", "streaming": True},
                ref="fleurs:train",
            )

            self.assertEqual(
                manager.dataset_cache_path(base),
                manager.dataset_cache_path(explicit_split),
            )


if __name__ == "__main__":
    unittest.main()
