import json
import tempfile
import unittest
from pathlib import Path

from anydataset import Source, Spec
from anydataset.cache import CacheManager


class CacheManagerTest(unittest.TestCase):
    def test_prepare_creates_stable_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(tmpdir)
            spec = Spec(
                source=Source.HF,
                path="ylecun/mnist",
                split="train",
            )

            first = manager.prepare(spec)
            second = manager.prepare(spec)

            self.assertEqual(first.cache_path, second.cache_path)
            self.assertTrue(first.metadata_path.exists())
            metadata = json.loads(Path(first.metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["source"], "huggingface")
            self.assertEqual(metadata["path"], "ylecun/mnist")
            self.assertEqual(metadata["split"], "train")
            self.assertEqual(metadata["id"], spec.id)

    def test_cache_path_uses_physical_spec_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(tmpdir)
            first = Spec(
                source=Source.HF,
                path="google/fleurs",
                split="train",
                load_options={"config_name": "en_us", "streaming": True},
            )
            second = Spec(
                source=Source.HF,
                path="google/fleurs",
                split="train",
                load_options={"config_name": "en_us", "streaming": True},
            )

            self.assertEqual(manager.prepare(first).cache_path, manager.prepare(second).cache_path)


if __name__ == "__main__":
    unittest.main()
