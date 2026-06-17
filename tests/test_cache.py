import json
import tempfile
import unittest
from pathlib import Path

from anydatasets.cache import CacheManager
from anydatasets.registry import DatasetSpec


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


if __name__ == "__main__":
    unittest.main()
