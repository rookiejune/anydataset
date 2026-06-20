import sys
import tempfile
import types
import unittest
from unittest import mock

from anydataset.api.cache import CacheManager
from anydataset.api.spec import DatasetSpec
from anydataset.datasets.huggingface import HuggingFaceDataset


class HuggingFaceDatasetTest(unittest.TestCase):
    def test_prepare_maps_config_name_to_load_dataset_name(self):
        calls = []
        fake_datasets = types.ModuleType("datasets")

        def load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return [{"value": 1}]

        fake_datasets.load_dataset = load_dataset
        spec = DatasetSpec(
            source="huggingface",
            path="org/audio",
            name="audio",
            split="train",
            load_options={
                "config_name": "clean",
                "streaming": True,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(tmpdir).prepare(spec)
            with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
                manifest = HuggingFaceDataset().prepare(spec, cache)

        self.assertEqual(manifest, [{"value": 1}])
        self.assertEqual(calls[0][0], ("org/audio",))
        self.assertEqual(calls[0][1]["split"], "train")
        self.assertEqual(calls[0][1]["name"], "clean")
        self.assertTrue(calls[0][1]["streaming"])

    def test_iter_indexed_samples_preserves_global_indices(self):
        manifest = _ShardableManifest(
            rows=[
                {"value": 0},
                {"value": 1},
                {"value": 2},
            ]
        )

        rows = list(
            HuggingFaceDataset().iter_indexed_samples(
                manifest,
                num_shards=2,
                shard_id=1,
            )
        )

        self.assertEqual(manifest.shard_calls, [])
        self.assertEqual(rows, [(1, {"value": 1})])


class _ShardableManifest:
    def __init__(self, rows):
        self.rows = rows
        self.shard_calls = []

    def __iter__(self):
        yield from self.rows

    def shard(self, num_shards, index):
        self.shard_calls.append((num_shards, index))
        return [
            row
            for row_index, row in enumerate(self.rows)
            if row_index % num_shards == index
        ]


if __name__ == "__main__":
    unittest.main()
