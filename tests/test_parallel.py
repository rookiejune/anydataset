from __future__ import annotations

import pickle
import unittest
from dataclasses import dataclass
from unittest import mock

from anydataset._parallel import (
    GlobalIndexSampler,
    MapIndexedDataset,
    map_style_indexed_loader,
)


class ParallelRuntimeTest(unittest.TestCase):
    def test_global_index_sampler_shards_by_rank(self):
        sampler = GlobalIndexSampler(sample_count=8, num_shards=3, shard_id=1)

        self.assertEqual(list(sampler), [1, 4, 7])
        self.assertEqual(len(sampler), 3)

    def test_map_indexed_dataset_drops_cached_dataset_when_pickled(self):
        dataset = _UnpicklableDataset(3)
        wrapper = MapIndexedDataset(_DatasetFactory(3), dataset=dataset)

        restored = pickle.loads(pickle.dumps(wrapper))

        self.assertEqual(restored[2], (2, 2))

    def test_map_style_loader_uses_rank_sampler_not_worker_shard(self):
        with mock.patch.dict("os.environ", {"WORLD_SIZE": "2", "RANK": "1"}):
            loader = map_style_indexed_loader(
                _DatasetFactory(6),
                sample_count=6,
                batch_size=2,
                num_workers=0,
            )

            rows = [row for batch in loader for row in batch]

        self.assertEqual(rows, [(1, 1), (3, 3), (5, 5)])


@dataclass(frozen=True)
class _DatasetFactory:
    count: int

    def __call__(self):
        return list(range(self.count))


class _UnpicklableDataset:
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> int:
        return index

    def __getstate__(self):
        raise TypeError("dataset instance must not be pickled")


if __name__ == "__main__":
    unittest.main()
