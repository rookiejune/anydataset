from __future__ import annotations

import os
import pickle
import unittest
from dataclasses import dataclass
from unittest import mock

from anydataset._parallel import (
    GlobalIndexSampler,
    MapIndexedDataset,
    SelectedIndexSampler,
    map_style_indexed_loader,
    restore_environment,
    set_single_worker_environment,
    validate_process_parent,
)
from anydataset._resume import missing_indexes


class ParallelRuntimeTest(unittest.TestCase):
    def test_global_index_sampler_shards_by_rank(self):
        sampler = GlobalIndexSampler(sample_count=8, num_shards=3, shard_id=1)

        self.assertEqual(list(sampler), [1, 4, 7])
        self.assertEqual(len(sampler), 3)

    def test_selected_index_sampler_shards_missing_indexes_by_rank(self):
        sampler = SelectedIndexSampler((2, 5, 9, 12), num_shards=2, shard_id=1)

        self.assertEqual(list(sampler), [5, 12])
        self.assertEqual(len(sampler), 2)

    def test_selected_index_sampler_accepts_compact_range(self):
        sampler = SelectedIndexSampler(
            range(20_000_000),
            num_shards=4,
            shard_id=3,
        )

        self.assertEqual(len(sampler), 5_000_000)
        self.assertEqual(next(iter(sampler)), 3)

    def test_selected_index_sampler_does_not_scan_lazy_complement(self):
        indexes = missing_indexes(frozenset({1, 4}), 20_000_000)

        with mock.patch.object(
            type(indexes),
            "__iter__",
            side_effect=AssertionError("lazy complement was scanned"),
        ):
            sampler = SelectedIndexSampler(indexes, num_shards=2, shard_id=0)
            first = next(iter(sampler))

        self.assertEqual(len(sampler), 9_999_999)
        self.assertEqual(first, 0)

    def test_selected_index_sampler_rejects_descending_range(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SelectedIndexSampler(range(4, -1, -1), num_shards=1, shard_id=0)

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

    def test_map_style_loader_reads_selected_indexes_only(self):
        dataset = _TrackedDataset(6)

        loader = map_style_indexed_loader(
            _DatasetFactory(6),
            sample_count=6,
            sample_indexes=(2, 5),
            batch_size=2,
            num_workers=0,
            dataset=dataset,
        )

        rows = [row for batch in loader for row in batch]

        self.assertEqual(rows, [(2, 2), (5, 5)])
        self.assertEqual(dataset.calls, [2, 5])

    def test_single_worker_environment_does_not_bind_free_port(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch(
                "anydataset._parallel.free_port",
                side_effect=AssertionError("free_port must not be called"),
            ):
                previous = set_single_worker_environment(
                    "cpu",
                    device_env="ANYDATASET_TEST_DEVICE",
                )
            try:
                self.assertEqual(os.environ["MASTER_PORT"], "0")
                self.assertEqual(os.environ["WORLD_SIZE"], "1")
                self.assertEqual(os.environ["ANYDATASET_TEST_DEVICE"], "cpu")
            finally:
                restore_environment(previous)

    def test_process_parent_rejects_daemonic_process(self):
        process = mock.Mock()
        process.daemon = True
        process.name = "daemon-parent"

        with mock.patch(
            "anydataset._parallel.multiprocessing.current_process",
            return_value=process,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "cannot start child processes.*application main process",
            ):
                validate_process_parent(context="materialization")

    def test_process_parent_accepts_non_daemonic_process(self):
        process = mock.Mock()
        process.daemon = False

        with mock.patch(
            "anydataset._parallel.multiprocessing.current_process",
            return_value=process,
        ):
            validate_process_parent(context="materialization")


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


class _TrackedDataset:
    def __init__(self, count: int) -> None:
        self.count = count
        self.calls: list[int] = []

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> int:
        self.calls.append(index)
        return index


if __name__ == "__main__":
    unittest.main()
