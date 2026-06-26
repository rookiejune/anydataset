import sys
import tempfile
import types
import unittest
from unittest import mock

from anydataset import AnyDataset, Source, Spec


class HuggingFaceDatasetTest(unittest.TestCase):
    def test_prepare_maps_config_name_to_load_dataset_name(self):
        calls = []
        fake_datasets = types.ModuleType("datasets")

        def load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return [{"value": 1}]

        fake_datasets.load_dataset = load_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                Spec(
                    source=Source.HF,
                    path="org/audio",
                    split="train",
                    load_options={
                        "config_name": "clean",
                        "streaming": True,
                    },
                ),
                cache_root=tmpdir,
            )
            with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
                prepared = dataset.prepare()

        self.assertEqual(prepared, [{"value": 1}])
        self.assertEqual(calls[0][0], ("org/audio",))
        self.assertEqual(calls[0][1]["split"], "train")
        self.assertEqual(calls[0][1]["name"], "clean")
        self.assertTrue(calls[0][1]["streaming"])

    def test_prepare_loads_dataset_from_disk_split(self):
        fake_datasets = types.ModuleType("datasets")

        class DatasetDict(dict):
            pass

        fake_datasets.DatasetDict = DatasetDict
        fake_datasets.load_from_disk = lambda *args, **kwargs: DatasetDict(
            train=[{"value": 1}],
            validation=[{"value": 2}],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                Spec(
                    source=Source.HF_DISK,
                    path="/tmp/saved_dataset",
                    split="validation",
                    load_options={"keep_in_memory": True},
                ),
                cache_root=tmpdir,
            )
            with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
                prepared = dataset.prepare()

        self.assertEqual(prepared, [{"value": 2}])

    def test_prepare_requires_split_for_dataset_dict(self):
        fake_datasets = types.ModuleType("datasets")

        class DatasetDict(dict):
            pass

        fake_datasets.DatasetDict = DatasetDict
        fake_datasets.load_from_disk = lambda *args, **kwargs: DatasetDict(
            train=[{"value": 1}],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                Spec(source=Source.HF_DISK, path="/tmp/saved_dataset"),
                cache_root=tmpdir,
            )
            with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
                with self.assertRaises(ValueError):
                    dataset.prepare()


if __name__ == "__main__":
    unittest.main()
