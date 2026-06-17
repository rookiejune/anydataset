import tempfile
import unittest

from anydatasets import AnyIterableDataset, DatasetSpec, Task
from anydatasets.adapters.base import DatasetAdapter


class StaticAdapter(DatasetAdapter):
    def __init__(self, rows):
        self.rows = rows

    def prepare(self, spec, cache):
        return self.rows

    def iter_samples(self, manifest):
        yield from manifest


class AnyIterableDatasetTest(unittest.TestCase):
    def test_rejects_string_task(self):
        with self.assertRaises(TypeError):
            AnyIterableDataset(datasets=["mnist:train"], task="image_classification", batch_size=2)

    def test_iterates_dataclass_batches(self):
        adapter = StaticAdapter(
            [
                {"image": [[1, 2], [3, 4]], "label": 0},
                {"image": [[5, 6], [7, 8]], "label": 1},
                {"image": [[9, 10], [11, 12]], "label": 2},
            ]
        )
        dataset_map = {
            "toy": DatasetSpec(source="static", path="toy", adapter=adapter),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyIterableDataset(
                datasets=["toy:train"],
                task=Task.IMAGE_CLASSIFICATION,
                batch_size=2,
                dataset_map=dataset_map,
                cache_dir=tmpdir,
                seed=1,
            )
            batches = list(dataset)

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].meta.dataset_names, ["toy:train", "toy:train"])
        self.assertEqual(batches[0].meta.sample_indices, [0, 1])
        self.assertEqual(batches[1].meta.sample_indices, [2])


if __name__ == "__main__":
    unittest.main()
