from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from anydataset import (
    AnyDataset,
    LocalFilesAdapter,
    DatasetSource,
    DatasetSpec,
    RoundRobinStrategy,
    Task,
    WeightedRandomStrategy,
)
from anydataset.adapters import DatasetAdapter
from anydataset import AudioKey, AudioView, ModalityKey


class StaticAdapter(DatasetAdapter):
    def __init__(self, rows):
        self.rows = rows

    def prepare(self, spec, cache):
        return self.rows

    def iter_samples(self, manifest):
        yield from manifest


class CountingMaterializeAdapter(StaticAdapter):
    def __init__(self, rows, delay):
        super().__init__(rows)
        self.delay = delay
        self.materialize_count = 0
        self._lock = threading.Lock()

    def prepare(self, spec, cache):
        if not cache.ready_path.exists():
            with self._lock:
                self.materialize_count += 1
            time.sleep(self.delay)
        return self.rows


class _StaticAudioAdapter(StaticAdapter):
    def __init__(self, rows):
        super().__init__(rows)
        self._fields = LocalFilesAdapter(
            waveform_field="samples",
            sample_rate_field="sr",
        )

    def audio(self, row, role=None):
        return self._fields.audio(row, role=role)


class AnyDatasetTest(unittest.TestCase):
    def test_rejects_string_task(self):
        with self.assertRaises(TypeError):
            AnyDataset(datasets=["mnist:train"], task="image_classification")

    def test_accepts_single_dataset_string(self):
        dataset = AnyDataset(
            datasets="mnist:train",
            task=Task.IMAGE_CLASSIFICATION,
        )

        self.assertEqual(len(dataset.specs), 1)
        self.assertEqual(dataset.specs[0].key, "mnist:train")

    def test_iterates_single_dataset_samples(self):
        adapter = StaticAdapter(
            [
                {"image": [[1, 2], [3, 4]], "label": 0},
                {"image": [[5, 6], [7, 8]], "label": 1},
            ]
        )
        dataset_map = {
            "toy": _static_spec("toy"),
        }
        adapter_map = {
            "toy": adapter,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                datasets=["toy:train"],
                task=Task.IMAGE_CLASSIFICATION,
                dataset_map=dataset_map,
                adapter_map=adapter_map,
                cache_dir=tmpdir,
            )
            samples = list(dataset)

        self.assertEqual([sample.dataset_name for sample in samples], ["toy:train", "toy:train"])
        self.assertEqual([sample.sample_index for sample in samples], [0, 1])
        self.assertEqual(samples[0].data["label"], 0)
        self.assertFalse(hasattr(dataset, "dataloader"))

    def test_default_strategy_iterates_datasets_sequentially(self):
        dataset_map = {
            "a": _static_spec("a"),
            "b": _static_spec("b"),
        }
        adapter_map = {
            "a": StaticAdapter([{"value": 1}]),
            "b": StaticAdapter([{"value": 2}]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                datasets=["a:train", "b:train"],
                task=Task.IMAGE_CLASSIFICATION,
                dataset_map=dataset_map,
                adapter_map=adapter_map,
                cache_dir=tmpdir,
            )
            samples = list(dataset)

        self.assertEqual([sample.dataset_name for sample in samples], ["a:train", "b:train"])
        self.assertEqual([sample.data["value"] for sample in samples], [1, 2])

    def test_round_robin_strategy_interleaves_datasets(self):
        dataset_map = {
            "a": _static_spec("a"),
            "b": _static_spec("b"),
        }
        adapter_map = {
            "a": StaticAdapter([{"value": "a0"}, {"value": "a1"}]),
            "b": StaticAdapter([{"value": "b0"}]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                datasets=["a:train", "b:train"],
                task=Task.IMAGE_CLASSIFICATION,
                dataset_map=dataset_map,
                adapter_map=adapter_map,
                cache_dir=tmpdir,
                strategy=RoundRobinStrategy(),
            )
            samples = list(dataset)

        self.assertEqual([sample.data["value"] for sample in samples], ["a0", "b0", "a1"])

    def test_weighted_random_strategy_can_disable_dataset(self):
        dataset_map = {
            "a": _static_spec("a"),
            "b": _static_spec("b"),
        }
        adapter_map = {
            "a": StaticAdapter([{"value": 1}]),
            "b": StaticAdapter([{"value": 2}]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                datasets=["a:train", "b:train"],
                task=Task.IMAGE_CLASSIFICATION,
                dataset_map=dataset_map,
                adapter_map=adapter_map,
                cache_dir=tmpdir,
                strategy=WeightedRandomStrategy(weights={"b:train": 0.0}, seed=1),
            )
            samples = list(dataset)

        self.assertEqual([sample.dataset_name for sample in samples], ["a:train"])

    def test_getitem_returns_single_dataset(self):
        dataset_map = {
            "a": _static_spec("a"),
            "b": _static_spec("b"),
        }
        adapter_map = {
            "a": StaticAdapter([{"value": 1}]),
            "b": StaticAdapter([{"value": 2}]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                datasets=["a:train", "b:train"],
                task=Task.IMAGE_CLASSIFICATION,
                dataset_map=dataset_map,
                adapter_map=adapter_map,
                cache_dir=tmpdir,
            )
            single = dataset["b:train"]
            samples = list(single)

        self.assertIsInstance(single, DatasetSource)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].dataset_name, "b:train")
        self.assertEqual(samples[0].data["value"], 2)

    def test_single_dataset_requires_resolved_spec(self):
        spec = DatasetSpec(
            source="static",
            path="toy",
            name="toy",
            split="train",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = DatasetSource(
                spec=spec,
                task=Task.IMAGE_CLASSIFICATION,
                adapter=StaticAdapter([{"value": 3}]),
                cache_dir=tmpdir,
            )
            samples = list(dataset)

        self.assertEqual([sample.dataset_name for sample in samples], ["toy:train"])
        self.assertEqual(samples[0].data["value"], 3)

        with self.assertRaises(TypeError):
            DatasetSource(
                spec="toy:train",
                task=Task.IMAGE_CLASSIFICATION,
            )

    def test_dataset_accepts_prebuilt_sources(self):
        dataset_map = {
            "a": DatasetSpec(source="static", path="a", name="a", split="train"),
            "b": DatasetSpec(source="static", path="b", name="b", split="train"),
        }
        adapter_map = {
            "a": StaticAdapter([{"value": 1}]),
            "b": StaticAdapter([{"value": 2}]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            singles = [
                DatasetSource(
                    spec=dataset_map["a"],
                    task=Task.IMAGE_CLASSIFICATION,
                    adapter=adapter_map["a"],
                    cache_dir=tmpdir,
                ),
                DatasetSource(
                    spec=dataset_map["b"],
                    task=Task.IMAGE_CLASSIFICATION,
                    adapter=adapter_map["b"],
                    cache_dir=tmpdir,
                ),
            ]
            dataset = AnyDataset(
                datasets=singles,
                strategy=RoundRobinStrategy(),
            )
            samples = list(dataset)

        self.assertEqual([sample.data["value"] for sample in samples], [1, 2])

    def test_prebuilt_sources_reject_outer_mapping_and_options(self):
        spec = _static_spec("toy")
        adapter = StaticAdapter([{"value": 1}])

        with tempfile.TemporaryDirectory() as tmpdir:
            single = DatasetSource(
                spec=spec,
                task=Task.IMAGE_CLASSIFICATION,
                adapter=adapter,
                cache_dir=tmpdir,
            )
            with self.assertRaises(ValueError):
                AnyDataset(
                    datasets=[single],
                    dataset_map={"toy": spec},
                )
            with self.assertRaises(ValueError):
                AnyDataset(
                    datasets=[single],
                    adapter_map={"toy": adapter},
                )
            with self.assertRaises(ValueError):
                AnyDataset(
                    datasets=[single],
                    cache_dir=tmpdir,
                )

    def test_audio_task_uses_modality_adapter(self):
        adapter = _StaticAudioAdapter(
            [
                {
                    "samples": [1.0],
                    "sr": 2,
                }
            ]
        )
        dataset_map = {
            "audio": DatasetSpec(
                source="static",
                path="audio",
                name="audio",
            ),
        }
        adapter_map = {"audio": adapter}
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                datasets=["audio:train"],
                task=Task.AUDIO_CODEC,
                dataset_map=dataset_map,
                adapter_map=adapter_map,
                cache_dir=tmpdir,
            )
            sample = next(iter(dataset))

        audio = sample.data[ModalityKey.AUDIO]
        self.assertEqual(audio[AudioKey.VIEWS][AudioView.WAVEFORM], [1.0])
        self.assertEqual(audio[AudioKey.SAMPLE_RATE], 2)

    def test_manual_shards_are_disjoint(self):
        rows = [{"image": [[index]], "label": index} for index in range(6)]
        dataset_map = {
            "toy": _static_spec("toy"),
        }
        adapter_map = {"toy": StaticAdapter(rows)}

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                datasets=["toy:train"],
                task=Task.IMAGE_CLASSIFICATION,
                dataset_map=dataset_map,
                adapter_map=adapter_map,
                cache_dir=tmpdir,
            )
            even_indices = _sample_indices(dataset.shard(2, 0))
            odd_indices = _sample_indices(dataset.shard(2, 1))

        self.assertEqual(even_indices, [0, 2, 4])
        self.assertEqual(odd_indices, [1, 3, 5])
        self.assertEqual(sorted(even_indices + odd_indices), list(range(6)))

    def test_workers_are_sharded_inside_single_dataset(self):
        rows = [{"image": [[index]], "label": index} for index in range(8)]
        dataset_map = {
            "toy": _static_spec("toy"),
        }
        adapter_map = {"toy": StaticAdapter(rows)}

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                datasets=["toy:train"],
                task=Task.IMAGE_CLASSIFICATION,
                dataset_map=dataset_map,
                adapter_map=adapter_map,
                cache_dir=tmpdir,
            )
            with mock.patch(
                "anydataset.api.dataset.get_worker_info",
                return_value=SimpleNamespace(id=0, num_workers=2),
            ):
                worker_zero_indices = _sample_indices(dataset)
            with mock.patch(
                "anydataset.api.dataset.get_worker_info",
                return_value=SimpleNamespace(id=1, num_workers=2),
            ):
                worker_one_indices = _sample_indices(dataset)

        indices = worker_zero_indices + worker_one_indices
        self.assertEqual(worker_zero_indices, [0, 2, 4, 6])
        self.assertEqual(worker_one_indices, [1, 3, 5, 7])
        self.assertEqual(sorted(indices), list(range(8)))
        self.assertEqual(len(indices), len(set(indices)))

    def test_distributed_iteration_requires_explicit_shard(self):
        rows = [{"image": [[index]], "label": index} for index in range(6)]
        dataset_map = {
            "toy": _static_spec("toy"),
        }
        adapter_map = {"toy": StaticAdapter(rows)}

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = AnyDataset(
                datasets=["toy:train"],
                task=Task.IMAGE_CLASSIFICATION,
                dataset_map=dataset_map,
                adapter_map=adapter_map,
                cache_dir=tmpdir,
            )
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1"}):
                with self.assertRaises(ValueError):
                    list(dataset)
                indices = _sample_indices(dataset.shard(2, 1))

        self.assertEqual(indices, [1, 3, 5])

    def test_concurrent_prepare_materializes_cache_once(self):
        adapter = CountingMaterializeAdapter(
            [{"image": [[1]], "label": 0}],
            delay=0.2,
        )
        dataset_map = {
            "toy": _static_spec("toy"),
        }
        adapter_map = {"toy": adapter}

        def consume(cache_dir):
            dataset = AnyDataset(
                datasets=["toy:train"],
                task=Task.IMAGE_CLASSIFICATION,
                dataset_map=dataset_map,
                adapter_map=adapter_map,
                cache_dir=cache_dir,
            )
            return [sample.sample_index for sample in dataset]

        with tempfile.TemporaryDirectory() as tmpdir:
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(consume, [tmpdir, tmpdir]))

        self.assertEqual(results, [[0], [0]])
        self.assertEqual(adapter.materialize_count, 1)


def _sample_indices(dataset):
    return [sample.sample_index for sample in dataset]


def _static_spec(name):
    return DatasetSpec(
        source="static",
        path=name,
        name=name,
    )


if __name__ == "__main__":
    unittest.main()
