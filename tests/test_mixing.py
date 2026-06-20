import time
import unittest

from anydataset.api.mixing import (
    PrefetchingDatasetMixer,
    SampleStream,
    WeightedDatasetMixer,
)
from anydataset.samples import Sample


def samples(name, count):
    for index in range(count):
        yield Sample(data={"value": index}, dataset_name=name, sample_index=index)


class WeightedDatasetMixerTest(unittest.TestCase):
    def test_yields_all_samples_and_removes_depleted_streams(self):
        mixer = WeightedDatasetMixer(
            [
                SampleStream("a", samples("a", 2), weight=1.0),
                SampleStream("b", samples("b", 3), weight=1.0),
            ],
            seed=7,
        )

        result = list(mixer)

        self.assertEqual(len(result), 5)
        self.assertEqual(
            sorted((sample.dataset_name, sample.sample_index) for sample in result),
            [("a", 0), ("a", 1), ("b", 0), ("b", 1), ("b", 2)],
        )

    def test_rejects_all_zero_weights(self):
        with self.assertRaises(ValueError):
            WeightedDatasetMixer([SampleStream("a", samples("a", 1), weight=0.0)])

        with self.assertRaises(ValueError):
            PrefetchingDatasetMixer(
                [SampleStream("a", samples("a", 1), weight=0.0)],
                queue_size=2,
            )

    def test_weighted_mixer_ignores_zero_weight_streams(self):
        mixer = WeightedDatasetMixer(
            [
                SampleStream("active", samples("active", 1), weight=1.0),
                SampleStream("disabled", samples("disabled", 2), weight=0.0),
            ],
            seed=7,
        )

        result = list(mixer)

        self.assertEqual(
            [(sample.dataset_name, sample.sample_index) for sample in result],
            [("active", 0)],
        )

    def test_prefetching_mixer_ignores_zero_weight_streams(self):
        mixer = PrefetchingDatasetMixer(
            [
                SampleStream("active", samples("active", 1), weight=1.0),
                SampleStream("disabled", samples("disabled", 2), weight=0.0),
            ],
            queue_size=2,
            seed=7,
        )

        result = list(mixer)

        self.assertEqual(
            [(sample.dataset_name, sample.sample_index) for sample in result],
            [("active", 0)],
        )

    def test_prefetching_mixer_uses_ready_stream_instead_of_blocking_on_slow_stream(self):
        def delayed_samples(name, count, delay):
            time.sleep(delay)
            yield from samples(name, count)

        mixer = PrefetchingDatasetMixer(
            [
                SampleStream("slow", delayed_samples("slow", 1, delay=0.5), weight=1.0),
                SampleStream("fast", samples("fast", 1), weight=1.0),
            ],
            queue_size=2,
            wait_timeout=1.0,
        )

        start = time.monotonic()
        first = next(iter(mixer))
        elapsed = time.monotonic() - start

        self.assertEqual(first.dataset_name, "fast")
        self.assertLess(elapsed, 0.3)

    def test_prefetching_mixer_yields_all_samples(self):
        mixer = PrefetchingDatasetMixer(
            [
                SampleStream("a", samples("a", 2), weight=1.0),
                SampleStream("b", samples("b", 3), weight=1.0),
            ],
            queue_size=4,
            seed=3,
        )

        result = list(mixer)

        self.assertEqual(len(result), 5)
        self.assertEqual(
            sorted((sample.dataset_name, sample.sample_index) for sample in result),
            [("a", 0), ("a", 1), ("b", 0), ("b", 1), ("b", 2)],
        )


if __name__ == "__main__":
    unittest.main()
