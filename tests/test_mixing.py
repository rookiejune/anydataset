import unittest

from anydatasets.mixing import SampleStream, WeightedDatasetMixer
from anydatasets.samples import Sample


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


if __name__ == "__main__":
    unittest.main()
