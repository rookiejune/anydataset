import unittest

from anydataset.dataset import (
    SpeakerCartesianDataset,
    SpeakerIdDataset,
    speaker_cartesian_indexes,
    speaker_for_index,
)
from anydataset.types import Modality, Role, TextItem, TextMeta, TextView


class SpeakerIdDatasetTest(unittest.TestCase):
    def test_aligned_mode_adds_speaker_id_to_text_meta(self):
        dataset = [
            {(Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "hello"})},
            {(Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "world"})},
        ]

        wrapped = SpeakerIdDataset(dataset, ("Vivian", "Ryan"))

        first = wrapped[0][(Role.DEFAULT, Modality.TEXT)]
        second = wrapped[1][(Role.DEFAULT, Modality.TEXT)]
        self.assertIsInstance(first, TextItem)
        self.assertIsInstance(second, TextItem)
        assert isinstance(first, TextItem)
        assert isinstance(second, TextItem)
        self.assertEqual(first.views[TextView.SPEAKERS], "Vivian")
        self.assertEqual(second.views[TextView.SPEAKERS], "Ryan")
        self.assertEqual(first.views[TextView.TEXT], "hello")

    def test_cycle_mode_reuses_speaker_ids(self):
        self.assertEqual(speaker_for_index(0, ("Vivian", "Ryan"), "cycle"), "Vivian")
        self.assertEqual(speaker_for_index(1, ("Vivian", "Ryan"), "cycle"), "Ryan")
        self.assertEqual(speaker_for_index(2, ("Vivian", "Ryan"), "cycle"), "Vivian")

    def test_aligned_mode_requires_matching_lengths(self):
        dataset = [
            {(Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "hello"})}
        ]

        with self.assertRaisesRegex(ValueError, "match dataset length"):
            SpeakerIdDataset(dataset, ("Vivian", "Ryan"))

    def test_cartesian_dataset_repeats_each_text_for_all_speakers(self):
        dataset = [
            {(Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "hello"})},
            {(Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "world"})},
        ]

        wrapped = SpeakerCartesianDataset(dataset, ("Vivian", "Ryan"))

        self.assertEqual(len(wrapped), 4)
        first = wrapped[0][Role.DEFAULT, Modality.TEXT]
        second = wrapped[1][Role.DEFAULT, Modality.TEXT]
        third = wrapped[2][Role.DEFAULT, Modality.TEXT]
        assert isinstance(first, TextItem)
        assert isinstance(second, TextItem)
        assert isinstance(third, TextItem)
        self.assertEqual(first.views[TextView.TEXT], "hello")
        self.assertEqual(first.views[TextView.SPEAKERS], "Vivian")
        self.assertEqual(first.meta[TextMeta.SOURCE_INDEX], 0)
        self.assertEqual(second.views[TextView.SPEAKERS], "Ryan")
        self.assertEqual(second.meta[TextMeta.SOURCE_INDEX], 0)
        self.assertEqual(third.views[TextView.TEXT], "world")
        self.assertEqual(third.meta[TextMeta.SOURCE_INDEX], 1)

    def test_cartesian_index_helper_maps_flat_index(self):
        self.assertEqual(speaker_cartesian_indexes(0, 3), (0, 0))
        self.assertEqual(speaker_cartesian_indexes(2, 3), (0, 2))
        self.assertEqual(speaker_cartesian_indexes(3, 3), (1, 0))


if __name__ == "__main__":
    unittest.main()
