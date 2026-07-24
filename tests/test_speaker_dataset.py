import pickle
import unittest
from typing import Optional

import torch

from anydataset.dataset import (
    GroupedSpeakerAudioDataset,
    SpeakerAssignment,
    SpeakerCartesianDataset,
    SpeakerIdDataset,
    speaker_cartesian_indexes,
    speaker_for_index,
)
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)


class SpeakerIdDatasetTest(unittest.TestCase):
    def test_aligned_mode_adds_speaker_view(self):
        text_ref = (Role.DEFAULT, Modality.TEXT)
        dataset = [
            {text_ref: TextItem(views={TextView.TEXT: "hello"})},
            {text_ref: TextItem(views={TextView.TEXT: "world"})},
        ]
        wrapped = SpeakerIdDataset(
            dataset,
            {text_ref: SpeakerAssignment(("Vivian", "Ryan"))},
        )

        first = wrapped[0][text_ref]
        second = wrapped[1][text_ref]

        self.assertIsInstance(first, TextItem)
        self.assertIsInstance(second, TextItem)
        assert isinstance(first, TextItem)
        assert isinstance(second, TextItem)
        self.assertEqual(first.views[TextView.SPEAKERS], "Vivian")
        self.assertEqual(second.views[TextView.SPEAKERS], "Ryan")
        self.assertEqual(first.views[TextView.TEXT], "hello")

    def test_assignments_support_multiple_text_references(self):
        source_ref = (Role.SOURCE, Modality.TEXT)
        target_ref = (Role.TARGET, Modality.TEXT)
        dataset = [
            {
                source_ref: TextItem(views={TextView.TEXT: "你好"}),
                target_ref: TextItem(views={TextView.TEXT: "hello"}),
            },
            {
                source_ref: TextItem(views={TextView.TEXT: "世界"}),
                target_ref: TextItem(views={TextView.TEXT: "world"}),
            },
            {
                source_ref: TextItem(views={TextView.TEXT: "再见"}),
                target_ref: TextItem(views={TextView.TEXT: "goodbye"}),
            },
        ]
        wrapped = SpeakerIdDataset(
            dataset,
            {
                source_ref: SpeakerAssignment(("Vivian", "Ryan", "Aiden")),
                target_ref: SpeakerAssignment(("Ryan", "Vivian"), mode="cycle"),
            },
        )

        first = wrapped[0]
        second = wrapped[1]
        third = wrapped[2]

        self.assertEqual(first[source_ref].views[TextView.SPEAKERS], "Vivian")
        self.assertEqual(first[target_ref].views[TextView.SPEAKERS], "Ryan")
        self.assertEqual(second[source_ref].views[TextView.SPEAKERS], "Ryan")
        self.assertEqual(second[target_ref].views[TextView.SPEAKERS], "Vivian")
        self.assertEqual(third[source_ref].views[TextView.SPEAKERS], "Aiden")
        self.assertEqual(third[target_ref].views[TextView.SPEAKERS], "Ryan")

    def test_cycle_mode_reuses_speaker_ids(self):
        self.assertEqual(speaker_for_index(0, ("Vivian", "Ryan"), "cycle"), "Vivian")
        self.assertEqual(speaker_for_index(1, ("Vivian", "Ryan"), "cycle"), "Ryan")
        self.assertEqual(speaker_for_index(2, ("Vivian", "Ryan"), "cycle"), "Vivian")

    def test_speaker_for_index_validates_mode_before_empty_speakers(self):
        with self.assertRaisesRegex(ValueError, "mode must be"):
            speaker_for_index(0, (), "invalid")

    def test_aligned_mode_requires_matching_lengths(self):
        text_ref = (Role.DEFAULT, Modality.TEXT)
        dataset = [{text_ref: TextItem(views={TextView.TEXT: "hello"})}]

        with self.assertRaisesRegex(ValueError, "match dataset length"):
            SpeakerIdDataset(
                dataset,
                {text_ref: SpeakerAssignment(("Vivian", "Ryan"))},
            )

    def test_assignments_must_not_be_empty(self):
        with self.assertRaisesRegex(ValueError, "non-empty mapping"):
            SpeakerIdDataset([], {})

    def test_cycle_speaker_ids_must_not_be_empty(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            SpeakerAssignment((), mode="cycle")

    def test_empty_aligned_assignment_supports_empty_dataset(self):
        wrapped = SpeakerIdDataset(
            [],
            {(Role.DEFAULT, Modality.TEXT): SpeakerAssignment(())},
        )

        self.assertEqual(len(wrapped), 0)

    def test_existing_speaker_must_match_assignment(self):
        text_ref = (Role.DEFAULT, Modality.TEXT)
        dataset = [
            {
                text_ref: TextItem(
                    views={
                        TextView.TEXT: "hello",
                        TextView.SPEAKERS: "Vivian",
                    }
                )
            }
        ]
        wrapped = SpeakerIdDataset(
            dataset,
            {text_ref: SpeakerAssignment(("Ryan",))},
        )

        with self.assertRaisesRegex(ValueError, "already has speaker id 'Vivian'"):
            _ = wrapped[0]

    def test_assignment_wrapper_is_picklable(self):
        text_ref = (Role.DEFAULT, Modality.TEXT)
        wrapped = SpeakerIdDataset(
            [{text_ref: TextItem(views={TextView.TEXT: "hello"})}],
            {text_ref: SpeakerAssignment(("Vivian",))},
        )

        restored = pickle.loads(pickle.dumps(wrapped))

        item = restored[0][text_ref]
        assert isinstance(item, TextItem)
        self.assertEqual(item.views[TextView.SPEAKERS], "Vivian")

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

    def test_grouped_audio_restores_text_axis(self):
        dataset = [
            _flat_sample("hello", "Vivian", 0, torch.tensor([[1.0, 2.0]])),
            _flat_sample("hello", "Ryan", 0, torch.tensor([[3.0]])),
            _flat_sample("world", "Vivian", 1, torch.tensor([[4.0]])),
            _flat_sample("world", "Ryan", 1, torch.tensor([[5.0, 6.0]])),
        ]

        grouped = GroupedSpeakerAudioDataset(dataset, ("Vivian", "Ryan"))
        sample = grouped[0]
        text = sample[Role.DEFAULT, Modality.TEXT]
        audio = sample[Role.DEFAULT, Modality.AUDIO]
        assert isinstance(text, TextItem)
        assert isinstance(audio, AudioItem)

        self.assertEqual(len(grouped), 2)
        self.assertEqual(text.views, {TextView.TEXT: "hello"})
        self.assertEqual(text.meta[TextMeta.SOURCE_INDEX], 0)
        self.assertEqual(audio.views[AudioView.SPEAKERS], ("Vivian", "Ryan"))
        self.assertTrue(
            torch.equal(audio.views[AudioView.SPEAKER_LENGTHS], torch.tensor([2, 1]))
        )
        waveform, sample_rate = audio.views[AudioView.WAVEFORM]
        self.assertEqual(sample_rate, 24_000)
        self.assertTrue(
            torch.equal(waveform, torch.tensor([[[1.0, 2.0]], [[3.0, 0.0]]]))
        )
        self.assertNotIn(AudioMeta.SPEAKER_ID, audio.meta)
        second_text = grouped[1][Role.DEFAULT, Modality.TEXT]
        assert isinstance(second_text, TextItem)
        self.assertEqual(second_text.views[TextView.TEXT], "world")

    def test_grouped_audio_requires_complete_speaker_groups(self):
        dataset = [
            _flat_sample("hello", "Vivian", 0, torch.tensor([[1.0]])),
        ]

        with self.assertRaisesRegex(ValueError, "divisible by speaker count"):
            GroupedSpeakerAudioDataset(dataset, ("Vivian", "Ryan"))

    def test_grouped_audio_rejects_mismatched_text_content(self):
        dataset = [
            _flat_sample("hello", "Vivian", 0, torch.tensor([[1.0]])),
            _flat_sample("different", "Ryan", 0, torch.tensor([[2.0]])),
        ]
        grouped = GroupedSpeakerAudioDataset(dataset, ("Vivian", "Ryan"))

        with self.assertRaisesRegex(ValueError, "text content differs"):
            _ = grouped[0]

    def test_grouped_audio_rejects_mismatched_source_index(self):
        dataset = [
            _flat_sample("hello", "Vivian", 0, torch.tensor([[1.0]])),
            _flat_sample("hello", "Ryan", 1, torch.tensor([[2.0]])),
        ]
        grouped = GroupedSpeakerAudioDataset(dataset, ("Vivian", "Ryan"))

        with self.assertRaisesRegex(ValueError, "expected 0"):
            _ = grouped[0]

    def test_grouped_audio_rejects_mismatched_speaker_order(self):
        dataset = [
            _flat_sample("hello", "Ryan", 0, torch.tensor([[1.0]])),
            _flat_sample("hello", "Vivian", 0, torch.tensor([[2.0]])),
        ]
        grouped = GroupedSpeakerAudioDataset(dataset, ("Vivian", "Ryan"))

        with self.assertRaisesRegex(ValueError, "expected 'Vivian'"):
            _ = grouped[0]

    def test_grouped_audio_rejects_mismatched_audio_speaker(self):
        dataset = [
            _flat_sample(
                "hello",
                "Vivian",
                0,
                torch.tensor([[1.0]]),
                audio_speaker_id="Ryan",
            )
        ]
        grouped = GroupedSpeakerAudioDataset(dataset, ("Vivian",))

        with self.assertRaisesRegex(ValueError, "audio speaker 'Ryan'"):
            _ = grouped[0]

    def test_grouped_audio_allows_missing_audio_speaker_meta(self):
        dataset = [
            _flat_sample(
                "hello",
                "Vivian",
                0,
                torch.tensor([[1.0]]),
                include_audio_speaker_id=False,
            )
        ]
        grouped = GroupedSpeakerAudioDataset(dataset, ("Vivian",))

        audio = grouped[0][Role.DEFAULT, Modality.AUDIO]

        assert isinstance(audio, AudioItem)
        self.assertEqual(audio.views[AudioView.SPEAKERS], ("Vivian",))

    def test_grouped_audio_rejects_invalid_sample_rate(self):
        dataset = [
            _flat_sample("hello", "Vivian", 0, torch.tensor([[1.0]]), sample_rate=0)
        ]
        grouped = GroupedSpeakerAudioDataset(dataset, ("Vivian",))

        with self.assertRaisesRegex(ValueError, "sample rate must be positive"):
            _ = grouped[0]

    def test_grouped_audio_rejects_mismatched_sample_rates(self):
        dataset = [
            _flat_sample("hello", "Vivian", 0, torch.tensor([[1.0]])),
            _flat_sample(
                "hello",
                "Ryan",
                0,
                torch.tensor([[2.0]]),
                sample_rate=16_000,
            ),
        ]
        grouped = GroupedSpeakerAudioDataset(dataset, ("Vivian", "Ryan"))

        with self.assertRaisesRegex(ValueError, "share one sample rate"):
            _ = grouped[0]

    def test_grouped_audio_rejects_non_2d_waveform(self):
        dataset = [
            _flat_sample("hello", "Vivian", 0, torch.tensor([1.0, 2.0]))
        ]
        grouped = GroupedSpeakerAudioDataset(dataset, ("Vivian",))

        with self.assertRaisesRegex(ValueError, r"shape \[channel, time\]"):
            _ = grouped[0]

    def test_grouped_audio_rejects_mismatched_channels(self):
        dataset = [
            _flat_sample("hello", "Vivian", 0, torch.ones(1, 2)),
            _flat_sample("hello", "Ryan", 0, torch.ones(2, 2)),
        ]
        grouped = GroupedSpeakerAudioDataset(dataset, ("Vivian", "Ryan"))

        with self.assertRaisesRegex(ValueError, "expected prefix shape"):
            _ = grouped[0]

    def test_cartesian_index_helper_maps_flat_index(self):
        self.assertEqual(speaker_cartesian_indexes(0, 3), (0, 0))
        self.assertEqual(speaker_cartesian_indexes(2, 3), (0, 2))
        self.assertEqual(speaker_cartesian_indexes(3, 3), (1, 0))


def _flat_sample(
    text: str,
    speaker_id: str,
    source_index: int,
    waveform: torch.Tensor,
    *,
    sample_rate: int = 24_000,
    audio_speaker_id: Optional[str] = None,
    include_audio_speaker_id: bool = True,
):
    audio_meta = (
        {
            AudioMeta.SPEAKER_ID: speaker_id
            if audio_speaker_id is None
            else audio_speaker_id
        }
        if include_audio_speaker_id
        else {}
    )
    return {
        (Role.DEFAULT, Modality.TEXT): TextItem(
            views={TextView.TEXT: text, TextView.SPEAKERS: speaker_id},
            meta={TextMeta.SOURCE_INDEX: source_index},
        ),
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (waveform, sample_rate)},
            meta=audio_meta,
        ),
    }


if __name__ == "__main__":
    unittest.main()
