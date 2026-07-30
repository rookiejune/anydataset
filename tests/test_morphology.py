from __future__ import annotations

import unittest

import torch
from anydataset.dataset.morphology import (
    Morphology,
    audio_collate,
    build_toy_audio_dataset,
    build_toy_speech_dataset,
    build_toy_speech_grid,
    speech_collate,
    speech_grid_collate,
)
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
)


class MorphologyContractTest(unittest.TestCase):
    def test_audio_collate_pads_waveforms(self) -> None:
        dataset = build_toy_audio_dataset(samples=2, seconds=0.01, sample_rate=16000)
        batch = audio_collate(dataset)
        self.assertEqual(batch.waveform.ndim, 3)
        self.assertEqual(tuple(batch.lengths.shape), (2,))

    def test_speech_collate_requires_text_not_speaker(self) -> None:
        dataset = build_toy_speech_dataset(samples=2, seconds=0.01, sample_rate=16000)
        batch = speech_collate(dataset)
        self.assertIsNone(batch.speaker_ids)
        self.assertEqual(len(batch.texts), 2)

    def test_speech_collate_keeps_optional_speaker(self) -> None:
        dataset = build_toy_speech_dataset(
            samples=2,
            seconds=0.01,
            sample_rate=16000,
            include_speaker=True,
        )
        batch = speech_collate(dataset)
        self.assertEqual(batch.speaker_ids, ("toy-speaker-0", "toy-speaker-1"))

    def test_speech_collate_rejects_missing_text(self) -> None:
        sample = {
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: (torch.zeros(1, 8), 16000)},
                meta={AudioMeta.SPEAKER_ID: "spk"},
            ),
        }
        with self.assertRaises(KeyError):
            speech_collate((sample,))

    def test_speech_grid_preserves_axes(self) -> None:
        view = build_toy_speech_grid()
        self.assertEqual(view.shape, (2, 2))
        full = view.full()
        self.assertEqual(full.shape, (1, 2, 2))
        self.assertEqual(full.waveforms.shape[:3], (1, 2, 2))
        self.assertEqual(full.speaker_ids, (("speaker-a", "speaker-b"),))
        self.assertEqual(full.texts, (("hello", "world"),))
        self.assertEqual(view.by_speaker("speaker-a").shape, (1, 1, 2))
        self.assertEqual(view.by_text(0).shape, (1, 2, 1))
        self.assertEqual(Morphology.SPEECH_GRID.name, "SPEECH_GRID")

    def test_speech_grid_collate_pads_independent_axes(self) -> None:
        left = build_toy_speech_grid(
            texts=("a",),
            speakers=("s0", "s1"),
            seconds=0.01,
        ).full()
        right = build_toy_speech_grid(
            texts=("x", "y", "z"),
            speakers=("s0",),
            seconds=0.02,
        ).full()
        batch = speech_grid_collate((left, right))
        self.assertEqual(batch.shape, (2, 2, 3))
        self.assertEqual(batch.speaker_ids, (("s0", "s1"), ("s0",)))
        self.assertEqual(batch.texts, (("a",), ("x", "y", "z")))
        self.assertEqual(int(batch.lengths[0, 1, 0].item()) > 0, True)
        self.assertEqual(int(batch.lengths[0, 0, 1].item()), 0)
        self.assertEqual(int(batch.lengths[1, 1, 0].item()), 0)

    def test_speech_grid_allows_unknown_axis_labels(self) -> None:
        from anydataset.dataset.morphology import SpeechGridBatch

        known = build_toy_speech_grid(
            texts=("hello", "world"),
            speakers=("speaker-a", "speaker-b"),
            seconds=0.01,
        ).full()
        unknown = SpeechGridBatch(
            waveforms=known.waveforms,
            lengths=known.lengths,
            speaker_ids=((None, "speaker-b"),),
            texts=(("hello", None),),
        )
        self.assertEqual(unknown.speaker_ids, ((None, "speaker-b"),))
        self.assertEqual(unknown.texts, (("hello", None),))
        batch = speech_grid_collate((known, unknown))
        self.assertEqual(
            batch.speaker_ids,
            (("speaker-a", "speaker-b"), (None, "speaker-b")),
        )
        self.assertEqual(
            batch.texts,
            (("hello", "world"), ("hello", None)),
        )


if __name__ == "__main__":
    unittest.main()
