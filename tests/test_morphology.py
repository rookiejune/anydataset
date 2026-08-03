from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import torch
from anydataset.dataset.morphology import (
    AudioBatch,
    Morphology,
    SpeechBatch,
    SpeechGridBatch,
    audio_collate,
    build_toy_audio_dataset,
    build_toy_speech_dataset,
    build_toy_speech_grid,
    speech_collate,
    speech_grid_batch,
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
    def test_audio_batch_validates_padded_tensor_contract(self) -> None:
        waveform = torch.zeros(2, 1, 4)
        AudioBatch(waveform=waveform, lengths=torch.tensor([4, 3]))

        with self.assertRaisesRegex(TypeError, "waveform must be a Tensor"):
            AudioBatch(waveform=[[0.0]], lengths=torch.tensor([1]))
        with self.assertRaisesRegex(ValueError, r"\[batch, channels, time\]"):
            AudioBatch(waveform=torch.zeros(2, 4), lengths=torch.tensor([4, 4]))
        with self.assertRaisesRegex(ValueError, r"lengths must have shape \[batch\]"):
            AudioBatch(waveform=waveform, lengths=torch.tensor([[4, 3]]))
        with self.assertRaisesRegex(TypeError, "dtype torch.int64"):
            AudioBatch(waveform=waveform, lengths=torch.tensor([4.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            AudioBatch(waveform=waveform, lengths=torch.tensor([4, -1]))
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            AudioBatch(waveform=waveform, lengths=torch.tensor([5, 3]))

    def test_speech_batch_validates_text_and_speaker_axes(self) -> None:
        waveform = torch.zeros(2, 1, 4)
        lengths = torch.tensor([4, 3])
        SpeechBatch(
            waveform=waveform,
            lengths=lengths,
            texts=("hello", "world"),
            speaker_ids=("a", "b"),
        )

        with self.assertRaisesRegex(TypeError, "texts must be a tuple"):
            SpeechBatch(waveform=waveform, lengths=lengths, texts=["hello", "world"])
        with self.assertRaisesRegex(
            ValueError, "texts must have length equal to batch"
        ):
            SpeechBatch(waveform=waveform, lengths=lengths, texts=("hello",))
        with self.assertRaisesRegex(TypeError, "texts entries must be str"):
            SpeechBatch(waveform=waveform, lengths=lengths, texts=("hello", None))
        with self.assertRaisesRegex(ValueError, "speaker_ids must have length"):
            SpeechBatch(
                waveform=waveform,
                lengths=lengths,
                texts=("hello", "world"),
                speaker_ids=("a",),
            )

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
        self.assertEqual(full.sample_rate, 16000)
        self.assertEqual(full.speaker_ids, (("speaker-a", "speaker-b"),))
        self.assertEqual(full.texts, (("hello", "world"),))
        self.assertEqual(view.by_speaker("speaker-a").shape, (1, 1, 2))
        self.assertEqual(view.by_text(0).shape, (1, 2, 1))
        self.assertEqual(view.by_text(-1).texts, (("world",),))
        self.assertEqual(Morphology.SPEECH_GRID.name, "SPEECH_GRID")

    def test_speech_grid_text_index_requires_an_integer(self) -> None:
        view = build_toy_speech_grid()

        for text in (True, 0.0, 1.9, "0"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(TypeError, "text row index"):
                    view.by_text(text)

    def test_speech_grid_view_is_waveform_only(self) -> None:
        view = build_toy_speech_grid()

        with self.assertRaises(TypeError):
            view.full(view=AudioView.LONGCAT)
        block = mock.Mock()
        block.audio_view = AudioView.LONGCAT
        with self.assertRaisesRegex(ValueError, "waveform audio block"):
            speech_grid_batch(block)

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
        self.assertEqual(batch.sample_rate, 16000)
        self.assertEqual(batch.shape, (2, 2, 3))
        self.assertEqual(batch.speaker_ids, (("s0", "s1"), ("s0",)))
        self.assertEqual(batch.texts, (("a",), ("x", "y", "z")))
        self.assertEqual(int(batch.lengths[0, 1, 0].item()) > 0, True)
        self.assertEqual(int(batch.lengths[0, 0, 1].item()), 0)
        self.assertEqual(int(batch.lengths[1, 1, 0].item()), 0)

    def test_speech_grid_allows_unknown_axis_labels(self) -> None:
        known = build_toy_speech_grid(
            texts=("hello", "world"),
            speakers=("speaker-a", "speaker-b"),
            seconds=0.01,
        ).full()
        unknown = SpeechGridBatch(
            waveforms=known.waveforms,
            lengths=known.lengths,
            sample_rate=known.sample_rate,
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

    def test_speech_grid_collate_requires_uniform_audio_contract(self) -> None:
        grid = build_toy_speech_grid(seconds=0.01).full()

        with self.assertRaisesRegex(ValueError, "uniform sample_rate"):
            speech_grid_collate((grid, replace(grid, sample_rate=8000)))

        stereo = replace(
            grid,
            waveforms=grid.waveforms.repeat(1, 1, 1, 2, 1),
        )
        with self.assertRaisesRegex(ValueError, "uniform channel counts"):
            speech_grid_collate((grid, stereo))

        float64 = replace(grid, waveforms=grid.waveforms.to(torch.float64))
        with self.assertRaisesRegex(TypeError, "uniform waveform dtypes"):
            speech_grid_collate((grid, float64))

    def test_speech_grid_validates_nested_axes_and_padding(self) -> None:
        waveforms = torch.zeros(1, 2, 2, 1, 4)
        lengths = torch.tensor([[[4, 0], [3, 0]]])
        SpeechGridBatch(
            waveforms=waveforms,
            lengths=lengths,
            sample_rate=16000,
            speaker_ids=(("a", "b"),),
            texts=(("hello",),),
        )

        with self.assertRaisesRegex(TypeError, "sample_rate must be an integer"):
            SpeechGridBatch(
                waveforms=waveforms,
                lengths=lengths,
                sample_rate=True,
                speaker_ids=(("a", "b"),),
                texts=(("hello",),),
            )
        with self.assertRaisesRegex(ValueError, "sample_rate must be positive"):
            SpeechGridBatch(
                waveforms=waveforms,
                lengths=lengths,
                sample_rate=0,
                speaker_ids=(("a", "b"),),
                texts=(("hello",),),
            )

        with self.assertRaisesRegex(TypeError, "speaker_ids must be a tuple"):
            SpeechGridBatch(
                waveforms=waveforms,
                lengths=lengths,
                sample_rate=16000,
                speaker_ids=(["a", "b"],),
                texts=(("hello",),),
            )
        invalid_lengths = lengths.clone()
        invalid_lengths[0, 0, 1] = 1
        with self.assertRaisesRegex(ValueError, "outside its labeled axes"):
            SpeechGridBatch(
                waveforms=waveforms,
                lengths=invalid_lengths,
                sample_rate=16000,
                speaker_ids=(("a", "b"),),
                texts=(("hello",),),
            )


if __name__ == "__main__":
    unittest.main()
