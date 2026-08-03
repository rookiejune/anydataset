from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index as integer_index
from typing import Any, cast

import torch
from torch import Tensor

from ..._compat import strict_zip
from ...types import AudioView, TextItem, TextView
from ..speaker import SpeakerAudioBlock, SpeakerAudioGrid
from .types import SpeechGridBatch


@dataclass(frozen=True)
class SpeechGridView:
    """Axis-preserving views over a speaker x text grid; never flattens by default."""

    grid: SpeakerAudioGrid

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_speaker, n_text)`` for this single grid sample."""

        return len(self.speaker_ids), len(self.texts)

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        return self.grid.speaker_ids

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(
            _row_text(self.grid, index) for index in range(len(self.grid.row_specs))
        )

    def full(self, *, view: AudioView = AudioView.WAVEFORM) -> SpeechGridBatch:
        """Materialize the full rectangle as a batch of size 1."""

        return speech_grid_batch(self.grid.select().load(view=view))

    def by_speaker(
        self,
        speaker: str,
        *,
        view: AudioView = AudioView.WAVEFORM,
    ) -> SpeechGridBatch:
        """Materialize one speaker column as a batch of size 1."""

        return speech_grid_batch(self.grid.select(speaker=speaker).load(view=view))

    def by_text(
        self,
        text: int,
        *,
        view: AudioView = AudioView.WAVEFORM,
    ) -> SpeechGridBatch:
        """Materialize one text column as a batch of size 1."""

        row = self._text_row(text)
        return speech_grid_batch(
            self.grid.select(source=row.source_index, text=row.role).load(view=view)
        )

    def cell(
        self,
        *,
        text: int,
        speaker: str,
        view: AudioView = AudioView.WAVEFORM,
    ) -> SpeechGridBatch:
        """Materialize one cell as a batch of size 1."""

        row = self._text_row(text)
        return speech_grid_batch(
            self.grid.select(
                source=row.source_index,
                text=row.role,
                speaker=speaker,
            ).load(view=view)
        )

    def pairs(
        self,
        *,
        text: int,
        speakers: Sequence[str],
        view: AudioView = AudioView.WAVEFORM,
    ) -> tuple[SpeechGridBatch, ...]:
        """Materialize one cell per speaker for the same text row."""

        if len(speakers) < 2:
            raise ValueError("pairs requires at least two speakers.")
        return tuple(self.cell(text=text, speaker=speaker, view=view) for speaker in speakers)

    def _text_row(self, text: int):
        if isinstance(text, bool):
            raise TypeError("text must be a text row index.")
        try:
            index = integer_index(text)
        except TypeError as exc:
            raise TypeError("text must be a text row index.") from exc
        if index < 0:
            index += len(self.grid.row_specs)
        if index < 0 or index >= len(self.grid.row_specs):
            raise IndexError("text row index out of range.")
        return self.grid.row_specs[index]


def speech_grid_batch(block: SpeakerAudioBlock) -> SpeechGridBatch:
    """Convert a SpeakerAudioBlock into a batch-1 speaker x text grid."""

    # SpeakerAudioBlock stores [n_text, n_speaker, ...]; morphology uses speaker x text.
    waveforms = block.waveforms.transpose(0, 1).unsqueeze(0)
    lengths = block.lengths.transpose(0, 1).unsqueeze(0)
    return SpeechGridBatch(
        waveforms=waveforms,
        lengths=lengths,
        speaker_ids=(tuple(block.speaker_ids),),
        texts=(tuple(block.texts),),
    )


def speech_grid_collate(samples: Sequence[SpeechGridBatch]) -> SpeechGridBatch:
    """Pad independent grid samples into one ``SpeechGridBatch``.

    Each input may already be batched; samples are expanded along the batch axis.
    Speaker and text axes are padded independently per sample using length 0.
    Axis labels may contain ``None`` (unknown); correspondence is preserved.
    """

    if not samples:
        raise ValueError("speech_grid_collate requires at least one sample.")

    speaker_ids: list[tuple[str | None, ...]] = []
    texts: list[tuple[str | None, ...]] = []
    pieces: list[tuple[Tensor, Tensor]] = []
    for sample in samples:
        if not isinstance(sample, SpeechGridBatch):
            raise TypeError("speech_grid_collate expects SpeechGridBatch samples.")
        for batch_index, (speakers, sample_texts) in enumerate(
            strict_zip(sample.speaker_ids, sample.texts)
        ):
            n_speaker = len(speakers)
            n_text = len(sample_texts)
            pieces.append(
                (
                    sample.waveforms[batch_index, :n_speaker, :n_text],
                    sample.lengths[batch_index, :n_speaker, :n_text],
                )
            )
            speaker_ids.append(speakers)
            texts.append(sample_texts)

    max_speakers = max(waveform.shape[0] for waveform, _ in pieces)
    max_texts = max(waveform.shape[1] for waveform, _ in pieces)
    channels = pieces[0][0].shape[2]
    max_time = max(waveform.shape[-1] for waveform, _ in pieces)
    batch_size = len(pieces)
    device = pieces[0][0].device
    dtype = pieces[0][0].dtype
    waveforms = torch.zeros(
        (batch_size, max_speakers, max_texts, channels, max_time),
        device=device,
        dtype=dtype,
    )
    lengths = torch.zeros(
        (batch_size, max_speakers, max_texts),
        device=device,
        dtype=pieces[0][1].dtype,
    )
    for index, (waveform, sample_lengths) in enumerate(pieces):
        n_speaker, n_text, _, time = waveform.shape
        waveforms[index, :n_speaker, :n_text, :, :time] = waveform
        lengths[index, :n_speaker, :n_text] = sample_lengths
    return SpeechGridBatch(
        waveforms=waveforms,
        lengths=lengths,
        speaker_ids=tuple(speaker_ids),
        texts=tuple(texts),
    )


def build_toy_speech_grid(
    *,
    texts: Sequence[str] = ("hello", "world"),
    speakers: Sequence[str] = ("speaker-a", "speaker-b"),
    seconds: float = 0.25,
    sample_rate: int = 16000,
) -> SpeechGridView:
    """Build a deterministic in-memory speaker x text grid for smoke tests."""

    from ...types import (
        AudioItem,
        AudioMeta,
        AudioView as AV,
        Modality,
        Role as R,
        TextItem,
        TextMeta,
        TextView as TV,
    )
    from ..speaker import SpeakerAudioGrid, SpeakerAudioRow

    if not texts:
        raise ValueError("toy speech grid requires at least one text.")
    if not speakers:
        raise ValueError("toy speech grid requires at least one speaker.")
    if seconds <= 0:
        raise ValueError("toy seconds must be positive.")
    if sample_rate <= 0:
        raise ValueError("toy sample_rate must be positive.")

    steps = max(1, int(seconds * sample_rate))
    time = torch.linspace(0.0, seconds, steps=steps)
    cells: list[dict[Any, Any]] = []
    row_specs = []
    for text_index, text in enumerate(texts):
        row_specs.append(SpeakerAudioRow(source_index=text_index, role=R.DEFAULT))
        for speaker_index, speaker in enumerate(speakers):
            frequency = 220.0 + 55.0 * speaker_index + 10.0 * text_index
            waveform = torch.sin(2 * torch.pi * frequency * time).unsqueeze(0) * 0.05
            cells.append(
                {
                    (R.DEFAULT, Modality.AUDIO): AudioItem(
                        views={AV.WAVEFORM: (waveform, sample_rate)},
                        meta={AudioMeta.SPEAKER_ID: speaker},
                    ),
                    (R.DEFAULT, Modality.TEXT): TextItem(
                        views={
                            TV.TEXT: text,
                            TV.SPEAKERS: speaker,
                        },
                        meta={TextMeta.SOURCE_INDEX: text_index},
                    ),
                }
            )
    grid = SpeakerAudioGrid(
        cast(Any, cells),
        speakers,
        row_specs=row_specs,
    )
    return SpeechGridView(grid)


def _row_text(grid: SpeakerAudioGrid, index: int) -> str:
    sample = grid.rows[index]
    text_item = sample[grid.text_ref]
    if not isinstance(text_item, TextItem):
        raise TypeError("speech grid text row must provide TextItem.")
    value = text_item.views.get(TextView.TEXT)
    if not isinstance(value, str) or not value:
        raise ValueError("speech grid text must be a non-empty string.")
    return value
