from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from torch import Tensor

from ..._compat import strict_zip


class Morphology(Enum):
    """Top-level sample shape for audio-task loaders."""

    AUDIO = auto()
    SPEECH = auto()
    SPEECH_GRID = auto()


@dataclass(eq=False)
class AudioBatch:
    """Padded generic audio utterances without speaker or text identity."""

    waveform: Tensor
    lengths: Tensor


@dataclass(eq=False)
class SpeechBatch:
    """Padded speech utterances with required text; speaker is optional meta.

    Speaker identity for contrastive / swap work belongs on ``SpeechGridBatch``.
    When present, ``speaker_ids`` must cover every row; otherwise it is ``None``.
    """

    waveform: Tensor
    lengths: Tensor
    texts: tuple[str, ...]
    speaker_ids: tuple[str, ...] | None = None


@dataclass(eq=False)
class SpeechGridBatch:
    """Batched speaker x text grids; each sample owns its axis labels.

    ``waveforms`` is ``[batch, n_speaker, n_text, channels, time]`` after padding
    within the batch. ``speaker_ids[b]`` / ``texts[b]`` label the live axes for
    sample ``b``; pad cells use length 0. A label entry may be ``None`` for
    unknown identity while keeping axis correspondence. Axis position is the
    index — no parallel index / role fields. Do not flatten into ``SpeechBatch``.
    """

    waveforms: Tensor
    lengths: Tensor
    speaker_ids: tuple[tuple[str | None, ...], ...]
    texts: tuple[tuple[str | None, ...], ...]

    def __post_init__(self) -> None:
        if self.waveforms.ndim != 5:
            raise ValueError(
                "SpeechGridBatch waveforms must have shape "
                "[batch, n_speaker, n_text, channels, time]."
            )
        if tuple(self.lengths.shape) != tuple(self.waveforms.shape[:3]):
            raise ValueError(
                "SpeechGridBatch lengths must have shape [batch, n_speaker, n_text]."
            )
        batch = int(self.waveforms.shape[0])
        if len(self.speaker_ids) != batch or len(self.texts) != batch:
            raise ValueError("speaker_ids and texts must have length equal to batch.")
        max_speakers = int(self.waveforms.shape[1])
        max_texts = int(self.waveforms.shape[2])
        for index, (speakers, sample_texts) in enumerate(
            strict_zip(self.speaker_ids, self.texts)
        ):
            if not speakers or not sample_texts:
                raise ValueError(
                    f"sample {index} must have non-empty speaker and text axes."
                )
            if len(speakers) > max_speakers or len(sample_texts) > max_texts:
                raise ValueError(
                    f"sample {index} axis labels exceed padded waveforms shape."
                )
            _axis_labels(speakers, name=f"sample {index} speaker_ids")
            _axis_labels(sample_texts, name=f"sample {index} texts")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Padded ``(batch, n_speaker, n_text)``; live extents are label lengths."""

        return (
            int(self.waveforms.shape[0]),
            int(self.waveforms.shape[1]),
            int(self.waveforms.shape[2]),
        )

    @property
    def batch_size(self) -> int:
        return int(self.waveforms.shape[0])


def _axis_labels(labels: tuple[str | None, ...], *, name: str) -> None:
    for label in labels:
        if label is not None and not isinstance(label, str):
            raise TypeError(f"{name} entries must be str or None.")
