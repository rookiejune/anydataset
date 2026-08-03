from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch
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

    def __post_init__(self) -> None:
        _padded_audio_axes(
            self.waveform,
            self.lengths,
            owner="AudioBatch",
            waveform_name="waveform",
            leading_axes=1,
            waveform_shape="[batch, channels, time]",
            lengths_shape="[batch]",
        )


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

    def __post_init__(self) -> None:
        (batch,) = _padded_audio_axes(
            self.waveform,
            self.lengths,
            owner="SpeechBatch",
            waveform_name="waveform",
            leading_axes=1,
            waveform_shape="[batch, channels, time]",
            lengths_shape="[batch]",
        )
        _axis_labels(self.texts, name="SpeechBatch texts", allow_none=False)
        if len(self.texts) != batch:
            raise ValueError("SpeechBatch texts must have length equal to batch.")
        if self.speaker_ids is None:
            return
        _axis_labels(
            self.speaker_ids,
            name="SpeechBatch speaker_ids",
            allow_none=False,
        )
        if len(self.speaker_ids) != batch:
            raise ValueError(
                "SpeechBatch speaker_ids must have length equal to batch when present."
            )


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
        batch, max_speakers, max_texts = _padded_audio_axes(
            self.waveforms,
            self.lengths,
            owner="SpeechGridBatch",
            waveform_name="waveforms",
            leading_axes=3,
            waveform_shape="[batch, n_speaker, n_text, channels, time]",
            lengths_shape="[batch, n_speaker, n_text]",
        )
        if not isinstance(self.speaker_ids, tuple):
            raise TypeError("SpeechGridBatch speaker_ids must be a tuple of axes.")
        if not isinstance(self.texts, tuple):
            raise TypeError("SpeechGridBatch texts must be a tuple of axes.")
        if len(self.speaker_ids) != batch or len(self.texts) != batch:
            raise ValueError("speaker_ids and texts must have length equal to batch.")
        for index, (speakers, sample_texts) in enumerate(
            strict_zip(self.speaker_ids, self.texts)
        ):
            _axis_labels(
                speakers,
                name=f"sample {index} speaker_ids",
                allow_none=True,
            )
            _axis_labels(
                sample_texts,
                name=f"sample {index} texts",
                allow_none=True,
            )
            if not speakers or not sample_texts:
                raise ValueError(
                    f"sample {index} must have non-empty speaker and text axes."
                )
            if len(speakers) > max_speakers or len(sample_texts) > max_texts:
                raise ValueError(
                    f"sample {index} axis labels exceed padded waveforms shape."
                )
            if bool(
                torch.any(self.lengths[index, len(speakers) :, :] != 0).item()
            ) or bool(
                torch.any(self.lengths[index, :, len(sample_texts) :] != 0).item()
            ):
                raise ValueError(
                    f"sample {index} lengths outside its labeled axes must be zero."
                )

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


def _padded_audio_axes(
    waveform: Tensor,
    lengths: Tensor,
    *,
    owner: str,
    waveform_name: str,
    leading_axes: int,
    waveform_shape: str,
    lengths_shape: str,
) -> tuple[int, ...]:
    if not isinstance(waveform, Tensor):
        raise TypeError(f"{owner} {waveform_name} must be a Tensor.")
    if not isinstance(lengths, Tensor):
        raise TypeError(f"{owner} lengths must be a Tensor.")
    if waveform.ndim != leading_axes + 2:
        raise ValueError(f"{owner} {waveform_name} must have shape {waveform_shape}.")
    leading_shape = tuple(int(size) for size in waveform.shape[:leading_axes])
    if tuple(lengths.shape) != leading_shape:
        raise ValueError(f"{owner} lengths must have shape {lengths_shape}.")
    if any(size <= 0 for size in waveform.shape):
        raise ValueError(f"{owner} {waveform_name} axes must be non-empty.")
    if lengths.dtype != torch.int64:
        raise TypeError(f"{owner} lengths must have dtype torch.int64.")
    if bool(torch.any(lengths < 0).item()):
        raise ValueError(f"{owner} lengths must be non-negative.")
    if bool(torch.any(lengths > waveform.shape[-1]).item()):
        raise ValueError(f"{owner} lengths must not exceed the padded time axis.")
    return leading_shape


def _axis_labels(
    labels: object,
    *,
    name: str,
    allow_none: bool,
) -> None:
    if not isinstance(labels, tuple):
        raise TypeError(f"{name} must be a tuple.")
    for label in labels:
        if isinstance(label, str) or (allow_none and label is None):
            continue
        expected = "str or None" if allow_none else "str"
        raise TypeError(f"{name} entries must be {expected}.")
