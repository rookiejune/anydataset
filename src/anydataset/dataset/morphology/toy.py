from __future__ import annotations

import torch

from ...types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
    Sample,
    TextItem,
    TextView,
)


def build_toy_audio_dataset(*, samples: int, seconds: float, sample_rate: int) -> list[Sample]:
    """Create deterministic audio-only samples without speaker or text metadata."""

    if samples <= 0:
        raise ValueError("toy samples must be positive.")
    if seconds <= 0:
        raise ValueError("toy seconds must be positive.")
    if sample_rate <= 0:
        raise ValueError("toy sample_rate must be positive.")

    steps = max(1, int(seconds * sample_rate))
    time = torch.linspace(0.0, seconds, steps=steps)
    dataset: list[Sample] = []
    for index in range(samples):
        frequency = 220.0 + 55.0 * (index % 4)
        waveform = torch.sin(2 * torch.pi * frequency * time).unsqueeze(0) * 0.05
        dataset.append(
            {
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: (waveform, sample_rate)},
                    meta={},
                ),
            }
        )
    return dataset


def build_toy_speech_dataset(
    *,
    samples: int,
    seconds: float,
    sample_rate: int,
    include_speaker: bool = False,
) -> list[Sample]:
    """Create deterministic speech samples with required text.

    Speaker meta is optional. Prefer ``SpeechGrid`` when speaker contrast is the
    experimental axis.
    """

    if samples <= 0:
        raise ValueError("toy samples must be positive.")
    if seconds <= 0:
        raise ValueError("toy seconds must be positive.")
    if sample_rate <= 0:
        raise ValueError("toy sample_rate must be positive.")

    steps = max(1, int(seconds * sample_rate))
    time = torch.linspace(0.0, seconds, steps=steps)
    dataset: list[Sample] = []
    for index in range(samples):
        frequency = 220.0 + 55.0 * (index % 4)
        waveform = torch.sin(2 * torch.pi * frequency * time).unsqueeze(0) * 0.05
        audio_meta = (
            {AudioMeta.SPEAKER_ID: f"toy-speaker-{index % 2}"}
            if include_speaker
            else {}
        )
        dataset.append(
            {
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: (waveform, sample_rate)},
                    meta=audio_meta,
                ),
                (Role.DEFAULT, Modality.TEXT): TextItem(
                    views={TextView.TEXT: f"toy-text-{index}"},
                    meta={},
                ),
            }
        )
    return dataset
