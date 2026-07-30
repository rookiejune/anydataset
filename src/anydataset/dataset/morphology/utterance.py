from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import torch
from torch import Tensor

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
from .types import AudioBatch, SpeechBatch

ItemT = TypeVar("ItemT")
AudioLoader = Callable[[str | Path], tuple[Tensor, int]]


def load_audio_file(path: str | Path) -> tuple[Tensor, int]:
    """Load a complete audio file as [channels, time] without clipping."""

    try:
        import torchaudio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading audio files requires the anydataset[audio] extra (torchaudio)."
        ) from exc
    waveform, sample_rate = torchaudio.load(str(path))
    return waveform, int(sample_rate)


def prepare_audio(
    audio: tuple[Tensor, int],
    *,
    sample_rate: int,
    channels: int,
) -> tuple[Tensor, int]:
    waveform, source_rate = audio
    source_rate = int(source_rate)
    waveform = _normalize_waveform(waveform, channels=channels)
    if source_rate != sample_rate:
        waveform = _resample(waveform, source_rate, sample_rate)
    return waveform.contiguous(), sample_rate


def audio_collate(
    samples: Sequence[Sample],
    *,
    audio_loader: AudioLoader = load_audio_file,
    sample_rate: int = 16000,
    channels: int = 1,
) -> AudioBatch:
    """Decode and pad audio-only utterances."""

    if not samples:
        raise ValueError("Cannot collate an empty audio batch.")
    prepared = tuple(
        _prepare_audio_sample(
            sample,
            audio_loader=audio_loader,
            sample_rate=sample_rate,
            channels=channels,
        )
        for sample in samples
    )
    waveform, lengths = _pad_waveforms(prepared)
    return AudioBatch(waveform=waveform, lengths=lengths)


def speech_collate(
    samples: Sequence[Sample],
    *,
    audio_loader: AudioLoader = load_audio_file,
    sample_rate: int = 16000,
    channels: int = 1,
) -> SpeechBatch:
    """Decode and pad speech utterances; text is required, speaker is optional meta."""

    if not samples:
        raise ValueError("Cannot collate an empty speech batch.")
    prepared = tuple(
        _prepare_speech_sample(
            sample,
            audio_loader=audio_loader,
            sample_rate=sample_rate,
            channels=channels,
        )
        for sample in samples
    )
    waveform, lengths = _pad_waveforms(tuple(item.waveform for item in prepared))
    return SpeechBatch(
        waveform=waveform,
        lengths=lengths,
        texts=tuple(item.text for item in prepared),
        speaker_ids=_optional_speaker_ids(tuple(item.speaker_id for item in prepared)),
    )


@dataclass(frozen=True)
class _PreparedSpeech:
    waveform: Tensor
    text: str
    speaker_id: str | None


def _prepare_audio_sample(
    sample: Sample,
    *,
    audio_loader: AudioLoader,
    sample_rate: int,
    channels: int,
) -> Tensor:
    audio_item = _sample_item(sample, (Role.DEFAULT, Modality.AUDIO), AudioItem)
    path = _audio_path(audio_item)
    loaded_waveform, source_sample_rate = _load_audio(
        audio_item, path=path, audio_loader=audio_loader
    )
    waveform, _ = prepare_audio(
        (loaded_waveform, source_sample_rate),
        sample_rate=sample_rate,
        channels=channels,
    )
    return waveform


def _prepare_speech_sample(
    sample: Sample,
    *,
    audio_loader: AudioLoader,
    sample_rate: int,
    channels: int,
) -> _PreparedSpeech:
    audio_item = _sample_item(sample, (Role.DEFAULT, Modality.AUDIO), AudioItem)
    text_item = _sample_item(sample, (Role.DEFAULT, Modality.TEXT), TextItem)
    path = _audio_path(audio_item)
    loaded_waveform, source_sample_rate = _load_audio(
        audio_item, path=path, audio_loader=audio_loader
    )
    waveform, _ = prepare_audio(
        (loaded_waveform, source_sample_rate),
        sample_rate=sample_rate,
        channels=channels,
    )
    return _PreparedSpeech(
        waveform=waveform,
        text=_text(text_item),
        speaker_id=_optional_speaker_id(audio_item),
    )


def _optional_speaker_ids(speakers: Sequence[str | None]) -> tuple[str, ...] | None:
    present = tuple(speaker is not None for speaker in speakers)
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "Speech batch speaker meta must be present on every sample or on none."
        )
    return tuple(str(speaker) for speaker in speakers)


def _optional_speaker_id(audio: AudioItem) -> str | None:
    value = audio.meta.get(AudioMeta.SPEAKER_ID)
    if value is None:
        return None
    return str(value)


def _sample_item(
    sample: Mapping[object, object],
    ref: tuple[Role, Modality],
    item_type: type[ItemT],
) -> ItemT:
    try:
        value = sample[ref]
    except KeyError as exc:
        raise KeyError(f"Sample requires {ref!r}.") from exc
    if not isinstance(value, item_type):
        raise TypeError(f"Sample {ref!r} must be {item_type.__name__}.")
    return value


def _audio_path(audio: AudioItem) -> str | None:
    if AudioView.FILE not in audio.views:
        return None
    return str(audio.views[AudioView.FILE])


def _load_audio(
    audio: AudioItem,
    *,
    path: str | None,
    audio_loader: AudioLoader,
) -> tuple[Tensor, int]:
    if AudioView.WAVEFORM in audio.views:
        waveform, sample_rate = audio.views[AudioView.WAVEFORM]
        return waveform, int(sample_rate)
    if path is None:
        raise ValueError("Audio sample requires waveform or file view.")
    waveform, sample_rate = audio_loader(Path(path))
    return waveform, int(sample_rate)


def _text(text_item: TextItem) -> str:
    try:
        value = text_item.views[TextView.TEXT]
    except KeyError as exc:
        raise KeyError("Speech sample requires TextView.TEXT.") from exc
    if not isinstance(value, str) or not value:
        raise ValueError("Speech sample TextView.TEXT must be a non-empty string.")
    return value


def _pad_waveforms(waveforms: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    channels = waveforms[0].shape[0]
    if any(waveform.ndim != 2 for waveform in waveforms):
        raise ValueError("prepared waveforms must have shape [channels, time].")
    if any(waveform.shape[0] != channels for waveform in waveforms):
        raise ValueError("prepared waveforms must share the same channel count.")

    lengths = torch.tensor([waveform.shape[-1] for waveform in waveforms], dtype=torch.long)
    max_length = int(lengths.max().item())
    batch = waveforms[0].new_zeros((len(waveforms), channels, max_length))
    for index, waveform in enumerate(waveforms):
        batch[index, :, : waveform.shape[-1]] = waveform
    return batch, lengths


def _normalize_waveform(waveform: Tensor, *, channels: int) -> Tensor:
    if channels <= 0:
        raise ValueError("channels must be positive.")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("loaded waveform must have shape [channels, time].")
    if waveform.shape[-1] <= 0:
        raise ValueError("loaded waveform must have a positive length.")

    waveform = waveform.float()
    if waveform.shape[0] == channels:
        return waveform
    if channels == 1:
        return waveform.mean(dim=0, keepdim=True)
    raise ValueError("loaded waveform channel count does not match target channels.")


def _resample(waveform: Tensor, source_rate: int, target_rate: int) -> Tensor:
    try:
        import torchaudio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Resampling audio requires the anydataset[audio] extra (torchaudio)."
        ) from exc
    return torchaudio.functional.resample(waveform, source_rate, target_rate)
