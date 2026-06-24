from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ...samples import Sample
from ..base import SampleFormatter
from ...modalities.audio import AudioView
from .schema import (
    AudioKey,
    AudioOptKey,
    ModalityKey,
    TextKey,
    TextOptKey,
)


class AudioCodecFormatter(SampleFormatter):
    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int | None = 2,
        max_clip_seconds: float | None = 8.0,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if channels is not None and channels <= 0:
            raise ValueError("channels must be positive or None.")
        if max_clip_seconds is not None and max_clip_seconds <= 0:
            raise ValueError("max_clip_seconds must be positive or None.")

        self.sample_rate = sample_rate
        self.channels = channels
        self.max_clip_seconds = max_clip_seconds

    def __call__(self, sample: Sample) -> Sample:
        waveform_value, sample_rate = self._extract_waveform(sample.data)
        waveform = self._to_waveform_tensor(waveform_value)
        if waveform.shape[-1] == 0:
            raise ValueError("Audio samples cannot contain an empty waveform.")

        waveform = self._resample(waveform, sample_rate)
        waveform = self._match_channels(waveform)
        waveform = self._truncate(waveform)

        data = dict(sample.data)
        audio = dict(data[ModalityKey.AUDIO])
        views = dict(audio[AudioKey.VIEWS])
        views[AudioView.WAVEFORM] = waveform.contiguous()
        audio[AudioKey.VIEWS] = views
        audio[AudioKey.SAMPLE_RATE] = self.sample_rate
        audio[AudioOptKey.DURATION] = waveform.shape[-1] / self.sample_rate
        data[ModalityKey.AUDIO] = audio
        text = self._extract_text(sample.data)
        if text is not None:
            data[ModalityKey.TEXT] = text
        return Sample(
            data=data,
            dataset_name=sample.dataset_name,
            sample_index=sample.sample_index,
        )

    def _extract_waveform(self, row: Mapping[str, Any]) -> tuple[Any, int | None]:
        if ModalityKey.AUDIO not in row:
            raise KeyError(f"Audio codec samples must include `{ModalityKey.AUDIO}`.")
        audio = row[ModalityKey.AUDIO]
        if not isinstance(audio, Mapping):
            raise TypeError(f"`{ModalityKey.AUDIO}` must be a mapping.")
        if AudioKey.VIEWS not in audio:
            raise KeyError(
                f"Audio codec samples must include `{ModalityKey.AUDIO}.{AudioKey.VIEWS}`."
            )
        views = audio[AudioKey.VIEWS]
        if not isinstance(views, Mapping):
            raise TypeError(f"`{ModalityKey.AUDIO}.{AudioKey.VIEWS}` must be a mapping.")
        if AudioView.WAVEFORM not in views:
            raise KeyError(
                f"Audio codec samples must include "
                f"`{ModalityKey.AUDIO}.{AudioKey.VIEWS}.{AudioView.WAVEFORM}`."
            )
        if AudioKey.SAMPLE_RATE not in audio:
            raise KeyError(
                f"Audio codec samples must include `{ModalityKey.AUDIO}.{AudioKey.SAMPLE_RATE}`."
            )
        sample_rate = audio[AudioKey.SAMPLE_RATE]
        return views[AudioView.WAVEFORM], _maybe_int(sample_rate)

    def _extract_text(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        if ModalityKey.TEXT in row and row[ModalityKey.TEXT] is not None:
            text = row[ModalityKey.TEXT]
            if isinstance(text, Mapping):
                data = dict(text)
                if TextKey.CONTENT not in data:
                    raise KeyError(
                        f"Text samples must include `{ModalityKey.TEXT}.{TextKey.CONTENT}`."
                    )
                if data[TextKey.CONTENT] is None:
                    raise ValueError(f"`{ModalityKey.TEXT}.{TextKey.CONTENT}` cannot be None.")
                data[TextKey.CONTENT] = str(data[TextKey.CONTENT])
                if TextOptKey.LANG in data and data[TextOptKey.LANG] is not None:
                    data[TextOptKey.LANG] = str(data[TextOptKey.LANG])
                return data
            return {TextKey.CONTENT: str(text)}
        return None

    def _to_waveform_tensor(self, waveform: Any) -> torch.Tensor:
        if isinstance(waveform, torch.Tensor):
            tensor = waveform.detach().to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(waveform, dtype=torch.float32)

        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError("Audio waveform must be a 1D or 2D value.")
        if tensor.shape[0] <= 8 and tensor.shape[0] <= tensor.shape[1]:
            return tensor
        if tensor.shape[1] <= 8:
            return tensor.transpose(0, 1).contiguous()
        return tensor

    def _resample(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if sample_rate == self.sample_rate:
            return waveform

        try:
            import torchaudio.functional as audio_functional
        except (ImportError, OSError):
            return _linear_resample(waveform, sample_rate, self.sample_rate)

        return audio_functional.resample(waveform, sample_rate, self.sample_rate)

    def _match_channels(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.channels is None:
            return waveform

        current_channels = waveform.shape[0]
        if current_channels == self.channels:
            return waveform
        if self.channels == 1:
            return waveform.mean(dim=0, keepdim=True)
        if current_channels == 1:
            return waveform.expand(self.channels, -1).clone()
        if current_channels > self.channels:
            return waveform[: self.channels]

        repeats = (self.channels + current_channels - 1) // current_channels
        return waveform.repeat(repeats, 1)[: self.channels]

    def _truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.max_clip_seconds is None:
            return waveform
        target_frames = max(1, round(self.max_clip_seconds * self.sample_rate))
        return waveform[..., :target_frames]


def _linear_resample(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    target_frames = max(
        1,
        round(waveform.shape[-1] * target_sample_rate / sample_rate),
    )
    return F.interpolate(
        waveform.unsqueeze(0),
        size=target_frames,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


AudioCodecTask = AudioCodecFormatter
