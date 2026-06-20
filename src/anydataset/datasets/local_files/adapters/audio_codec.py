from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from anydataset.datasets.base import TaskSampleAdapter
from anydataset.tasks.audio_codec import (
    SAMPLE_RATE_KEY,
    TEXT_KEY,
    WAVEFORM_KEY,
)


@dataclass(frozen=True)
class AudioCodecSampleAdapter(TaskSampleAdapter):
    waveform_key: str = WAVEFORM_KEY
    sample_rate_key: str = SAMPLE_RATE_KEY
    text_key: str | None = None
    audio_key: str | None = None
    audio_waveform_key: str = "array"
    audio_sample_rate_key: str = "sampling_rate"

    def adapt(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        waveform, sample_rate = self._extract_waveform(row)
        data = {
            WAVEFORM_KEY: waveform,
            SAMPLE_RATE_KEY: sample_rate,
        }
        if self.text_key is not None:
            data[TEXT_KEY] = row[self.text_key]
        return data

    def _extract_waveform(self, row: Mapping[str, Any]) -> tuple[Any, int | None]:
        if self.audio_key is None:
            return row[self.waveform_key], _maybe_int(row.get(self.sample_rate_key))

        audio = row[self.audio_key]
        if isinstance(audio, Mapping):
            return (
                audio[self.audio_waveform_key],
                _maybe_int(audio.get(self.audio_sample_rate_key)),
            )
        decoded = _maybe_decode_audio(audio)
        if decoded is not None:
            return decoded
        return audio, _maybe_int(row.get(self.sample_rate_key))


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _maybe_decode_audio(audio: Any) -> tuple[Any, int] | None:
    get_all_samples = getattr(audio, "get_all_samples", None)
    if get_all_samples is None:
        return None

    samples = get_all_samples()
    data = getattr(samples, "data")
    sample_rate = getattr(samples, "sample_rate")
    return data, int(sample_rate)
