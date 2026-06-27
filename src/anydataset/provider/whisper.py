from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..types.item import AudioView, TextView
from .abc import AudioProvider


class WhisperASRProvider(AudioProvider):
    output = TextView.TEXT

    def __init__(
        self,
        *,
        decode_options: Mapping[str, Any] | None = None,
        **load_options: Any,
    ) -> None:
        try:
            from anytrain.evaluator.speech import WhisperASREvaluator
        except ImportError as exc:
            raise ImportError(
                "WhisperASRProvider requires `anytrain[speech]`."
            ) from exc
        self.asr = WhisperASREvaluator(
            decode_options=decode_options,
            **load_options,
        )

    def __call__(self, views: Mapping[AudioView, Any]) -> Any:
        waveform, sample_rate = self._waveform(views)
        return self.asr.transcribe(waveform, sample_rate)
