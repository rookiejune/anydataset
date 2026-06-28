from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..dataset.collate import Batch
from ..types.item import AudioView, Modality, Role, TextView
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

    def call_batch(self, batch: Batch) -> Sequence[str]:
        ref = _audio_ref(batch)
        audio = batch.sample[ref]
        waveform, sample_rates = audio.views[AudioView.WAVEFORM]
        sample_rate = _single_sample_rate(sample_rates)
        return _text_outputs(self.asr.transcribe(waveform, sample_rate))


def _audio_ref(batch: Batch) -> tuple[Role, Modality]:
    refs = tuple(
        ref
        for ref in batch.sample
        if ref[1] is Modality.AUDIO
        and AudioView.WAVEFORM in batch.sample[ref].views
    )
    if len(refs) != 1:
        raise ValueError(
            "WhisperASRProvider.call_batch expects exactly one audio waveform input."
        )
    return refs[0]


def _single_sample_rate(sample_rates: torch.Tensor) -> int:
    if sample_rates.ndim != 1:
        raise ValueError("Batched waveform sample rates must have shape [batch].")
    if sample_rates.numel() == 0:
        raise ValueError("Batched waveform sample rates must not be empty.")
    first = sample_rates[0].item()
    if not torch.equal(sample_rates, sample_rates.new_full(sample_rates.shape, first)):
        raise ValueError("WhisperASRProvider.call_batch requires one sample rate per batch.")
    return int(first)


def _text_outputs(output: Any) -> Sequence[str]:
    if isinstance(output, str):
        return [output]
    if not isinstance(output, Sequence):
        raise TypeError("WhisperASRProvider batch transcribe output must be a string sequence.")
    texts = list(output)
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("WhisperASRProvider batch transcribe output must contain strings.")
    return texts
