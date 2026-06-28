from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..dataset.collate import Batch
from ..types.item import Modality, Role
from ..types.item import AudioView, TextView


class MossTTSProvider:
    output = AudioView.WAVEFORM

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        options: Any | None = None,
        runtime_kwargs: Mapping[str, object] | None = None,
        **load_options: Any,
    ) -> None:
        try:
            from anytrain.tts.moss import MossTTS
        except ImportError as exc:
            raise ImportError(
                "MossTTSProvider requires `anytrain[moss-tts]`."
            ) from exc
        kwargs = {"runtime_kwargs": runtime_kwargs, **load_options}
        if model is None:
            self.tts = MossTTS.from_pretrained(**kwargs)
        else:
            self.tts = MossTTS.from_pretrained(model, **kwargs)
        self.options = options

    def __call__(self, views: Mapping[TextView, Any]) -> Any:
        text = views[TextView.TEXT]
        if not isinstance(text, str):
            raise TypeError("MossTTSProvider expects a string TextView.TEXT input.")
        output = self.tts.synthesize(text, self.options)
        return _audio_output(output)

    def call_batch(self, batch: Batch) -> Sequence[Any]:
        ref = _text_ref(batch)
        texts = _text_batch(batch.sample[ref].views[TextView.TEXT])
        outputs = self.tts.synthesize(texts, self.options)
        if not isinstance(outputs, Sequence):
            raise TypeError("MossTTS batch synthesize output must be a sequence.")
        return [_audio_output(output) for output in outputs]


def _text_ref(batch: Batch) -> tuple[Role, Modality]:
    refs = tuple(
        ref
        for ref in batch.sample
        if ref[1] is Modality.TEXT
        and TextView.TEXT in batch.sample[ref].views
    )
    if len(refs) != 1:
        raise ValueError("MossTTSProvider.call_batch expects exactly one text input.")
    return refs[0]


def _text_batch(value: Any) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("MossTTSProvider.call_batch expects a text sequence.")
    texts = list(value)
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("MossTTSProvider.call_batch expects string text inputs.")
    return texts


def _audio_output(output: Any) -> tuple[Any, int]:
    return output.waveform, output.sample_rate
