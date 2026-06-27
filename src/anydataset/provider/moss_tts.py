from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
        return output.waveform, output.sample_rate
