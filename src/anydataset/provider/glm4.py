"""Online GLM-4-Voice frame-token materialization."""

from __future__ import annotations

import torch

from ..types.item import AudioView
from .codec import AudioTokenizerProvider


class GLM4Provider(AudioTokenizerProvider):
    """Materialize ``AudioView.GLM4`` through AnyTrain's tokenizer capability."""

    output = AudioView.GLM4

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        try:
            from anytrain.codec import load_audio_tokenizer
        except ImportError as exc:
            raise ImportError(
                "GLM4Provider requires the current AnyTrain package. Install "
                "AnyTrain and configure GLM4_VOICE_SOURCE_ROOT for a GLM-4-Voice "
                "checkout or fork. The source Git commit is not pinned; AnyTrain "
                "validates its files, import origin, APIs, and model/preprocess "
                "contract while keeping the tokenizer weights revision fixed. "
                "Online GLM-4 requires transformers==4.44.1, which conflicts with "
                "transformers>=4.46 in the BiCodec extra and transformers>=4.51.0 "
                "in the module extra; use an independent GLM producer when those "
                "environments cannot be combined."
            ) from exc

        super().__init__(
            load_audio_tokenizer("glm4", device=device),
            self.output,
        )


__all__ = ["GLM4Provider"]
