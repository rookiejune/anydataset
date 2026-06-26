from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from ..types.item import AudioView
from .abc import AudioProvider

type LongCatDecoderName = Literal[
    "16k_4codebooks",
    "24k_2codebooks",
    "24k_4codebooks",
]


class LongCatViewProvider(nn.Module, AudioProvider):
    output = AudioView.LONGCAT

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        decoders: Sequence[LongCatDecoderName] = ("16k_4codebooks",),
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        n_acoustic_codebooks: int | None = None,
    ) -> None:
        super().__init__()
        try:
            from anytrain.codec.longcat import LongCatAudioCodec
        except ImportError as exc:
            raise ImportError(
                "LongCatViewProvider requires `anytrain[longcat]`; install anytrain with "
                "its longcat extra."
            ) from exc

        self.longcat_codec = LongCatAudioCodec.from_pretrained(
            cache_dir=cache_dir,
            decoders=decoders,
            device=device,
            local_files_only=local_files_only,
            force_download=force_download,
        )

        self.longcat_codec.eval()

    @torch.inference_mode()
    def forward(self, views: Mapping[AudioView, Any]):
        waveform, sample_rate = self._batch(views)
        semantic_codes, acoustic_codes = self.longcat_codec.encode(
            waveform,
            sample_rate,
        )
        return {
            "semantic_codes": self._tensor(semantic_codes)[0],
            "acoustic_codes": self._tensor(acoustic_codes)[0],
        }
