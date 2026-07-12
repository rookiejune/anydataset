from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..types.item import AudioView
from .codec import CodecProvider

if TYPE_CHECKING:
    from anytrain.codec.longcat import LongCatDecoderName


class LongCatProvider(CodecProvider):
    output = AudioView.LONGCAT

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        decoder: LongCatDecoderName = "16k_4codebooks",
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> None:
        try:
            from anytrain.codec.longcat import LongCat
        except ImportError as exc:
            raise ImportError(
                "LongCatProvider requires `anytrain[longcat]`; install anytrain with "
                "its longcat extra."
            ) from exc

        super().__init__(
            LongCat.from_pretrained(
                cache_dir=cache_dir,
                decoder=decoder,
                device=device,
                local_files_only=local_files_only,
                force_download=force_download,
            ),
            self.output,
        )
