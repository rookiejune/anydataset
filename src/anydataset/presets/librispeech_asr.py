from __future__ import annotations

from functools import partial
from typing import Any

from ..dataset import AudioView, Lang, TextMeta, TextView
from ..dataset.abc import IterableAnyDataset
from ..types import Preset
from ..types.item import Transforms
from ..utils import sample_from_row
from .registry import preset_spec


class LibriSpeechASR(IterableAnyDataset):
    def __init__(
        self,
        split: str | None = None,
        *,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> None:
        super().__init__(
            spec=preset_spec(Preset.LIBRISPEECH_ASR, split=split, **load_options),
            parse_fn=partial(
                sample_from_row,
                audio={"audio": AudioView.WAVEFORM},
                text={"text": TextView.TEXT},
                text_values={TextMeta.LANG: Lang.EN},
            ),
            transforms=transforms,
        )
