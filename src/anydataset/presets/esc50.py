from __future__ import annotations

from functools import partial
from typing import Any

from ..types import AudioMeta, AudioView
from ..dataset.abc import IterableAnyDataset
from ..types import Preset
from ..types.item import Transforms
from ..rowmap import labels, sample_from_row
from .registry import preset_spec


class ESC50(IterableAnyDataset):
    def __init__(
        self,
        split: str | None = None,
        *,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> None:
        super().__init__(
            spec=preset_spec(Preset.ESC50, split=split, **load_options),
            parse_fn=partial(
                sample_from_row,
                audio={
                    "audio": AudioView.WAVEFORM,
                    "category": AudioMeta.LABEL,
                    "target": labels("target"),
                    "esc10": labels("esc10"),
                },
            ),
            transforms=transforms,
        )
