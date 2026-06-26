from __future__ import annotations

from functools import partial
from typing import Any

from ..dataset import AudioView
from ..dataset.abc import IterableAnyDataset
from ..types import Preset
from ..utils import labels, sample_from_row
from .registry import preset_spec


class NSynth(IterableAnyDataset):
    def __init__(self, split: str | None = None, **load_options: Any) -> None:
        super().__init__(
            spec=preset_spec(Preset.NSYNTH, split=split, **load_options),
            parse_fn=partial(
                sample_from_row,
                audio={
                    "audio": AudioView.WAVEFORM,
                    "instrument_family_str": labels("instrument_family"),
                    "instrument_source_str": labels("instrument_source"),
                    "pitch": labels("pitch"),
                    "velocity": labels("velocity"),
                },
            ),
        )
