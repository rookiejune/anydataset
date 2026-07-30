from __future__ import annotations

from functools import partial
from typing import Any

from ..dataset.abc import AnyDataset
from ..rowmap import sample_from_row
from ..types import AudioView, Preset
from ..types.item import Transforms
from .registry import preset_spec


class FSD50K(AnyDataset):
    def __init__(
        self,
        split: str | None = None,
        *,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> None:
        if split is not None and split not in _VALID_SPLITS:
            raise ValueError("FSD50K split must be 'dev' or 'eval'.")
        extra = set(load_options) - {"revision"}
        if extra:
            name = min(extra)
            raise TypeError(f"Unexpected FSD50K load option: {name}.")
        revision = load_options.get("revision", "main")
        if not isinstance(revision, str) or not revision:
            raise ValueError("FSD50K revision must be a non-empty string.")
        super().__init__(
            spec=preset_spec(Preset.FSD50K, split=split, revision=revision),
            parse_fn=partial(
                sample_from_row,
                audio={
                    "audio": AudioView.WAVEFORM,
                },
            ),
            transforms=transforms,
        )


_VALID_SPLITS = frozenset({"dev", "eval"})
