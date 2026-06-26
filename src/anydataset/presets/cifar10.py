from __future__ import annotations

from functools import partial
from typing import Any

from ..dataset import ImageOptKey, ImageView
from ..dataset.abc import AnyDataset
from ..types import Preset
from ..utils import sample_from_row
from .registry import preset_spec


class CIFAR10(AnyDataset):
    def __init__(self, split: str | None = None, **load_options: Any) -> None:
        super().__init__(
            spec=preset_spec(Preset.CIFAR10, split=split, **load_options),
            parse_fn=partial(
                sample_from_row,
                image={
                    "image": ImageView.PIXEL,
                    "label": ImageOptKey.LABEL,
                },
            ),
        )
