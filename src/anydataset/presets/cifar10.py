from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..types import ImageItem, ImageMeta, ImageView, Modality, Role
from ..dataset.abc import AnyDataset
from ..types import Preset
from ..types.item import Sample, Transforms
from .registry import preset_spec


class CIFAR10(AnyDataset):
    def __init__(
        self,
        split: str | None = None,
        *,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> None:
        super().__init__(
            spec=preset_spec(Preset.CIFAR10, split=split, **load_options),
            parse_fn=_parse,
            transforms=transforms,
        )


def _parse(row: Mapping[str, Any]) -> Sample:
    return {
        (Role.DEFAULT, Modality.IMAGE): ImageItem(
            views={ImageView.PIXEL: row["image"]},
            meta={ImageMeta.LABEL: torch.as_tensor(row["label"], dtype=torch.long)},
        )
    }
