from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..dataset.abc import AnyDataset
from ..types import ImageItem, ImageMeta, ImageView, Modality, Role, Source, Spec
from ..types.item import Sample, Transforms


class CIFAR10(AnyDataset):
    def __init__(
        self,
        split: str | None = None,
        *,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> None:
        super().__init__(
            spec=create_spec(split=split, **load_options),
            parse_fn=_parse,
            transforms=transforms,
        )


def create_spec(split: str | None = None, **load_options: Any) -> Spec:
    return Spec(
        source=Source.HF,
        path="uoft-cs/cifar10",
        split="train" if split is None else split,
        load_options=load_options,
    )


def _parse(row: Mapping[str, Any]) -> Sample:
    return {
        (Role.DEFAULT, Modality.IMAGE): ImageItem(
            views={ImageView.PIXEL: row["image"]},
            meta={ImageMeta.LABEL: torch.as_tensor(row["label"], dtype=torch.long)},
        )
    }
