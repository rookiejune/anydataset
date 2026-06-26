from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..dataset.abc import AnyDataset, IterableAnyDataset
    from . import Spec


class Preset(StrEnum):
    MNIST = auto()
    CIFAR10 = auto()
    FLEURS = auto()
    LIBRISPEECH_ASR = auto()
    ESC50 = auto()
    NSYNTH = auto()
    FSD50K = auto()
    WMT19 = auto()

    def spec(self, split: str | None = None, **load_options: Any) -> Spec:
        from ..presets.registry import preset_spec

        return preset_spec(self, split=split, **load_options)

    def create(
        self,
        split: str | None = None,
        **load_options: Any,
    ) -> AnyDataset | IterableAnyDataset:
        from ..presets.registry import create_preset

        return create_preset(self, split=split, **load_options)
