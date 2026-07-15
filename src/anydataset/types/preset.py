from __future__ import annotations

from enum import auto

from .._compat import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import Spec


class Preset(StrEnum):
    MNIST = auto()
    CIFAR10 = auto()
    FLEURS = auto()
    LIBRISPEECH_ASR = auto()
    COMMON_VOICE = auto()
    ESC50 = auto()
    NSYNTH = auto()
    FSD50K = auto()
    WMT19 = auto()

    def spec(self, split: str | None = None, **load_options: Any) -> Spec:
        from ..presets.registry import preset_spec

        return preset_spec(self, split=split, **load_options)
