from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .registry import load_export as _load_export

if TYPE_CHECKING:
    from .cifar10 import CIFAR10 as CIFAR10
    from .common_voice import CommonVoice as CommonVoice
    from .esc50 import ESC50 as ESC50
    from .fleurs import Fleurs as Fleurs
    from .fsd50k import FSD50K as FSD50K
    from .librispeech_asr import LibriSpeechASR as LibriSpeechASR
    from .mnist import MNIST as MNIST
    from .nsynth import NSynth as NSynth
    from .wmt19 import WMT19 as WMT19

__all__ = [
    "CIFAR10",
    "CommonVoice",
    "ESC50",
    "FSD50K",
    "Fleurs",
    "LibriSpeechASR",
    "MNIST",
    "NSynth",
    "WMT19",
]


def __getattr__(name: str) -> Any:
    value = _load_export(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
