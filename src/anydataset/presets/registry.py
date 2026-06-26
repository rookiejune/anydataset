from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..types import Preset, Source, Spec

if TYPE_CHECKING:
    from ..dataset.abc import AnyDataset, IterableAnyDataset


def preset_spec(
    preset: Preset,
    split: str | None = None,
    **load_options: Any,
) -> Spec:
    match preset:
        case Preset.MNIST:
            spec = Spec(source=Source.HF, path="ylecun/mnist", split="train")
        case Preset.CIFAR10:
            spec = Spec(source=Source.HF, path="uoft-cs/cifar10", split="train")
        case Preset.FLEURS:
            spec = Spec(
                source=Source.HF,
                path="google/fleurs",
                split="train",
                load_options={
                    "config_name": "en_us",
                    "streaming": True,
                },
            )
        case Preset.LIBRISPEECH_ASR:
            spec = Spec(
                source=Source.HF,
                path="openslr/librispeech_asr",
                split="train.100",
                load_options={
                    "config_name": "clean",
                    "streaming": True,
                },
            )
        case Preset.ESC50:
            spec = Spec(
                source=Source.HF,
                path="ashraq/esc50",
                split="train",
                load_options={"streaming": True},
            )
        case Preset.NSYNTH:
            spec = Spec(
                source=Source.HF,
                path="confit/nsynth-parquet",
                split="train",
                load_options={
                    "config_name": "instrument",
                    "streaming": True,
                },
            )
        case Preset.FSD50K:
            spec = Spec(source=Source.HF, path="Fhrozen/FSD50k", split="dev")
        case Preset.WMT19:
            spec = Spec(
                source=Source.HF,
                path="wmt/wmt19",
                split="train",
                load_options={
                    "config_name": "cs-en",
                    "streaming": True,
                },
            )

    return replace(
        spec,
        split=split or spec.split,
        load_options={**spec.load_options, **load_options},
    )


def create_preset(
    preset: Preset,
    split: str | None = None,
    **load_options: Any,
) -> AnyDataset | IterableAnyDataset:
    match preset:
        case Preset.MNIST:
            from .mnist import MNIST

            return MNIST(split=split, **load_options)
        case Preset.CIFAR10:
            from .cifar10 import CIFAR10

            return CIFAR10(split=split, **load_options)
        case Preset.FLEURS:
            from .fleurs import Fleurs

            return Fleurs(split=split, **load_options)
        case Preset.LIBRISPEECH_ASR:
            from .librispeech_asr import LibriSpeechASR

            return LibriSpeechASR(split=split, **load_options)
        case Preset.ESC50:
            from .esc50 import ESC50

            return ESC50(split=split, **load_options)
        case Preset.NSYNTH:
            from .nsynth import NSynth

            return NSynth(split=split, **load_options)
        case Preset.FSD50K:
            from .fsd50k import FSD50K

            return FSD50K(split=split, **load_options)
        case Preset.WMT19:
            from .wmt19 import WMT19

            return WMT19(split=split, **load_options)
