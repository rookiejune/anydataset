from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..types import Preset, Source, Spec

if TYPE_CHECKING:
    from ..dataset.abc import AnyDataset, IterableAnyDataset
    from ..types.item import Transforms


def preset_spec(
    preset: Preset,
    split: str | None = None,
    **load_options: Any,
) -> Spec:
    if preset is Preset.MNIST:
        spec = Spec(source=Source.HF, path="ylecun/mnist", split="train")
    elif preset is Preset.CIFAR10:
        spec = Spec(source=Source.HF, path="uoft-cs/cifar10", split="train")
    elif preset is Preset.FLEURS:
        spec = Spec(
            source=Source.HF,
            path="google/fleurs",
            split="train",
            load_options={
                "config_name": "en_us",
                "streaming": True,
            },
        )
    elif preset is Preset.LIBRISPEECH_ASR:
        spec = Spec(
            source=Source.HF,
            path="openslr/librispeech_asr",
            split="train.100",
            load_options={
                "config_name": "clean",
                "streaming": True,
            },
        )
    elif preset is Preset.COMMON_VOICE:
        from .common_voice import common_voice_spec

        return common_voice_spec(split=split, **load_options)
    elif preset is Preset.ESC50:
        spec = Spec(
            source=Source.HF,
            path="ashraq/esc50",
            split="train",
            load_options={"streaming": True},
        )
    elif preset is Preset.NSYNTH:
        spec = Spec(
            source=Source.HF,
            path="confit/nsynth-parquet",
            split="train",
            load_options={
                "config_name": "instrument",
                "streaming": True,
            },
        )
    elif preset is Preset.FSD50K:
        spec = Spec(source=Source.HF, path="Fhrozen/FSD50k", split="dev")
    elif preset is Preset.WMT19:
        spec = Spec(
            source=Source.HF,
            path="wmt/wmt19",
            split="train",
            load_options={
                "config_name": "cs-en",
                "streaming": True,
            },
        )
    else:
        raise ValueError(f"Unsupported preset: {preset!r}.")

    return replace(
        spec,
        split=split or spec.split,
        load_options={**spec.load_options, **load_options},
    )


def create_map_preset(
    preset: Preset,
    split: str | None = None,
    transforms: Transforms | None = None,
    **load_options: Any,
) -> AnyDataset:
    if preset is Preset.MNIST:
        from .mnist import MNIST

        return MNIST(split=split, transforms=transforms, **load_options)
    if preset is Preset.CIFAR10:
        from .cifar10 import CIFAR10

        return CIFAR10(split=split, transforms=transforms, **load_options)
    if preset is Preset.FSD50K:
        from .fsd50k import FSD50K

        return FSD50K(split=split, transforms=transforms, **load_options)
    raise ValueError(
        f"Preset {preset.value!r} is iterable; use IterableAnyDataset.preset()."
    )


def create_iterable_preset(
    preset: Preset,
    split: str | None = None,
    transforms: Transforms | None = None,
    **load_options: Any,
) -> IterableAnyDataset:
    if preset is Preset.FLEURS:
        from .fleurs import Fleurs

        return Fleurs(split=split, transforms=transforms, **load_options)
    if preset is Preset.LIBRISPEECH_ASR:
        from .librispeech_asr import LibriSpeechASR

        return LibriSpeechASR(split=split, transforms=transforms, **load_options)
    if preset is Preset.COMMON_VOICE:
        from .common_voice import create_common_voice

        return create_common_voice(split=split, transforms=transforms, **load_options)
    if preset is Preset.ESC50:
        from .esc50 import ESC50

        return ESC50(split=split, transforms=transforms, **load_options)
    if preset is Preset.NSYNTH:
        from .nsynth import NSynth

        return NSynth(split=split, transforms=transforms, **load_options)
    if preset is Preset.WMT19:
        from .wmt19 import WMT19

        return WMT19(split=split, transforms=transforms, **load_options)
    raise ValueError(f"Preset {preset.value!r} is map-style; use AnyDataset.preset().")
