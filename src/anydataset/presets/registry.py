from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, cast

from ..types import Preset, Spec

if TYPE_CHECKING:
    from ..dataset.abc import AnyDataset, IterableAnyDataset
    from ..types.item import Transforms


@dataclass(frozen=True)
class _PresetRegistration:
    module: str
    dataset_type: str


_PRESETS = {
    Preset.MNIST: _PresetRegistration("mnist", "MNIST"),
    Preset.CIFAR10: _PresetRegistration("cifar10", "CIFAR10"),
    Preset.FLEURS: _PresetRegistration("fleurs", "Fleurs"),
    Preset.LIBRISPEECH_ASR: _PresetRegistration(
        "librispeech_asr",
        "LibriSpeechASR",
    ),
    Preset.COMMON_VOICE: _PresetRegistration("common_voice", "CommonVoice"),
    Preset.ESC50: _PresetRegistration("esc50", "ESC50"),
    Preset.NSYNTH: _PresetRegistration("nsynth", "NSynth"),
    Preset.FSD50K: _PresetRegistration("fsd50k", "FSD50K"),
    Preset.WMT19: _PresetRegistration("wmt19", "WMT19"),
}


def preset_spec(
    preset: Preset,
    split: str | None = None,
    **load_options: Any,
) -> Spec:
    _, module = _load(preset)
    create_spec = cast(Callable[..., Spec], getattr(module, "create_spec"))
    return create_spec(split=split, **load_options)


def create_map_preset(
    preset: Preset,
    split: str | None = None,
    transforms: Transforms | None = None,
    **load_options: Any,
) -> AnyDataset:
    registration, module = _load(preset)
    dataset_type = cast(
        Callable[..., Any],
        getattr(module, registration.dataset_type),
    )
    return dataset_type(split=split, transforms=transforms, **load_options)


def create_iterable_preset(
    preset: Preset,
    split: str | None = None,
    transforms: Transforms | None = None,
    **load_options: Any,
) -> IterableAnyDataset:
    del split, transforms, load_options
    raise ValueError(
        f"Preset {preset.value!r} is map-style; use AnyDataset.preset()."
    )


def _load(preset: Preset) -> tuple[_PresetRegistration, ModuleType]:
    registration = _PRESETS.get(preset)
    if registration is None:
        raise ValueError(f"Unsupported preset: {preset!r}.")
    return registration, import_module(f".{registration.module}", __package__)
