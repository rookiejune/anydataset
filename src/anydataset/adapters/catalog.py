from __future__ import annotations

from typing import Callable

from .base import DatasetAdapter
from .esc50 import ESC50Adapter, esc50_spec
from .fleurs import FleursAdapter, fleurs_spec
from .fsd50k import FSD50KAdapter, fsd50k_spec
from .huggingface import HuggingFaceAdapter
from .librispeech_asr import LibriSpeechASRAdapter, librispeech_asr_spec
from .nsynth import NSynthAdapter, nsynth_spec
from ..api.spec import DatasetSpec

type AdapterFactory = Callable[[DatasetSpec], DatasetAdapter]
type AdapterBinding = DatasetAdapter | AdapterFactory


DEFAULT_DATASET_MAP: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(source="huggingface", path="ylecun/mnist", name="mnist"),
    "cifar10": DatasetSpec(source="huggingface", path="uoft-cs/cifar10", name="cifar10"),
    "fleurs": fleurs_spec(),
    "librispeech_asr": librispeech_asr_spec(),
    "esc50": esc50_spec(),
    "nsynth": nsynth_spec(),
    "fsd50k": fsd50k_spec(),
}


def _huggingface_adapter(_: DatasetSpec) -> DatasetAdapter:
    return HuggingFaceAdapter()


def _esc50_adapter(_: DatasetSpec) -> DatasetAdapter:
    return ESC50Adapter()


def _fleurs_adapter(spec: DatasetSpec) -> DatasetAdapter:
    return FleursAdapter(lang=str(spec.load_options.get("config_name", "en_us")))


def _librispeech_asr_adapter(_: DatasetSpec) -> DatasetAdapter:
    return LibriSpeechASRAdapter()


def _nsynth_adapter(_: DatasetSpec) -> DatasetAdapter:
    return NSynthAdapter()


def _fsd50k_adapter(_: DatasetSpec) -> DatasetAdapter:
    return FSD50KAdapter()


DEFAULT_ADAPTER_MAP: dict[str, AdapterBinding] = {
    "mnist": _huggingface_adapter,
    "cifar10": _huggingface_adapter,
    "fleurs": _fleurs_adapter,
    "librispeech_asr": _librispeech_asr_adapter,
    "esc50": _esc50_adapter,
    "nsynth": _nsynth_adapter,
    "fsd50k": _fsd50k_adapter,
}
