from __future__ import annotations

from anydataset.api.spec import DatasetSpec

from .esc50.dataset import esc50_spec
from .fleurs.dataset import fleurs_spec
from .fsd50k.dataset import fsd50k_spec
from .librispeech_asr.dataset import librispeech_asr_spec
from .nsynth.dataset import nsynth_spec


DEFAULT_DATASET_MAP: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(source="huggingface", path="ylecun/mnist", name="mnist"),
    "cifar10": DatasetSpec(source="huggingface", path="uoft-cs/cifar10", name="cifar10"),
    "fleurs": fleurs_spec(),
    "librispeech_asr": librispeech_asr_spec(),
    "esc50": esc50_spec(),
    "nsynth": nsynth_spec(),
    "fsd50k": fsd50k_spec(),
}
