from __future__ import annotations

from anydataset.api.spec import DatasetSpec

from .esc50.dataset import (
    esc50_spec,
    register_task_adapters as register_esc50_task_adapters,
)
from .fleurs.dataset import (
    fleurs_spec,
    register_task_adapters as register_fleurs_task_adapters,
)
from .fsd50k.dataset import (
    fsd50k_spec,
    register_task_adapters as register_fsd50k_task_adapters,
)
from .librispeech_asr.dataset import (
    librispeech_asr_spec,
    register_task_adapters as register_librispeech_asr_task_adapters,
)
from .nsynth.dataset import (
    nsynth_spec,
    register_task_adapters as register_nsynth_task_adapters,
)


DEFAULT_DATASET_MAP: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(source="huggingface", path="ylecun/mnist", name="mnist"),
    "cifar10": DatasetSpec(source="huggingface", path="uoft-cs/cifar10", name="cifar10"),
    "fleurs": fleurs_spec(),
    "librispeech_asr": librispeech_asr_spec(),
    "esc50": esc50_spec(),
    "nsynth": nsynth_spec(),
    "fsd50k": fsd50k_spec(),
}

DEFAULT_TASK_ADAPTER_REGISTRARS = (
    register_esc50_task_adapters,
    register_fleurs_task_adapters,
    register_fsd50k_task_adapters,
    register_librispeech_asr_task_adapters,
    register_nsynth_task_adapters,
)
