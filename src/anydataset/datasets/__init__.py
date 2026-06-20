from .base import DatasetAdapter, TaskSampleAdapter
from .catalog import DEFAULT_DATASET_MAP
from .esc50 import ESC50AudioCodecAdapter, esc50_spec
from .fleurs import FleursAudioCodecAdapter, fleurs_spec
from .fsd50k import FSD50KAudioCodecAdapter, FSD50KDataset, fsd50k_spec
from .huggingface import HuggingFaceDataset
from .librispeech_asr import LibriSpeechASRAudioCodecAdapter, librispeech_asr_spec
from .local_files import LocalFilesDataset
from .nsynth import NSynthAudioCodecAdapter, nsynth_spec
from .task_adapters import (
    TaskAdapterFactory,
    TaskAdapterRegistry,
    default_task_adapter_registry,
    register_builtin_task_adapters,
)

__all__ = [
    "DatasetAdapter",
    "DEFAULT_DATASET_MAP",
    "ESC50AudioCodecAdapter",
    "FSD50KAudioCodecAdapter",
    "FSD50KDataset",
    "FleursAudioCodecAdapter",
    "HuggingFaceDataset",
    "LibriSpeechASRAudioCodecAdapter",
    "LocalFilesDataset",
    "NSynthAudioCodecAdapter",
    "TaskSampleAdapter",
    "TaskAdapterFactory",
    "TaskAdapterRegistry",
    "esc50_spec",
    "default_task_adapter_registry",
    "fleurs_spec",
    "fsd50k_spec",
    "librispeech_asr_spec",
    "nsynth_spec",
    "register_builtin_task_adapters",
]
