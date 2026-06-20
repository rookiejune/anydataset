from __future__ import annotations

from typing import TYPE_CHECKING

from anydataset.api.spec import DatasetSpec

if TYPE_CHECKING:
    from anydataset.datasets.task_adapters import TaskAdapterRegistry


def librispeech_asr_spec(
    config_name: str = "clean",
    split: str = "train.100",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="openslr/librispeech_asr",
        name="librispeech_asr",
        split=split,
        load_options={
            "config_name": config_name,
            "streaming": streaming,
        },
    )


def register_task_adapters(registry: "TaskAdapterRegistry") -> None:
    from anydataset.tasks import Task

    from .adapters.audio_codec import LibriSpeechASRAudioCodecAdapter

    registry.register(
        "librispeech_asr",
        Task.AUDIO_CODEC,
        lambda spec: LibriSpeechASRAudioCodecAdapter(),
    )
