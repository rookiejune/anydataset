from __future__ import annotations

from typing import TYPE_CHECKING

from anydataset.api.spec import DatasetSpec

if TYPE_CHECKING:
    from anydataset.datasets.task_adapters import TaskAdapterRegistry


def fleurs_spec(
    language: str = "en_us",
    split: str = "train",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="google/fleurs",
        name="fleurs",
        split=split,
        load_options={
            "config_name": language,
            "streaming": streaming,
        },
    )


def register_task_adapters(registry: "TaskAdapterRegistry") -> None:
    from anydataset.tasks import Task

    from .adapters.audio_codec import FleursAudioCodecAdapter

    registry.register("fleurs", Task.AUDIO_CODEC, lambda spec: FleursAudioCodecAdapter())
