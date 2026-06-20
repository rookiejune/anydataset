from __future__ import annotations

from typing import TYPE_CHECKING

from anydataset.api.spec import DatasetSpec

if TYPE_CHECKING:
    from anydataset.datasets.task_adapters import TaskAdapterRegistry


def nsynth_spec(
    config_name: str = "instrument",
    split: str = "train",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="confit/nsynth-parquet",
        name="nsynth",
        split=split,
        load_options={
            "config_name": config_name,
            "streaming": streaming,
        },
    )


def register_task_adapters(registry: "TaskAdapterRegistry") -> None:
    from anydataset.tasks import Task

    from .adapters.audio_codec import NSynthAudioCodecAdapter

    registry.register("nsynth", Task.AUDIO_CODEC, lambda spec: NSynthAudioCodecAdapter())
