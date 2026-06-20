from __future__ import annotations

from typing import TYPE_CHECKING

from anydataset.api.spec import DatasetSpec

if TYPE_CHECKING:
    from anydataset.datasets.task_adapters import TaskAdapterRegistry


def esc50_spec(
    split: str = "train",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="ashraq/esc50",
        name="esc50",
        split=split,
        load_options={
            "streaming": streaming,
        },
    )


def register_task_adapters(registry: "TaskAdapterRegistry") -> None:
    from anydataset.tasks import Task

    from .adapters.audio_codec import ESC50AudioCodecAdapter

    registry.register("esc50", Task.AUDIO_CODEC, lambda spec: ESC50AudioCodecAdapter())
