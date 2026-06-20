from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from anydataset.datasets.base import TaskSampleAdapter
from anydataset.tasks import Task

if TYPE_CHECKING:
    from anydataset.api.spec import DatasetSpec


TaskAdapterFactory = Callable[["DatasetSpec"], TaskSampleAdapter]


class TaskAdapterRegistry:
    def __init__(self):
        self._factories: dict[tuple[str, Task], TaskAdapterFactory] = {}

    def register(
        self,
        dataset_name: str,
        task: Task,
        factory: TaskAdapterFactory,
    ) -> None:
        _validate_dataset_name(dataset_name)
        if not isinstance(task, Task):
            raise TypeError("task must be a Task enum value.")
        if not callable(factory):
            raise TypeError("factory must be callable.")

        key = (dataset_name, task)
        if key in self._factories:
            raise ValueError(
                f"Task adapter already registered for dataset {dataset_name!r} "
                f"and task {task.value!r}."
            )
        self._factories[key] = factory

    def resolve(self, spec: "DatasetSpec", task: Task) -> TaskSampleAdapter | None:
        if not isinstance(task, Task):
            raise TypeError("task must be a Task enum value.")

        factory = self._factories.get((spec.name, task))
        if factory is None:
            return None

        adapter = factory(spec)
        if not isinstance(adapter, TaskSampleAdapter):
            raise TypeError("task adapter factory must return a TaskSampleAdapter.")
        return adapter

    def copy(self) -> "TaskAdapterRegistry":
        registry = TaskAdapterRegistry()
        registry._factories.update(self._factories)
        return registry


def default_task_adapter_registry() -> TaskAdapterRegistry:
    registry = TaskAdapterRegistry()
    register_builtin_task_adapters(registry)
    return registry


def register_builtin_task_adapters(registry: TaskAdapterRegistry) -> None:
    from anydataset.datasets.esc50.adapters.audio_codec import ESC50AudioCodecAdapter
    from anydataset.datasets.fleurs.adapters.audio_codec import FleursAudioCodecAdapter
    from anydataset.datasets.fsd50k.adapters.audio_codec import FSD50KAudioCodecAdapter
    from anydataset.datasets.librispeech_asr.adapters.audio_codec import (
        LibriSpeechASRAudioCodecAdapter,
    )
    from anydataset.datasets.nsynth.adapters.audio_codec import NSynthAudioCodecAdapter

    registry.register("esc50", Task.AUDIO_CODEC, lambda spec: ESC50AudioCodecAdapter())
    registry.register("fleurs", Task.AUDIO_CODEC, lambda spec: FleursAudioCodecAdapter())
    registry.register("fsd50k", Task.AUDIO_CODEC, lambda spec: FSD50KAudioCodecAdapter())
    registry.register(
        "librispeech_asr",
        Task.AUDIO_CODEC,
        lambda spec: LibriSpeechASRAudioCodecAdapter(),
    )
    registry.register("nsynth", Task.AUDIO_CODEC, lambda spec: NSynthAudioCodecAdapter())


def _validate_dataset_name(dataset_name: str) -> None:
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("dataset_name must be a non-empty string.")
