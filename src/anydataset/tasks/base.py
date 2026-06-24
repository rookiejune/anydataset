from __future__ import annotations

from abc import ABC, abstractmethod
from enum import auto
from typing import TYPE_CHECKING, Any, Mapping

from ..enums import AutoNameEnum
from ..samples import Sample

if TYPE_CHECKING:
    from anydataset.adapters import ModalityAdapter


class Task(AutoNameEnum):
    IMAGE_CLASSIFICATION = auto()
    AUDIO_CODEC = auto()


class TaskAdapter(ABC):
    @abstractmethod
    def adapt(
        self,
        row: Mapping[str, Any],
        adapter: "ModalityAdapter",
    ) -> Mapping[str, Any]:
        raise NotImplementedError


class SampleFormatter(ABC):
    def __call__(self, sample: Sample) -> Sample:
        return sample


def get_task_adapter(task: Task) -> TaskAdapter:
    if not isinstance(task, Task):
        raise TypeError(f"task must be a Task enum value, got {task!r}.")

    if task is Task.AUDIO_CODEC:
        from .audio_codec import AudioCodecAdapter

        return AudioCodecAdapter()

    raise ValueError(f"Task {task.value!r} does not define a dataset adapter hook.")


def get_sample_formatter(task: Task, **formatter_kwargs) -> SampleFormatter:
    if not isinstance(task, Task):
        raise TypeError(f"task must be a Task enum value, got {task!r}.")

    if task is Task.IMAGE_CLASSIFICATION:
        from .image_classification import ImageClassificationFormatter

        return ImageClassificationFormatter(**formatter_kwargs)

    if task is Task.AUDIO_CODEC:
        from .audio_codec import AudioCodecFormatter

        return AudioCodecFormatter(**formatter_kwargs)

    raise ValueError(f"Unsupported task: {task!r}")
