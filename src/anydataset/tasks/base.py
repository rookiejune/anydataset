from __future__ import annotations

from abc import ABC
from enum import Enum, auto

from anydataset.samples import Sample


class AutoNameEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class Task(AutoNameEnum):
    IMAGE_CLASSIFICATION = auto()
    AUDIO_CODEC = auto()


class SampleFormatter(ABC):
    def __call__(self, sample: Sample) -> Sample:
        return sample


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
