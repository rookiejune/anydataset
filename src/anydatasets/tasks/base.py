from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Sequence

from anydatasets.samples import Sample


class AutoNameEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class Task(AutoNameEnum):
    IMAGE_CLASSIFICATION = auto()


class BatchBuilder(ABC):
    @abstractmethod
    def build(self, samples: Sequence[Sample]):
        raise NotImplementedError


def get_batch_builder(task: Task) -> BatchBuilder:
    if not isinstance(task, Task):
        raise TypeError(f"task must be a Task enum value, got {task!r}.")

    if task is Task.IMAGE_CLASSIFICATION:
        from .image_classification import ImageClassificationTask

        return ImageClassificationTask()

    raise ValueError(f"Unsupported task: {task!r}")
