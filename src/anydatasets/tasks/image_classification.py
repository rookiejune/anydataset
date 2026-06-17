from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from anydatasets.samples import Sample
from anydatasets.tasks.base import BatchBuilder


@dataclass(frozen=True)
class BatchMeta:
    dataset_names: list[str]
    sample_indices: list[int]


@dataclass(frozen=True)
class ImageClassificationBatch:
    images: torch.Tensor
    labels: torch.Tensor
    meta: BatchMeta


class ImageClassificationTask(BatchBuilder):
    def __init__(self, image_key: str = "image", label_key: str = "label"):
        self.image_key = image_key
        self.label_key = label_key

    def build(self, samples: Sequence[Sample]) -> ImageClassificationBatch:
        if not samples:
            raise ValueError("Cannot build a batch from an empty sample list.")

        images = [self._to_image_tensor(sample.data[self.image_key]) for sample in samples]
        labels = [sample.data[self.label_key] for sample in samples]
        meta = BatchMeta(
            dataset_names=[sample.dataset_name for sample in samples],
            sample_indices=[sample.sample_index for sample in samples],
        )
        return ImageClassificationBatch(
            images=torch.stack(images, dim=0),
            labels=torch.as_tensor(labels, dtype=torch.long),
            meta=meta,
        )

    def _to_image_tensor(self, image):
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = torch.as_tensor(image)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        return tensor
