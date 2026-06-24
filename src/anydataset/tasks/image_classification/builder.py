from __future__ import annotations

import torch

from ...samples import Sample
from ..base import SampleFormatter
from .schema import IMAGE_KEY, LABEL_KEY


class ImageClassificationFormatter(SampleFormatter):
    def __init__(self, image_key: str = IMAGE_KEY, label_key: str = LABEL_KEY):
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, sample: Sample) -> Sample:
        data = dict(sample.data)
        data[self.image_key] = self._to_image_tensor(data[self.image_key])
        data[self.label_key] = int(data[self.label_key])
        return Sample(
            data=data,
            dataset_name=sample.dataset_name,
            sample_index=sample.sample_index,
        )

    def _to_image_tensor(self, image):
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = _as_tensor(image)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        return tensor


def _as_tensor(value):
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError):
        try:
            import numpy as np
        except ImportError as import_exc:
            raise TypeError("Image values must be tensor-like or NumPy/PIL compatible.") from import_exc

        array = np.asarray(value)
        if not array.flags.writeable:
            array = array.copy()
        try:
            return torch.as_tensor(array)
        except (TypeError, ValueError, RuntimeError) as tensor_exc:
            raise TypeError("Image values must be tensor-like or NumPy/PIL compatible.") from tensor_exc


ImageClassificationTask = ImageClassificationFormatter
