from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

try:
    import torchaudio
except ImportError as exc:
    raise ImportError(
        "AudioProvider requires `pip install anydataset[audio]`."
    ) from exc


from ..types import AudioView
from ..dataset.collate import Batch, FieldGroup, FieldRef
from ..types.item import Modality, Role


class AudioProvider(ABC):
    @abstractmethod
    def __call__(self, views: Mapping[AudioView, Any]) -> Any: ...

    def _waveform(self, views: Mapping[AudioView, Any]) -> tuple[Tensor, int]:
        if AudioView.WAVEFORM in views:
            return views[AudioView.WAVEFORM]
        if AudioView.FILE in views:
            return torchaudio.load(_audio_source(views[AudioView.FILE]))
        raise ValueError("AudioProvider expects an audio waveform or file input view.")

    def _batch(self, views: Mapping[AudioView, Any]) -> tuple[Tensor, int]:
        waveform, sample_rate = self._waveform(views)
        return waveform.unsqueeze(0), sample_rate

    def _waveform_batch(
        self,
        batch: Batch,
        ref: tuple[Role, Modality],
    ) -> tuple[Tensor, Tensor, Tensor]:
        views = batch.sample[ref].views
        if AudioView.WAVEFORM in views:
            waveform, sample_rates = views[AudioView.WAVEFORM]
            lengths = batch.lengths(FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM))
            return waveform, sample_rates, lengths
        if AudioView.FILE in views:
            files = _audio_files(views[AudioView.FILE])
            waveforms: list[Tensor] = []
            sample_rates: list[int] = []
            for file in files:
                waveform, sample_rate = torchaudio.load(_audio_source(file))
                waveforms.append(waveform)
                sample_rates.append(sample_rate)
            return _pad_waveforms(waveforms, sample_rates)
        raise ValueError("AudioProvider expects an audio waveform or file input view.")

    @staticmethod
    def _tensor(input: Tensor) -> Tensor:
        return input.detach().cpu().contiguous()


def _audio_path(value: Any) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _audio_source(value: Any) -> BytesIO | str:
    if isinstance(value, bytes):
        return BytesIO(value)
    if not isinstance(value, (str, Path)):
        raise TypeError("input must be bytes or a filesystem path.")
    return str(_audio_path(value))


def _audio_files(value: Any) -> list[Any]:
    if isinstance(value, (str, Path, bytes)):
        return [value]
    if not isinstance(value, list):
        raise TypeError("batched audio file view must be a list of paths or bytes.")
    return value


def _pad_waveforms(
    waveforms: list[Tensor],
    sample_rates: list[int],
) -> tuple[Tensor, Tensor, Tensor]:
    if not waveforms:
        raise ValueError("Batched audio file view must not be empty.")
    tensors = [
        waveform if isinstance(waveform, torch.Tensor) else torch.as_tensor(waveform)
        for waveform in waveforms
    ]
    shapes = [tuple(waveform.shape) for waveform in tensors]
    rank = len(shapes[0])
    prefix = shapes[0][:-1]
    if rank == 0 or any(len(shape) != rank or shape[:-1] != prefix for shape in shapes):
        raise ValueError("Only the last waveform dimension may vary in audio file batches.")

    max_len = max(shape[-1] for shape in shapes)
    padded = []
    lengths = []
    for waveform in tensors:
        length = waveform.shape[-1]
        lengths.append(length)
        if length < max_len:
            padding = waveform.new_zeros((*prefix, max_len - length))
            waveform = torch.cat((waveform, padding), dim=-1)
        padded.append(waveform)

    batch = torch.stack(tuple(padded))
    rates = torch.tensor(sample_rates, dtype=torch.int64, device=batch.device)
    length_tensor = torch.tensor(lengths, dtype=torch.int64, device=batch.device)
    return batch, rates, length_tensor
