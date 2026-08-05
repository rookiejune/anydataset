from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..types import AudioItem, AudioView, FileBytes
from ..dataset.collate import Batch, FieldGroup, FieldRef
from ..types.item import Modality, Role

torchaudio: Any | None = None


class AudioProvider(ABC):
    @abstractmethod
    def __call__(self, views: Mapping[AudioView, Any]) -> Any: ...

    def _waveform(self, views: Mapping[AudioView, Any]) -> tuple[Tensor, int]:
        if AudioView.WAVEFORM in views:
            return views[AudioView.WAVEFORM]
        if AudioView.FILE in views:
            return _torchaudio().load(_audio_source(views[AudioView.FILE]))
        raise ValueError("AudioProvider expects an audio waveform or file input view.")

    def _batch(self, views: Mapping[AudioView, Any]) -> tuple[Tensor, int]:
        waveform, sample_rate = self._waveform(views)
        return waveform.unsqueeze(0), sample_rate

    def _waveform_batch(
        self,
        batch: Batch,
        ref: tuple[Role, Modality],
    ) -> tuple[Tensor, Tensor, Tensor]:
        item = batch.sample[ref]
        if not isinstance(item, AudioItem):
            raise TypeError(f"{ref!r} requires a collated AudioItem.")
        views = item.views
        if AudioView.WAVEFORM in views:
            value = views[AudioView.WAVEFORM]
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError("collated waveform view must be a pair.")
            waveform, sample_rates = value
            if not isinstance(waveform, Tensor) or not isinstance(sample_rates, Tensor):
                raise TypeError("collated waveform view must contain tensors.")
            lengths = batch.lengths(FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM))
            return waveform, sample_rates, lengths
        if AudioView.FILE in views:
            files = _audio_files(views[AudioView.FILE])
            waveforms: list[Tensor] = []
            sample_rates: list[int] = []
            torchaudio = _torchaudio()
            for file in files:
                waveform, sample_rate = torchaudio.load(_audio_source(file))
                waveforms.append(waveform)
                sample_rates.append(sample_rate)
            return _pad_waveforms(waveforms, sample_rates)
        raise ValueError("AudioProvider expects an audio waveform or file input view.")

    @staticmethod
    def _tensor(input: Tensor) -> Tensor:
        return input.detach().cpu().contiguous()

    @staticmethod
    def _length_groups(
        lengths: Tensor,
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        groups: dict[int, list[int]] = {}
        for index, value in enumerate(lengths.tolist()):
            groups.setdefault(int(value), []).append(index)
        return tuple((length, tuple(indexes)) for length, indexes in groups.items())


def _torchaudio():
    global torchaudio
    if torchaudio is not None:
        return torchaudio
    try:
        import torchaudio as loaded
    except ImportError as exc:
        raise ImportError(
            "AudioProvider file views require pip install anydataset[audio]."
        ) from exc
    torchaudio = loaded
    return torchaudio


def _audio_path(value: Any) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _audio_source(value: Any) -> BytesIO | str:
    if isinstance(value, FileBytes):
        return value.open()
    if isinstance(value, bytes):
        return BytesIO(value)
    if not isinstance(value, (str, Path)):
        raise TypeError("input must be FileBytes, bytes, or a filesystem path.")
    return str(_audio_path(value))


def _audio_files(value: Any) -> list[Any]:
    if isinstance(value, (str, Path, bytes, FileBytes)):
        return [value]
    if not isinstance(value, list):
        raise TypeError(
            "batched audio file view must be a list of paths, FileBytes, or bytes."
        )
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
        raise ValueError(
            "Only the last waveform dimension may vary in audio file batches."
        )

    device = tensors[0].device
    if any(waveform.device != device for waveform in tensors[1:]):
        raise ValueError("Audio file waveforms must share one device.")
    dtype = tensors[0].dtype
    for waveform in tensors[1:]:
        dtype = torch.promote_types(dtype, waveform.dtype)

    lengths = [waveform.shape[-1] for waveform in tensors]
    max_len = max(lengths)
    batch = tensors[0].new_zeros(
        (len(tensors), *prefix, max_len),
        dtype=dtype,
    )
    for index, waveform in enumerate(tensors):
        batch[index, ..., : waveform.shape[-1]].copy_(waveform)

    rates = torch.tensor(sample_rates, dtype=torch.int64, device=batch.device)
    length_tensor = torch.tensor(lengths, dtype=torch.int64, device=batch.device)
    return batch, rates, length_tensor
