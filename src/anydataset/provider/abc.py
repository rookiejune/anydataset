from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from torch import Tensor

try:
    import torchaudio
except ImportError as exc:
    raise ImportError(
        "AudioProvider requires `pip install anydataset[audio]`."
    ) from exc


from ..types import AudioView


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
    if not isinstance(value, str | Path):
        raise TypeError("input must be bytes or a filesystem path.")
    return str(_audio_path(value))
