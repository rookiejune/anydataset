from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from ..types.item import AudioKey, AudioView, Modality
from ..store import ViewInput, ViewMaterializer
from ..store.manifest import ViewRef

DEFAULT_INPUT_REF = ViewRef(Modality.AUDIO, AudioView.WAVEFORM)
DEFAULT_OUTPUT_REF = ViewRef(Modality.AUDIO, AudioView.LONGCAT)
type LongCatDecoderName = Literal[
    "16k_4codebooks",
    "24k_2codebooks",
    "24k_4codebooks",
]


class LongCatCodec(Protocol):
    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
        *,
        n_acoustic_codebooks: int | None = None,
    ) -> tuple[Tensor, Tensor]: ...


@dataclass
class LongCatViewProvider:
    codec: LongCatCodec | None = None
    cache_dir: str | Path | None = None
    decoders: Sequence[LongCatDecoderName] = ("16k_4codebooks",)
    device: str | torch.device | None = None
    local_files_only: bool = False
    force_download: bool = False
    n_acoustic_codebooks: int | None = None
    provider_version: str = "anytrain.LongCatAudioCodec"
    _resolved_codec: LongCatCodec | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_acoustic_codebooks is not None and self.n_acoustic_codebooks < 1:
            raise ValueError("n_acoustic_codebooks must be positive.")
        if isinstance(self.decoders, str):
            raise TypeError(
                "decoders must be a sequence of decoder names, not a string."
            )
        if not self.decoders:
            raise ValueError("decoders must not be empty.")
        self.decoders = tuple(self.decoders)
        self._resolved_codec = self.codec

    @property
    def config(self) -> dict[str, Any]:
        return {"n_acoustic_codebooks": self.n_acoustic_codebooks}

    def __call__(self, view: ViewInput) -> dict[str, Any]:
        return _LongCatMaterializer(self)(view)

    def call_for_waveform(self, waveform: Any, *, sample_rate: int) -> dict[str, Any]:
        audio = _waveform(waveform)
        return self.call_for_batch(audio.unsqueeze(0), sample_rate=sample_rate)

    def call_for_batch(self, audio: Any, *, sample_rate: int) -> dict[str, Any]:
        tensor = audio.detach() if isinstance(audio, Tensor) else torch.as_tensor(audio)
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.float32)
        if tensor.ndim != 3:
            raise ValueError(
                "LongCat batch input must have shape [batch, channel, time]."
            )
        if tensor.shape[1] != 1:
            raise ValueError("LongCat batch input must have one channel.")
        semantic_codes, acoustic_codes = self._codec().encode(
            tensor.contiguous(),
            sample_rate,
            n_acoustic_codebooks=self.n_acoustic_codebooks,
        )
        return {
            "semantic_codes": _cpu_tensor(semantic_codes),
            "acoustic_codes": _cpu_tensor(acoustic_codes),
            "sample_rate": sample_rate,
        }

    def materializer(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        *,
        input_ref: ViewRef = DEFAULT_INPUT_REF,
        output_ref: ViewRef = DEFAULT_OUTPUT_REF,
    ) -> ViewMaterializer:
        return ViewMaterializer(
            input_dir=input_dir,
            output_dir=output_dir,
            input_ref=input_ref,
            output_ref=output_ref,
            transform=_LongCatMaterializer(self),
            provider_name="longcat",
            provider_version=self.provider_version,
            config=self.config,
        )

    def _codec(self) -> LongCatCodec:
        if self._resolved_codec is None:
            self._resolved_codec = _load_anytrain_codec(
                cache_dir=self.cache_dir,
                decoders=self.decoders,
                device=self.device,
                local_files_only=self.local_files_only,
                force_download=self.force_download,
            )
        return self._resolved_codec


@dataclass(frozen=True)
class _LongCatMaterializer:
    provider: LongCatViewProvider

    def __call__(self, view: ViewInput) -> dict[str, Any]:
        input_view = view.ref.view_key
        if input_view == AudioView.WAVEFORM:
            item = view.sample.item(view.ref.sample_ref)
            if item is None:
                raise ValueError(
                    "LongCatViewProvider requires an audio sample item with sample_rate."
                )
            sample_rate = item.required.get(AudioKey.SAMPLE_RATE)
            if sample_rate is None:
                raise ValueError("LongCatViewProvider requires audio sample_rate.")
            if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
                raise TypeError("LongCatViewProvider sample_rate must be an integer.")
            audio = _waveform(view.value)
            return self.provider.call_for_batch(
                audio.unsqueeze(0),
                sample_rate=sample_rate,
            )
        if input_view == AudioView.FILE:
            waveform, sample_rate = _load_audio(view.value)
            audio = _waveform(waveform)
            return self.provider.call_for_batch(
                audio.unsqueeze(0),
                sample_rate=sample_rate,
            )
        raise ValueError(
            "LongCatViewProvider expects an audio waveform or file input view."
        )


def _load_anytrain_codec(
    *,
    cache_dir: str | Path | None,
    decoders: Sequence[LongCatDecoderName],
    device: str | torch.device | None,
    local_files_only: bool,
    force_download: bool,
) -> LongCatCodec:
    try:
        from anytrain.codec.longcat import LongCatAudioCodec
    except ImportError as exc:
        raise ImportError(
            "LongCatViewProvider requires `anytrain[longcat]` when no codec is passed. "
            "Pass an initialized codec explicitly, or install anytrain with its longcat extra."
        ) from exc
    return LongCatAudioCodec.from_pretrained(
        cache_dir=cache_dir,
        decoders=decoders,
        device=device,
        local_files_only=local_files_only,
        force_download=force_download,
    )


def _load_audio(value: Any) -> tuple[Tensor, int]:
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "LongCatViewProvider file inputs require `pip install anydataset[audio]`."
        ) from exc

    source = BytesIO(value) if isinstance(value, bytes) else str(_audio_path(value))
    waveform, sample_rate = torchaudio.load(source)
    return waveform, int(sample_rate)


def _audio_path(value: Any) -> Path:
    if not isinstance(value, str | Path):
        raise TypeError("LongCat file input must be bytes or a filesystem path.")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _waveform(value: Any) -> Tensor:
    tensor = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 2:
        raise ValueError(
            "LongCat waveform input must have shape [time] or [channel, time]."
        )
    return tensor.contiguous()


def _cpu_tensor(value: Tensor) -> Tensor:
    return value.detach().cpu().contiguous()
