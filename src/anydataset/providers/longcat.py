from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from ..modalities import ModalityKey, ViewRef
from ..modalities.audio import AudioView
from ..store import ViewInput, ViewMaterializer


DEFAULT_INPUT_REF = ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM)
DEFAULT_OUTPUT_REF = ViewRef(ModalityKey.AUDIO, AudioView.LONGCAT)


class LongCatCodec(Protocol):
    def encode(
        self,
        audio: Tensor,
        sample_rate: int | None = None,
        *,
        n_acoustic_codebooks: int | None = None,
    ) -> tuple[Tensor, Tensor | None]: ...


@dataclass
class LongCatViewProvider:
    codec: LongCatCodec | None = None
    cache_dir: str | Path | None = None
    decoders: Sequence[str] = ("16k_4codebooks",)
    device: str | torch.device | None = None
    local_files_only: bool = False
    force_download: bool = False
    n_acoustic_codebooks: int | None = None
    provider_version: str = "anytrain.LongCatAudioCodec"
    _resolved_codec: LongCatCodec | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_acoustic_codebooks is not None and self.n_acoustic_codebooks < 0:
            raise ValueError("n_acoustic_codebooks must be non-negative.")
        if isinstance(self.decoders, str):
            raise TypeError("decoders must be a sequence of decoder names, not a string.")
        if not self.decoders:
            raise ValueError("decoders must not be empty.")
        self.decoders = tuple(self.decoders)
        self._resolved_codec = self.codec

    @property
    def config(self) -> dict[str, Any]:
        return {"n_acoustic_codebooks": self.n_acoustic_codebooks}

    def __call__(self, view: ViewInput) -> dict[str, Any]:
        if view.ref.view_key != AudioView.WAVEFORM:
            raise ValueError("LongCatViewProvider expects an audio waveform input view.")
        sample_rate = _sample_rate(view)
        audio = _waveform(view.value)
        semantic_codes, acoustic_codes = self._codec().encode(
            audio,
            sample_rate,
            n_acoustic_codebooks=self.n_acoustic_codebooks,
        )
        return {
            "semantic_codes": _cpu_tensor(semantic_codes),
            "acoustic_codes": None if acoustic_codes is None else _cpu_tensor(acoustic_codes),
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
            transform=self,
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


def _load_anytrain_codec(
    *,
    cache_dir: str | Path | None,
    decoders: Sequence[str],
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


def _sample_rate(view: ViewInput) -> int:
    sample_rate = view.sample.sample_rate
    if sample_rate is None:
        raise ValueError("LongCatViewProvider requires sample.sample_rate.")
    return sample_rate


def _waveform(value: Any) -> Tensor:
    tensor = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 3:
        raise ValueError("LongCat waveform input must have shape [time], [channel, time], or [batch, channel, time].")
    return tensor.contiguous()


def _cpu_tensor(value: Tensor) -> Tensor:
    return value.detach().cpu().contiguous()
