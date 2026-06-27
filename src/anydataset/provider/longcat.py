from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from ..dataset.collate import Batch, FieldGroup, FieldRef
from ..types.item import Modality, Role
from ..types.item import AudioView
from .abc import AudioProvider

type LongCatDecoderName = Literal[
    "16k_4codebooks",
    "24k_2codebooks",
    "24k_4codebooks",
]


class LongCatProvider(nn.Module, AudioProvider):
    output = AudioView.LONGCAT

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        decoders: Sequence[LongCatDecoderName] = ("16k_4codebooks",),
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        n_acoustic_codebooks: int | None = None,
    ) -> None:
        super().__init__()
        try:
            from anytrain.codec.longcat import LongCatAudioCodec
        except ImportError as exc:
            raise ImportError(
                "LongCatProvider requires `anytrain[longcat]`; install anytrain with "
                "its longcat extra."
            ) from exc

        self.longcat_codec = LongCatAudioCodec.from_pretrained(
            cache_dir=cache_dir,
            decoders=decoders,
            device=device,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        self.n_acoustic_codebooks = n_acoustic_codebooks

        self.longcat_codec.eval()

    @torch.inference_mode()
    def forward(self, views: Mapping[AudioView, Any]):
        waveform, sample_rate = self._longcat_batch(views)
        semantic_codes, acoustic_codes = self._encode(
            waveform,
            sample_rate,
        )
        return _align_code_lengths(
            {
                "semantic_codes": self._tensor(semantic_codes)[0],
                "acoustic_codes": self._tensor(acoustic_codes)[0],
            }
        )

    @torch.inference_mode()
    def call_batch(self, batch: Batch) -> Sequence[dict[str, torch.Tensor]]:
        ref = _audio_ref(batch)
        audio = batch.sample[ref]
        waveform, sample_rates = audio.views[AudioView.WAVEFORM]
        if waveform.is_floating_point():
            waveform = waveform.float()
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        sample_rate = _single_sample_rate(sample_rates)
        semantic_codes, acoustic_codes = self._encode(
            waveform,
            sample_rate,
        )
        lengths = batch.lengths(FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM))
        return _split_codes(
            {
                "semantic_codes": self._tensor(semantic_codes),
                "acoustic_codes": self._tensor(acoustic_codes),
            },
            lengths,
            waveform.shape[-1],
        )

    def _longcat_batch(self, views: Mapping[AudioView, Any]):
        waveform, sample_rate = self._waveform(views)
        waveform = (
            waveform if isinstance(waveform, torch.Tensor) else torch.as_tensor(waveform)
        )
        if waveform.is_floating_point():
            waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.unsqueeze(0), sample_rate

    def _encode(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_acoustic_codebooks is None:
            return self.longcat_codec.encode(waveform, sample_rate)
        return self.longcat_codec.encode(
            waveform,
            sample_rate,
            n_acoustic_codebooks=self.n_acoustic_codebooks,
        )


def _audio_ref(batch: Batch) -> tuple[Role, Modality]:
    refs = tuple(
        ref
        for ref in batch.sample
        if ref[1] is Modality.AUDIO
        and AudioView.WAVEFORM in batch.sample[ref].views
    )
    if len(refs) != 1:
        raise ValueError(
            "LongCatProvider.call_batch expects exactly one audio waveform input."
        )
    return refs[0]


def _single_sample_rate(sample_rates: torch.Tensor) -> int:
    if sample_rates.ndim != 1:
        raise ValueError("Batched waveform sample rates must have shape [batch].")
    if sample_rates.numel() == 0:
        raise ValueError("Batched waveform sample rates must not be empty.")
    first = sample_rates[0].item()
    if not torch.equal(sample_rates, sample_rates.new_full(sample_rates.shape, first)):
        raise ValueError("LongCatProvider.call_batch requires one sample rate per batch.")
    return int(sample_rates[0].item())


def _split_codes(
    codes: Mapping[str, torch.Tensor],
    waveform_lengths: torch.Tensor,
    padded_waveform_length: int,
) -> list[dict[str, torch.Tensor]]:
    if padded_waveform_length <= 0:
        raise ValueError("Batched waveform must have a positive time dimension.")
    aligned = _align_code_lengths(codes)
    batch_size = int(waveform_lengths.numel())
    for name, code in aligned.items():
        if code.shape[0] != batch_size:
            raise ValueError(
                f"LongCat code {name!r} batch size does not match waveform batch."
            )

    code_length = min(code.shape[-1] for code in aligned.values())
    code_lengths = torch.div(
        waveform_lengths.cpu() * code_length + padded_waveform_length - 1,
        padded_waveform_length,
        rounding_mode="floor",
    )
    return [
        {
            name: value[index, ..., : int(code_lengths[index].item())].contiguous()
            for name, value in aligned.items()
        }
        for index in range(batch_size)
    ]


def _align_code_lengths(codes: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    lengths = []
    for name, code in codes.items():
        if code.ndim == 0:
            raise ValueError(f"LongCat code {name!r} must have a time dimension.")
        lengths.append(code.shape[-1])

    length = min(lengths)
    return {name: code[..., :length].contiguous() for name, code in codes.items()}
