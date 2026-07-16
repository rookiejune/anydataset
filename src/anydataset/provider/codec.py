"""Materialize audio codec views through the shared anytrain contract.

The provider accepts waveform or file views and writes complete ordered codec
ids. It owns batching and frame trimming, but does not interpret codebook
semantics or codec-specific configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..dataset.collate import Batch
from ..types.item import AudioView, Modality, Role
from .abc import AudioProvider

if TYPE_CHECKING:
    from anytrain.codec import Codec


class CodecProvider(nn.Module, AudioProvider):
    def __init__(self, codec: Codec, output: AudioView) -> None:
        super().__init__()
        self.codec = codec
        self.output = output
        self.codec.eval()

    @torch.inference_mode()
    def forward(self, views: Mapping[AudioView, Any]) -> torch.Tensor:
        waveform, sample_rate = self._audio_batch(views)
        codes = _codes(
            self.codec.encode(waveform, sample_rate),
            self.codec.codebook_sizes,
        )
        return self._tensor(codes[0])

    @torch.inference_mode()
    def call_batch(
        self,
        batch: Batch,
    ) -> (
        Sequence[torch.Tensor]
        | Mapping[tuple[Role, Modality], Sequence[torch.Tensor]]
    ):
        refs = _audio_refs(batch)
        outputs = {ref: self._encode_ref_batch(batch, ref) for ref in refs}
        if len(refs) == 1:
            return outputs[refs[0]]
        return outputs

    def _encode_ref_batch(
        self,
        batch: Batch,
        ref: tuple[Role, Modality],
    ) -> Sequence[torch.Tensor]:
        waveform, sample_rates, lengths = self._waveform_batch(batch, ref)
        if waveform.is_floating_point():
            waveform = waveform.float()
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        sample_rate = _single_sample_rate(sample_rates)
        codes = _codes(
            self.codec.encode(waveform, sample_rate),
            self.codec.codebook_sizes,
        )
        return _split_codes(
            self._tensor(codes),
            lengths,
            waveform.shape[-1],
        )

    def _audio_batch(self, views: Mapping[AudioView, Any]) -> tuple[torch.Tensor, int]:
        waveform, sample_rate = self._waveform(views)
        waveform = (
            waveform if isinstance(waveform, torch.Tensor) else torch.as_tensor(waveform)
        )
        if waveform.is_floating_point():
            waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.unsqueeze(0), sample_rate


def _audio_refs(batch: Batch) -> tuple[tuple[Role, Modality], ...]:
    refs = tuple(
        ref
        for ref in batch.sample
        if ref[1] is Modality.AUDIO
        and (
            AudioView.WAVEFORM in batch.sample[ref].views
            or AudioView.FILE in batch.sample[ref].views
        )
    )
    if not refs:
        raise ValueError(
            "CodecProvider.call_batch expects at least one audio waveform input."
        )
    return refs


def _single_sample_rate(sample_rates: torch.Tensor) -> int:
    if sample_rates.ndim != 1:
        raise ValueError("Batched waveform sample rates must have shape [batch].")
    if sample_rates.numel() == 0:
        raise ValueError("Batched waveform sample rates must not be empty.")
    first = sample_rates[0].item()
    if not torch.equal(sample_rates, sample_rates.new_full(sample_rates.shape, first)):
        raise ValueError("CodecProvider.call_batch requires one sample rate per batch.")
    return int(first)


def _split_codes(
    codes: torch.Tensor,
    waveform_lengths: torch.Tensor,
    padded_waveform_length: int,
) -> list[torch.Tensor]:
    if padded_waveform_length <= 0:
        raise ValueError("Batched waveform must have a positive time dimension.")
    batch_size = int(waveform_lengths.numel())
    if codes.shape[0] != batch_size:
        raise ValueError("Codec codes batch size does not match waveform batch.")

    code_length = codes.shape[1]
    code_lengths = torch.div(
        waveform_lengths.cpu() * code_length + padded_waveform_length - 1,
        padded_waveform_length,
        rounding_mode="floor",
    )
    return [
        codes[index, : int(code_lengths[index].item())].contiguous()
        for index in range(batch_size)
    ]


def _codes(codes: torch.Tensor, codebook_sizes: Sequence[int]) -> torch.Tensor:
    if not isinstance(codes, torch.Tensor):
        raise TypeError("Codec encode must return a Tensor.")
    if codes.ndim != 3:
        raise ValueError("Codec codes must have shape [batch, frame, codebook].")
    codebooks = len(codebook_sizes)
    if codes.shape[-1] != codebooks:
        raise ValueError(f"Codec codes must contain all configured {codebooks} codebooks.")
    if codes.dtype == torch.bool or codes.is_floating_point() or codes.is_complex():
        raise TypeError("Codec codes must contain integer ids.")
    if codes.numel() == 0:
        return codes

    minimum = codes.amin(dim=(0, 1))
    maximum = codes.amax(dim=(0, 1))
    limits = torch.as_tensor(codebook_sizes, dtype=torch.int64, device=codes.device)
    invalid = (minimum < 0) | (maximum >= limits)
    if invalid.any().item():
        observed = torch.stack((minimum, maximum), dim=1).cpu().tolist()
        details = "; ".join(
            f"codebook {index} observed [{low}, {high}], expected [0, {size})"
            for index, ((low, high), size) in enumerate(zip(observed, codebook_sizes))
            if low < 0 or high >= size
        )
        raise ValueError(f"Codec code ids are outside configured ranges: {details}.")
    return codes


__all__ = ["CodecProvider"]
