"""Materialize frame-code audio views through shared AnyTrain contracts.

The providers accept waveform or file views and write complete ordered code
ids. ``AudioTokenizerProvider`` needs only the waveform-to-codes capability;
``CodecProvider`` retains a complete codec for callers that also decode. Both
own batching and frame trimming without interpreting codebook semantics.
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
    from anytrain.codec import AudioTokenizer, FrameCodec


class _FrameCodeProvider(nn.Module, AudioProvider):
    def __init__(self, output: AudioView) -> None:
        super().__init__()
        self.output = output

    @property
    def codebook_sizes(self) -> Sequence[int]:
        raise NotImplementedError

    @torch.inference_mode()
    def forward(self, views: Mapping[AudioView, Any]) -> torch.Tensor:
        waveform, sample_rate = self._audio_batch(views)
        codes = _codes(
            self._tokenize(waveform, sample_rate),
            self.codebook_sizes,
        )
        return self._tensor(codes[0])

    @torch.inference_mode()
    def call_batch(
        self,
        batch: Batch,
    ) -> (
        Sequence[torch.Tensor] | Mapping[tuple[Role, Modality], Sequence[torch.Tensor]]
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
        outputs: dict[int, torch.Tensor] = {}
        for length, indexes in self._length_groups(lengths):
            clipped = waveform[list(indexes), ..., :length].contiguous()
            codes = _codes(
                self._tokenize(clipped, sample_rate),
                self.codebook_sizes,
            )
            if codes.shape[0] != len(indexes):
                raise ValueError(
                    "Codec encode must return one output per input waveform."
                )
            codes = self._tensor(codes)
            outputs.update(
                (sample_index, codes[batch_index])
                for batch_index, sample_index in enumerate(indexes)
            )
        return [outputs[index] for index in range(len(lengths))]

    def _audio_batch(self, views: Mapping[AudioView, Any]) -> tuple[torch.Tensor, int]:
        waveform, sample_rate = self._waveform(views)
        waveform = (
            waveform
            if isinstance(waveform, torch.Tensor)
            else torch.as_tensor(waveform)
        )
        if waveform.is_floating_point():
            waveform = waveform.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.unsqueeze(0), sample_rate

    def _tokenize(self, audio: torch.Tensor, sample_rate: int) -> object:
        raise NotImplementedError


class AudioTokenizerProvider(_FrameCodeProvider):
    """Materialize one frame-code view from a tokenizer-only capability."""

    def __init__(
        self,
        tokenizer: AudioTokenizer[object],
        output: AudioView,
    ) -> None:
        spec = tokenizer.spec
        if spec.view != output.value:
            raise ValueError(
                f"Audio tokenizer spec view {spec.view!r} does not match "
                f"provider output {output.value!r}."
            )
        codebook_sizes = spec.frame_codebook_sizes
        if not codebook_sizes:
            raise ValueError(
                "AudioTokenizerProvider requires a frame-code tokenizer spec."
            )
        super().__init__(output)
        self._codebook_sizes = tuple(codebook_sizes)
        self.tokenizer = tokenizer
        if isinstance(tokenizer, nn.Module):
            tokenizer.eval()
        if isinstance(tokenizer.backend, nn.Module):
            tokenizer.backend.eval()

    def _tokenize(self, audio: torch.Tensor, sample_rate: int) -> object:
        return self.tokenizer.tokenize(audio, sample_rate)

    @property
    def codebook_sizes(self) -> Sequence[int]:
        return self._codebook_sizes


class CodecProvider(_FrameCodeProvider):
    """Materialize frame codes while retaining a complete codec for decoding."""

    def __init__(self, codec: FrameCodec, output: AudioView) -> None:
        super().__init__(output)
        self.codec = codec
        if isinstance(codec, nn.Module):
            codec.eval()

    def _tokenize(self, audio: torch.Tensor, sample_rate: int) -> object:
        return self.codec.encode(audio, sample_rate)

    @property
    def codebook_sizes(self) -> Sequence[int]:
        return self.codec.codebook_sizes


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


def _codes(codes: object, codebook_sizes: Sequence[int]) -> torch.Tensor:
    if not isinstance(codes, torch.Tensor):
        raise TypeError("Codec encode must return a Tensor.")
    if codes.ndim != 3:
        raise ValueError("Codec codes must have shape [batch, frame, codebook].")
    codebooks = len(codebook_sizes)
    if codes.shape[-1] != codebooks:
        raise ValueError(
            f"Codec codes must contain all configured {codebooks} codebooks."
        )
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


__all__ = ["AudioTokenizerProvider", "CodecProvider"]
