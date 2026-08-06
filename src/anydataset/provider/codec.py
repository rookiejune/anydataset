"""Materialize native AnyTrain audio-token views from waveform inputs.

``AudioTokenizerProvider`` adapts AnyTrain's directional ``AudioTokenizer``
capability to canonical ``AudioItem`` views without changing its token schema.
``CodecProvider`` preserves the legacy complete-frame Tensor contract for
callers that intentionally retain a full frame codec.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Union

import torch
from torch import nn

from ..dataset.collate import Batch
from ..types.item import (
    AudioView,
    Modality,
    Role,
    SemanticAcousticView,
    SemanticGlobalView,
)
from .abc import AudioProvider

if TYPE_CHECKING:
    from anytrain.codec import AudioCodeSpec, AudioCodes, AudioTokenizer, FrameCodec


_TokenView = Union[torch.Tensor, SemanticAcousticView, SemanticGlobalView]
_SIGNED_INTEGER_DTYPES = frozenset({torch.int8, torch.int16, torch.int32, torch.int64})


class _BatchedAudioProvider(nn.Module, AudioProvider):
    _operation = "Audio provider"

    def __init__(self, output: AudioView) -> None:
        super().__init__()
        self.output = output

    @torch.inference_mode()
    def forward(self, views: Mapping[AudioView, Any]) -> Any:
        waveform, sample_rate = self._audio_batch(views)
        values = self._batch_values(waveform, sample_rate)
        if len(values) != 1:
            raise ValueError(
                f"{self._operation} must return one output per input waveform."
            )
        return values[0]

    @torch.inference_mode()
    def call_batch(
        self,
        batch: Batch,
    ) -> Sequence[Any] | Mapping[tuple[Role, Modality], Sequence[Any]]:
        refs = _audio_refs(batch)
        outputs = {ref: self._encode_ref_batch(batch, ref) for ref in refs}
        if len(refs) == 1:
            return outputs[refs[0]]
        return outputs

    def _encode_ref_batch(
        self,
        batch: Batch,
        ref: tuple[Role, Modality],
    ) -> Sequence[Any]:
        waveform, sample_rates, lengths = self._waveform_batch(batch, ref)
        waveform = _batched_waveform(waveform)
        outputs: dict[int, Any] = {}
        for sample_rate, length, indexes in self._batch_groups(
            sample_rates,
            lengths,
        ):
            clipped = waveform[list(indexes), ..., :length].contiguous()
            values = self._batch_values(clipped, sample_rate)
            if len(values) != len(indexes):
                raise ValueError(
                    f"{self._operation} must return one output per input waveform."
                )
            outputs.update(zip(indexes, values))
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
        if waveform.ndim != 2:
            raise ValueError(
                "Single audio waveform must have shape [time] or [channel, time]."
            )
        return waveform.unsqueeze(0), _sample_rate(sample_rate)

    def _batch_values(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> Sequence[Any]:
        raise NotImplementedError

    def _batch_groups(
        self,
        sample_rates: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        raise NotImplementedError


class _FrameCodeProvider(_BatchedAudioProvider):
    _operation = "Codec encode"

    @property
    def codebook_sizes(self) -> Sequence[int]:
        raise NotImplementedError

    def _batch_values(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> Sequence[torch.Tensor]:
        codes = _codes(
            self._tokenize(audio, sample_rate),
            self.codebook_sizes,
        )
        if codes.shape[0] != audio.shape[0]:
            raise ValueError("Codec encode must return one output per input waveform.")
        return tuple(self._tensor(value) for value in codes)

    def _batch_groups(
        self,
        sample_rates: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        sample_rate = _single_sample_rate(sample_rates)
        return tuple(
            (sample_rate, length, indexes)
            for length, indexes in self._length_groups(lengths)
        )

    def _tokenize(self, audio: torch.Tensor, sample_rate: int) -> object:
        raise NotImplementedError


class AudioTokenizerProvider(_BatchedAudioProvider):
    """Adapt one AnyTrain tokenizer to its declared native audio-code schema."""

    _operation = "Audio tokenizer"

    def __init__(
        self,
        tokenizer: AudioTokenizer[AudioCodes],
        output: AudioView,
    ) -> None:
        spec = tokenizer.spec
        if spec.view != output.value:
            raise ValueError(
                f"Audio tokenizer spec view {spec.view!r} does not match "
                f"provider output {output.value!r}."
            )
        schema = _schema(spec)
        _validate_spec(spec, schema)

        super().__init__(output)
        self.spec = spec
        self.tokenizer = tokenizer
        self._schema = schema
        if isinstance(tokenizer, nn.Module):
            tokenizer.eval()
        if isinstance(tokenizer.backend, nn.Module):
            tokenizer.backend.eval()

    def _batch_values(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> Sequence[_TokenView]:
        return _token_views(
            self.tokenizer.tokenize(audio, sample_rate),
            self.spec,
            self._schema,
            batch_size=audio.shape[0],
        )

    def _batch_groups(
        self,
        sample_rates: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        return _rate_length_groups(sample_rates, lengths)


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
            "Audio token provider call_batch expects at least one audio input."
        )
    return refs


def _batched_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.is_floating_point():
        waveform = waveform.float()
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    if waveform.ndim != 3:
        raise ValueError(
            "Batched audio waveform must have shape [batch, time] or "
            "[batch, channel, time]."
        )
    return waveform


def _sample_rate(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Audio waveform sample rate must be an integer.")
    if value <= 0:
        raise ValueError("Audio waveform sample rate must be positive.")
    return value


def _batch_axes(
    sample_rates: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[list[int], list[int]]:
    if sample_rates.ndim != 1:
        raise ValueError("Batched waveform sample rates must have shape [batch].")
    if lengths.ndim != 1:
        raise ValueError("Batched waveform lengths must have shape [batch].")
    if sample_rates.numel() == 0:
        raise ValueError("Batched waveform sample rates must not be empty.")
    if sample_rates.shape != lengths.shape:
        raise ValueError("Batched waveform sample rates and lengths must align.")
    rates = [int(value) for value in sample_rates.tolist()]
    sizes = [int(value) for value in lengths.tolist()]
    if any(value <= 0 for value in rates):
        raise ValueError("Batched waveform sample rates must be positive.")
    if any(value <= 0 for value in sizes):
        raise ValueError("Batched waveform lengths must be positive.")
    return rates, sizes


def _single_sample_rate(sample_rates: torch.Tensor) -> int:
    rates, _ = _batch_axes(
        sample_rates,
        torch.ones_like(sample_rates, dtype=torch.int64),
    )
    first = rates[0]
    if any(value != first for value in rates[1:]):
        raise ValueError("CodecProvider.call_batch requires one sample rate per batch.")
    return first


def _rate_length_groups(
    sample_rates: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    rates, sizes = _batch_axes(sample_rates, lengths)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(zip(rates, sizes)):
        groups.setdefault(key, []).append(index)
    return tuple(
        (sample_rate, length, tuple(indexes))
        for (sample_rate, length), indexes in groups.items()
    )


def _schema(spec: AudioCodeSpec) -> str:
    value = spec.schema
    if isinstance(value, str):
        schema = value
    else:
        schema = getattr(value, "value", None)
    if schema not in {"frame", "semantic_acoustic", "semantic_global"}:
        raise ValueError(f"Unsupported audio tokenizer schema: {schema!r}.")
    return schema


def _layout(spec: AudioCodeSpec) -> str:
    value = spec.acoustic_layout
    if isinstance(value, str):
        layout = value
    else:
        layout = getattr(value, "value", None)
    if layout not in {"frame_aligned", "fixed_length"}:
        raise ValueError(f"Unsupported semantic-acoustic layout: {layout!r}.")
    return layout


def _validate_spec(spec: AudioCodeSpec, schema: str) -> None:
    if schema == "frame":
        _codebook_sizes(spec.frame_codebook_sizes, name="frame_codebook_sizes")
        return
    _codebook_sizes(
        spec.semantic_codebook_sizes,
        name="semantic_codebook_sizes",
    )
    if schema == "semantic_acoustic":
        _codebook_sizes(
            spec.acoustic_codebook_sizes,
            name="acoustic_codebook_sizes",
        )
        layout = _layout(spec)
        if layout == "fixed_length":
            _unit_length(
                spec.acoustic_unit_length,
                name="acoustic_unit_length",
            )
        return
    _codebook_sizes(spec.global_codebook_sizes, name="global_codebook_sizes")
    _unit_length(spec.global_unit_length, name="global_unit_length")


def _codebook_sizes(value: Sequence[int], *, name: str) -> None:
    if not value:
        raise ValueError(f"Audio tokenizer spec requires {name}.")
    if any(isinstance(size, bool) or not isinstance(size, int) for size in value):
        raise TypeError(f"Audio tokenizer spec {name} must contain integers.")
    if any(size <= 0 for size in value):
        raise ValueError(f"Audio tokenizer spec {name} must be positive.")


def _unit_length(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Audio tokenizer spec {name} must be an integer.")
    if value <= 0:
        raise ValueError(f"Audio tokenizer spec {name} must be positive.")
    return value


def _token_views(
    codes: object,
    spec: AudioCodeSpec,
    schema: str,
    *,
    batch_size: int,
) -> tuple[_TokenView, ...]:
    if schema == "frame":
        values = _token_tensor(
            codes,
            spec.frame_codebook_sizes,
            name="Frame codes",
            batch_size=batch_size,
        )
        return tuple(_cpu(value) for value in values)
    if schema == "semantic_acoustic":
        return _semantic_acoustic_views(codes, spec, batch_size=batch_size)
    return _semantic_global_views(codes, spec, batch_size=batch_size)


def _semantic_acoustic_views(
    codes: object,
    spec: AudioCodeSpec,
    *,
    batch_size: int,
) -> tuple[SemanticAcousticView, ...]:
    try:
        from anytrain.codec import SemanticAcousticCodes
    except ImportError as exc:
        raise ImportError(
            "Semantic-acoustic token materialization requires the current AnyTrain "
            "package."
        ) from exc
    if not isinstance(codes, SemanticAcousticCodes):
        raise TypeError(
            "Semantic-acoustic tokenizer must return SemanticAcousticCodes."
        )
    semantic = _token_tensor(
        codes.semantic,
        spec.semantic_codebook_sizes,
        name="Semantic codes",
        batch_size=batch_size,
    )
    acoustic = _token_tensor(
        codes.acoustic,
        spec.acoustic_codebook_sizes,
        name="Acoustic codes",
        batch_size=batch_size,
    )
    layout = _layout(spec)
    if layout == "frame_aligned":
        if semantic.shape[:2] != acoustic.shape[:2]:
            raise ValueError(
                "Frame-aligned semantic and acoustic codes must align on batch "
                "and time."
            )
    else:
        unit_length = _unit_length(
            spec.acoustic_unit_length,
            name="acoustic_unit_length",
        )
        if acoustic.shape[1] != unit_length:
            raise ValueError(
                "Fixed-length acoustic codes must use the configured unit length "
                f"{unit_length}, got {acoustic.shape[1]}."
            )
    return tuple(
        {
            "semantic": _cpu(semantic[index]),
            "acoustic": _cpu(acoustic[index]),
        }
        for index in range(batch_size)
    )


def _semantic_global_views(
    codes: object,
    spec: AudioCodeSpec,
    *,
    batch_size: int,
) -> tuple[SemanticGlobalView, ...]:
    try:
        from anytrain.codec import SemanticGlobalCodes
    except ImportError as exc:
        raise ImportError(
            "Semantic-global token materialization requires the current AnyTrain "
            "package."
        ) from exc
    if not isinstance(codes, SemanticGlobalCodes):
        raise TypeError("Semantic-global tokenizer must return SemanticGlobalCodes.")
    semantic = _token_tensor(
        codes.semantic,
        spec.semantic_codebook_sizes,
        name="Semantic codes",
        batch_size=batch_size,
    )
    global_codes = _token_tensor(
        codes.global_codes,
        spec.global_codebook_sizes,
        name="Global codes",
        batch_size=batch_size,
    )
    unit_length = _unit_length(
        spec.global_unit_length,
        name="global_unit_length",
    )
    if global_codes.shape[1] != unit_length:
        raise ValueError(
            "Global codes must use the configured unit length "
            f"{unit_length}, got {global_codes.shape[1]}."
        )
    return tuple(
        {
            "semantic": _cpu(semantic[index]),
            "global": _cpu(global_codes[index]),
        }
        for index in range(batch_size)
    )


def _token_tensor(
    value: object,
    codebook_sizes: Sequence[int],
    *,
    name: str,
    batch_size: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, unit, codebook].")
    if value.shape[0] != batch_size:
        raise ValueError(f"{name} must return one output per input waveform.")
    if value.shape[-1] != len(codebook_sizes):
        raise ValueError(
            f"{name} must contain all configured {len(codebook_sizes)} codebooks."
        )
    if value.dtype not in _SIGNED_INTEGER_DTYPES:
        raise TypeError(f"{name} must contain signed integer ids.")
    _validate_ranges(value, codebook_sizes, name=name)
    return value


def _validate_ranges(
    codes: torch.Tensor,
    codebook_sizes: Sequence[int],
    *,
    name: str,
) -> None:
    if codes.numel() == 0:
        return
    minimum = codes.amin(dim=(0, 1))
    maximum = codes.amax(dim=(0, 1))
    limits = torch.as_tensor(
        codebook_sizes,
        dtype=torch.int64,
        device=codes.device,
    )
    invalid = (minimum < 0) | (maximum >= limits)
    if not invalid.any().item():
        return
    observed = torch.stack((minimum, maximum), dim=1).cpu().tolist()
    details = "; ".join(
        f"codebook {index} observed [{low}, {high}], expected [0, {size})"
        for index, ((low, high), size) in enumerate(zip(observed, codebook_sizes))
        if low < 0 or high >= size
    )
    raise ValueError(f"{name} ids are outside configured ranges: {details}.")


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous()


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
