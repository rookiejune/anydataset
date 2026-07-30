"""Speech quality predicate for waveform or frame-codec speech samples."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
import unicodedata
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import Tensor

from ..filter import FilterDecision
from ..filter.types import JsonValue
from ..types import AudioItem, AudioView, Modality, Role, Sample, TextItem, TextView
from .rules import QualityLabel

if TYPE_CHECKING:
    from ..provider.codec import CodecProvider


class SpeechEvaluatorProtocol(Protocol):
    def __call__(
        self,
        audio: Any,
        sample_rate: int,
        reference_text: str,
        **decode_options: Any,
    ) -> Mapping[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class SpeechQualityProfile:
    min_utmos: float = 2.8
    max_wer: float | None = None
    min_chrf: float = 50.0
    max_seconds_per_text_unit: float | None = 4.0
    min_peak_amplitude: float | None = 0.05
    min_bleu: float | None = None

    def __post_init__(self) -> None:
        _finite_threshold(self.min_utmos, name="min_utmos")
        if self.max_wer is not None:
            _finite_threshold(self.max_wer, name="max_wer")
        _finite_threshold(self.min_chrf, name="min_chrf")
        if self.max_seconds_per_text_unit is not None:
            _finite_threshold(
                self.max_seconds_per_text_unit,
                name="max_seconds_per_text_unit",
            )
        if self.min_peak_amplitude is not None:
            _finite_threshold(self.min_peak_amplitude, name="min_peak_amplitude")
        if self.min_bleu is not None:
            _finite_threshold(self.min_bleu, name="min_bleu")


@dataclass
class SpeechQuality:
    profile: SpeechQualityProfile = field(default_factory=SpeechQualityProfile)
    evaluator: SpeechEvaluatorProtocol | None = None
    decode_options: Mapping[str, Any] = field(default_factory=dict)
    codec_provider: CodecProvider | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decode_options, Mapping):
            raise TypeError("decode_options must be a mapping.")
        self.decode_options = dict(self.decode_options)
        if self.codec_provider is not None and not isinstance(
            self.codec_provider.output, AudioView
        ):
            raise TypeError("codec_provider.output must be an AudioView.")

    def __call__(self, sample: Sample) -> FilterDecision:
        if self.codec_provider is not None:
            return self.call_batch((sample,))[0]
        return self._waveform_decision(sample)

    def call_batch(self, samples: Sequence[Sample]) -> Sequence[FilterDecision]:
        if self.codec_provider is None:
            return tuple(self._waveform_decision(sample) for sample in samples)
        return self._codec_decisions(samples)

    def _waveform_decision(self, sample: Sample) -> FilterDecision:
        state = _SpeechState()

        for role, item in _audio_items(sample):
            state.audio_count += 1
            audio, sample_rate, audio_warning = _waveform(item)
            reference_text, text_warning = _reference_text(sample, role)
            _record_audio_warning(state, role, audio_warning)
            _record_text_warning(state, role, text_warning)
            if audio_warning is not None or text_warning is not None:
                continue

            self._evaluate(state, role, audio, sample_rate, reference_text)

        return state.decision()

    def _codec_decisions(
        self,
        samples: Sequence[Sample],
    ) -> tuple[FilterDecision, ...]:
        provider = self.codec_provider
        if provider is None:
            raise RuntimeError("codec_provider is required for codec evaluation.")
        sample_rate = _sample_rate(provider.codec.sample_rate)
        states = [_SpeechState() for _sample in samples]
        inputs: list[_CodecInput] = []

        for sample_index, sample in enumerate(samples):
            state = states[sample_index]
            for role, item in _audio_items(sample):
                state.audio_count += 1
                codes = _codec_codes(item, provider.output)
                reference_text, text_warning = _reference_text(sample, role)
                _record_text_warning(state, role, text_warning)
                if text_warning is not None:
                    continue
                inputs.append(
                    _CodecInput(
                        sample_index=sample_index,
                        role=role,
                        codes=codes,
                        reference_text=reference_text,
                    )
                )

        decoded = _decode(provider, inputs)
        for input, audio in zip(inputs, decoded):
            state = states[input.sample_index]
            _audio, _rate, audio_warning = _waveform_value((audio, sample_rate))
            _record_audio_warning(state, input.role, audio_warning)
            if audio_warning is not None:
                continue
            self._evaluate(
                state,
                input.role,
                _audio,
                _rate,
                input.reference_text,
            )

        return tuple(state.decision() for state in states)

    def _evaluate(
        self,
        state: _SpeechState,
        role: Role,
        audio: Any,
        sample_rate: int,
        reference_text: str,
    ) -> None:
        metrics = self._evaluator()(
            audio,
            sample_rate,
            reference_text=reference_text,
            **self.decode_options,
        )
        values = _metrics(metrics)
        values.update(_audio_metrics(audio, sample_rate, reference_text))
        item_flags = _flags(values, self.profile)
        state.flags.extend(_role_key(role, flag) for flag in item_flags)
        state.checked_count += 1
        state.items.append(_item_log(role, reference_text, values, item_flags))

    def _evaluator(self) -> SpeechEvaluatorProtocol:
        if self.evaluator is not None:
            return self.evaluator
        self.evaluator = _default_evaluator()
        return self.evaluator


@dataclass
class _SpeechState:
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    items: list[Mapping[str, JsonValue]] = field(default_factory=list)
    audio_count: int = 0
    checked_count: int = 0

    def decision(self) -> FilterDecision:
        warnings = (
            [*self.warnings, "no_audio"] if self.audio_count == 0 else self.warnings
        )
        label = QualityLabel.REJECT if self.flags else QualityLabel.ACCEPT
        return _decision(
            label,
            flags=self.flags,
            warnings=warnings,
            audio_count=self.audio_count,
            checked_count=self.checked_count,
            items=self.items,
        )


@dataclass(frozen=True)
class _CodecInput:
    sample_index: int
    role: Role
    codes: Tensor
    reference_text: str


def _default_evaluator() -> SpeechEvaluatorProtocol:
    try:
        from anytrain.evaluator.speech import SpeechEvaluator
    except ImportError as exc:
        raise ImportError(
            "SpeechQuality requires `anytrain[speech]` when evaluator is not provided."
        ) from exc
    return SpeechEvaluator()


def _audio_items(sample: Sample) -> tuple[tuple[Role, AudioItem], ...]:
    output: list[tuple[Role, AudioItem]] = []
    for reference, item in sample.items():
        role, modality = reference
        if modality == Modality.AUDIO and isinstance(item, AudioItem):
            output.append((role, item))
    return tuple(output)


def _waveform(item: AudioItem) -> tuple[Any, int, str | None]:
    return _waveform_value(item.views.get(AudioView.WAVEFORM))


def _waveform_value(value: object) -> tuple[Any, int, str | None]:
    if value is None:
        return None, 0, "missing_waveform"
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None, 0, "invalid_waveform"
    audio, sample_rate = value
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        return None, 0, "invalid_sample_rate"
    if sample_rate <= 0:
        return None, 0, "invalid_sample_rate"
    try:
        wave = torch.as_tensor(audio)
    except (TypeError, ValueError):
        return None, 0, "invalid_waveform"
    if not bool(torch.isfinite(wave).all().item()):
        return None, 0, "non_finite_waveform"
    return audio, sample_rate, None


def _codec_codes(item: AudioItem, view: AudioView) -> Tensor:
    if view not in item.views:
        raise ValueError(
            f"SpeechQuality codec provider expects audio view {view.value!r}."
        )
    codes = item.views[view]
    if not isinstance(codes, Tensor):
        raise TypeError(f"Audio view {view.value!r} must contain codec Tensor ids.")
    if codes.ndim != 2:
        raise ValueError(
            f"Audio view {view.value!r} codec ids must have shape [frame, codebook]."
        )
    if codes.dtype == torch.bool or codes.is_floating_point() or codes.is_complex():
        raise TypeError(f"Audio view {view.value!r} codec ids must be integers.")
    return codes


def _sample_rate(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("codec_provider.codec.sample_rate must be an integer.")
    if value <= 0:
        raise ValueError("codec_provider.codec.sample_rate must be positive.")
    return value


def _decode(
    provider: CodecProvider,
    inputs: Sequence[_CodecInput],
) -> tuple[Tensor, ...]:
    if not inputs:
        return ()

    groups: dict[int, list[int]] = defaultdict(list)
    for index, input in enumerate(inputs):
        groups[input.codes.shape[0]].append(index)

    decoded: list[Tensor | None] = [None] * len(inputs)
    for indexes in groups.values():
        codes = _stack_codes(tuple(inputs[index].codes for index in indexes))
        with torch.inference_mode():
            audio = provider.codec.decode(codes)
        waveforms = _decoded_waveforms(audio, expected=len(indexes))
        for index, waveform in zip(indexes, waveforms):
            decoded[index] = waveform

    if any(waveform is None for waveform in decoded):
        raise RuntimeError("codec decode did not produce every requested waveform.")
    return tuple(waveform for waveform in decoded if waveform is not None)


def _stack_codes(codes: Sequence[Tensor]) -> Tensor:
    first = codes[0]
    if any(value.shape != first.shape for value in codes[1:]):
        raise ValueError("Codec ids with the same frame length must share one shape.")
    if any(value.dtype != first.dtype for value in codes[1:]):
        raise TypeError("Batched codec ids must share one dtype.")
    if any(value.device != first.device for value in codes[1:]):
        raise ValueError("Batched codec ids must share one device.")
    return torch.stack(tuple(codes))


def _decoded_waveforms(value: object, *, expected: int) -> tuple[Tensor, ...]:
    if not isinstance(value, Tensor):
        raise TypeError("codec_provider.codec.decode() must return a Tensor.")
    if value.ndim not in {2, 3}:
        raise ValueError(
            "codec_provider.codec.decode() must return audio with shape "
            "[batch, time] or [batch, channel, time]."
        )
    if value.shape[0] != expected:
        raise ValueError(
            "codec_provider.codec.decode() returned "
            f"{value.shape[0]} waveforms for {expected} codec inputs."
        )
    return tuple(value[index] for index in range(expected))


def _record_audio_warning(
    state: _SpeechState,
    role: Role,
    warning: str | None,
) -> None:
    if warning is None:
        return
    output = _role_key(role, warning)
    if warning == "non_finite_waveform":
        state.flags.append(output)
    else:
        state.warnings.append(output)


def _record_text_warning(
    state: _SpeechState,
    role: Role,
    warning: str | None,
) -> None:
    if warning is not None:
        state.warnings.append(_role_key(role, warning))


def _reference_text(sample: Sample, role: Role) -> tuple[str, str | None]:
    item = sample.get((role, Modality.TEXT))
    if not isinstance(item, TextItem):
        return "", "missing_text"

    text = item.views.get(TextView.TEXT)
    if text is None:
        return "", "missing_text_view"
    if not isinstance(text, str):
        return "", "invalid_text_view"
    text = _normalize_text(text)
    if text == "":
        return "", "empty_text"
    return text, None


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    return {
        "utmos": _metric(metrics, "utmos"),
        "wer": _metric(metrics, "wer"),
        "chrf": _metric(metrics, "chrf"),
        "bleu": _metric(metrics, "bleu"),
    }


def _audio_metrics(
    audio: Any, sample_rate: int, reference_text: str
) -> dict[str, float]:
    wave = torch.as_tensor(audio)
    if wave.numel() == 0:
        duration_seconds = 0.0
        peak_amplitude = 0.0
    else:
        duration_seconds = float(wave.shape[-1]) / float(sample_rate)
        peak_amplitude = float(wave.detach().abs().max().cpu().item())

    text_units = _text_units(reference_text)
    seconds_per_text_unit = duration_seconds / float(text_units)
    return {
        "duration_seconds": duration_seconds,
        "peak_amplitude": peak_amplitude,
        "text_units": float(text_units),
        "seconds_per_text_unit": seconds_per_text_unit,
    }


def _text_units(text: str) -> int:
    count = 0
    in_word = False
    for char in text:
        if _is_cjk(char):
            count += 1
            in_word = False
        elif char.isalnum():
            if not in_word:
                count += 1
                in_word = True
        elif unicodedata.category(char).startswith("M"):
            continue
        else:
            in_word = False
    return max(count, 1)


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x2CEB0 <= codepoint <= 0x2EBEF
        or 0x30000 <= codepoint <= 0x3134F
        or 0x31350 <= codepoint <= 0x323AF
    )


def _metric(metrics: Mapping[str, object], name: str) -> float:
    if name not in metrics:
        raise ValueError(f"speech evaluator must return metric {name!r}.")
    value = metrics[name]
    if isinstance(value, bool):
        raise TypeError(f"speech metric {name!r} must be a float.")
    if isinstance(value, (int, float)):
        output = float(value)
    elif isinstance(value, Tensor):
        if value.ndim != 0:
            raise ValueError(f"speech metric {name!r} must be a 0-d tensor.")
        output = float(value.detach().cpu().item())
    else:
        raise TypeError(f"speech metric {name!r} must be a float.")

    if not isfinite(output):
        raise ValueError(f"speech metric {name!r} must be finite.")
    return output


def _flags(metrics: Mapping[str, float], profile: SpeechQualityProfile) -> list[str]:
    flags: list[str] = []
    if metrics["utmos"] < profile.min_utmos:
        flags.append("utmos_low")
    if profile.max_wer is not None and metrics["wer"] > profile.max_wer:
        flags.append("wer_high")
    if metrics["chrf"] < profile.min_chrf:
        flags.append("chrf_low")
    if (
        profile.max_seconds_per_text_unit is not None
        and metrics["seconds_per_text_unit"] > profile.max_seconds_per_text_unit
    ):
        flags.append("duration_per_text_unit_high")
    if (
        profile.min_peak_amplitude is not None
        and metrics["peak_amplitude"] < profile.min_peak_amplitude
    ):
        flags.append("peak_amplitude_low")
    if profile.min_bleu is not None and metrics["bleu"] < profile.min_bleu:
        flags.append("bleu_low")
    return flags


def _decision(
    label: QualityLabel,
    *,
    flags: list[str],
    warnings: list[str],
    audio_count: int,
    checked_count: int,
    items: list[Mapping[str, JsonValue]],
) -> FilterDecision:
    output: dict[str, JsonValue] = {
        "decision": label.value,
        "flags": flags,
        "warnings": warnings,
        "audio_count": audio_count,
        "checked_count": checked_count,
        "items": list(items),
    }
    return FilterDecision(label=label, metrics=output)


def _item_log(
    role: Role,
    reference_text: str,
    metrics: Mapping[str, float],
    flags: list[str],
) -> Mapping[str, JsonValue]:
    return {
        "role": role.value,
        "reference_text": reference_text,
        "utmos": round(metrics["utmos"], 6),
        "wer": round(metrics["wer"], 6),
        "chrf": round(metrics["chrf"], 6),
        "bleu": round(metrics["bleu"], 6),
        "duration_seconds": round(metrics["duration_seconds"], 6),
        "peak_amplitude": round(metrics["peak_amplitude"], 6),
        "text_units": int(metrics["text_units"]),
        "seconds_per_text_unit": round(metrics["seconds_per_text_unit"], 6),
        "flags": flags,
    }


def _role_key(role: Role, value: str) -> str:
    return f"{role.value}_{value}"


def _finite_threshold(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float.")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


__all__ = ["QualityLabel", "SpeechQuality", "SpeechQualityProfile"]
