"""Speech quality predicate for canonical speech samples.

The predicate scans every audio item in a canonical `Sample`, evaluates audio
items that expose a waveform and same-role reference text with
`anytrain.evaluator.speech.SpeechEvaluator`, and returns an `accept` or
`reject` filter label with lightweight audit metrics. It does not own dataset
loading, filter cache construction, or speech model configuration beyond
explicit evaluator/decode options.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum, auto
from math import isfinite
import unicodedata
from typing import Any, Protocol

import torch
from torch import Tensor

from ..filter import FilterDecision
from ..filter.types import JsonValue
from ..types import AudioItem, AudioView, Modality, Role, Sample, TextItem, TextView


class Label(StrEnum):
    ACCEPT = auto()
    REJECT = auto()


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
class Profile:
    min_utmos: float = 3.0
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
class Predicate:
    profile: Profile = field(default_factory=Profile)
    evaluator: SpeechEvaluatorProtocol | None = None
    decode_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decode_options, Mapping):
            raise TypeError("decode_options must be a mapping.")
        self.decode_options = dict(self.decode_options)

    def __call__(self, sample: Sample) -> FilterDecision:
        flags: list[str] = []
        warnings: list[str] = []
        items: list[Mapping[str, JsonValue]] = []
        audio_count = 0
        checked_count = 0

        for role, item in _audio_items(sample):
            audio_count += 1
            audio, sample_rate, audio_warning = _waveform(item)
            reference_text, text_warning = _reference_text(sample, role)
            if audio_warning is not None:
                warnings.append(_role_key(role, audio_warning))
            if text_warning is not None:
                warnings.append(_role_key(role, text_warning))
            if audio_warning is not None or text_warning is not None:
                continue

            metrics = self._evaluator()(
                audio,
                sample_rate,
                reference_text=reference_text,
                **self.decode_options,
            )
            values = _metrics(metrics)
            values.update(_audio_metrics(audio, sample_rate, reference_text))
            item_flags = _flags(values, self.profile)
            flags.extend(_role_key(role, flag) for flag in item_flags)
            checked_count += 1
            items.append(_item_log(role, reference_text, values, item_flags))

        if audio_count == 0:
            warnings.append("no_audio")

        label = Label.REJECT if flags else Label.ACCEPT
        return _decision(
            label,
            flags=flags,
            warnings=warnings,
            audio_count=audio_count,
            checked_count=checked_count,
            items=items,
        )

    def _evaluator(self) -> SpeechEvaluatorProtocol:
        if self.evaluator is not None:
            return self.evaluator
        self.evaluator = _default_evaluator()
        return self.evaluator


def _default_evaluator() -> SpeechEvaluatorProtocol:
    try:
        from anytrain.evaluator.speech import SpeechEvaluator
    except ImportError as exc:
        raise ImportError(
            "Speech quality Predicate requires `anytrain[speech]` when evaluator is "
            "not provided."
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
    value = item.views.get(AudioView.WAVEFORM)
    if value is None:
        return None, 0, "missing_waveform"
    if not isinstance(value, tuple | list) or len(value) != 2:
        return None, 0, "invalid_waveform"
    audio, sample_rate = value
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        return None, 0, "invalid_sample_rate"
    if sample_rate <= 0:
        return None, 0, "invalid_sample_rate"
    return audio, sample_rate, None


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


def _audio_metrics(audio: Any, sample_rate: int, reference_text: str) -> dict[str, float]:
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
    if isinstance(value, int | float):
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


def _flags(metrics: Mapping[str, float], profile: Profile) -> list[str]:
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
    label: Label,
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
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a float.")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


__all__ = ["Label", "Predicate", "Profile"]
