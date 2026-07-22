"""Atomic quality predicates for canonical machine-translation samples.

This module only owns source/target pair checks and optional pair-level model
rules. Single-text checks are provided by anydataset.quality.text, and cross-rule
label transitions are provided by anydataset.quality.rules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property

from .._compat import Self
from .._validation import positive_float, positive_int
from ..filter import FilterDecision
from ..filter.types import JsonValue
from ..types import Lang, Preset, Role, Sample
from . import _text as text
from .rules import QualityLabel

Scorer = Callable[[str, str], float]


@dataclass
class TranslationQualityProfile:
    source_lang: Lang
    target_lang: Lang
    review_min_ratio: float = 0.2
    review_max_ratio: float = 6.0
    reject_min_ratio: float = 0.05
    reject_max_ratio: float = 20.0
    min_identical_script_chars: int = 4

    def __post_init__(self) -> None:
        self.source_lang = _lang("source_lang", self.source_lang)
        self.target_lang = _lang("target_lang", self.target_lang)
        self.min_identical_script_chars = positive_int(
            "min_identical_script_chars",
            self.min_identical_script_chars,
        )
        self.reject_min_ratio = positive_float(
            "reject_min_ratio",
            self.reject_min_ratio,
        )
        self.review_min_ratio = positive_float(
            "review_min_ratio",
            self.review_min_ratio,
        )
        self.review_max_ratio = positive_float(
            "review_max_ratio",
            self.review_max_ratio,
        )
        self.reject_max_ratio = positive_float(
            "reject_max_ratio",
            self.reject_max_ratio,
        )
        if not (
            self.reject_min_ratio
            <= self.review_min_ratio
            <= self.review_max_ratio
            <= self.reject_max_ratio
        ):
            raise ValueError(
                "length ratios must satisfy reject_min_ratio <= review_min_ratio "
                "<= review_max_ratio <= reject_max_ratio."
            )


@dataclass
class _Metrics:
    source: text.Metrics
    target: text.Metrics

    @classmethod
    def from_sample(cls, sample: Sample, profile: TranslationQualityProfile) -> Self:
        return cls(
            source=text.metrics(sample, Role.SOURCE, profile.source_lang),
            target=text.metrics(sample, Role.TARGET, profile.target_lang),
        )

    @cached_property
    def char_ratio(self) -> float:
        return text.length_ratio(self.source.chars, self.target.chars)

    @cached_property
    def number_value_overlap(self) -> float:
        return text.overlap(self.source.number_values, self.target.number_values)

    @cached_property
    def number_surface_overlap(self) -> float:
        return text.overlap(self.source.numbers, self.target.numbers)

    @cached_property
    def complex_numbers(self) -> bool:
        return self.source.complex_numbers or self.target.complex_numbers


@dataclass(frozen=True)
class _Decision:
    label: QualityLabel
    matched: bool


_Rule = Callable[[_Metrics, TranslationQualityProfile], _Decision]


@dataclass(frozen=True)
class Bicleaner:
    scorer: Scorer
    source_lang: Lang
    target_lang: Lang
    min_score: float = 0.6

    def __post_init__(self) -> None:
        if not callable(self.scorer):
            raise TypeError("bicleaner scorer must be callable.")
        object.__setattr__(
            self,
            "min_score",
            text.unit_ratio("bicleaner min_score", self.min_score),
        )
        object.__setattr__(self, "source_lang", _lang("source_lang", self.source_lang))
        object.__setattr__(self, "target_lang", _lang("target_lang", self.target_lang))

    @classmethod
    def from_preset(
        cls,
        preset: Preset,
        *,
        source_lang: Lang,
        target_lang: Lang,
        scorer: Scorer,
        min_score: float = 0.6,
    ) -> Self:
        if preset != Preset.WMT19:
            raise ValueError("bicleaner quality profile is only defined for WMT19.")

        source = _lang("source_lang", source_lang)
        target = _lang("target_lang", target_lang)
        if source != Lang.ZH or target != Lang.EN:
            raise ValueError("WMT19 bicleaner quality profile is only defined for zh-en.")
        return cls(
            scorer,
            min_score=min_score,
            source_lang=source,
            target_lang=target,
        )

    def __call__(self, sample: Sample) -> FilterDecision:
        metrics = _Metrics.from_sample(
            sample,
            TranslationQualityProfile(
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            ),
        )
        score = text.unit_ratio(
            "bicleaner scorer output",
            self.scorer(metrics.source.text, metrics.target.text),
        )
        label = QualityLabel.ACCEPT if score >= self.min_score else QualityLabel.REJECT
        flag = (
            "bicleaner_accept"
            if label == QualityLabel.ACCEPT
            else "bicleaner_reject"
        )
        return FilterDecision(
            label=label,
            metrics={
                "decision": label.value,
                "source": metrics.source.text,
                "target": metrics.target.text,
                "source_lang": metrics.source.lang.value,
                "target_lang": metrics.target.lang.value,
                "bicleaner_score": round(score, 6),
                "flags": [flag],
            },
        )


@dataclass(frozen=True)
class TranslationQuality:
    profile: TranslationQualityProfile

    @classmethod
    def from_preset(
        cls,
        preset: Preset,
        *,
        source_lang: Lang,
        target_lang: Lang,
    ) -> Self:
        if preset != Preset.WMT19:
            raise ValueError("translation quality profile is only defined for WMT19.")

        source = _lang("source_lang", source_lang)
        target = _lang("target_lang", target_lang)
        if source != Lang.ZH or target != Lang.EN:
            raise ValueError("WMT19 translation quality profile is only defined for zh-en.")
        return cls(TranslationQualityProfile(source_lang=source, target_lang=target))

    def __call__(self, sample: Sample) -> FilterDecision:
        metrics = _Metrics.from_sample(sample, self.profile)
        flags: list[str] = []
        label = QualityLabel.ACCEPT
        for flag, rule in _RULES:
            decision = rule(metrics, self.profile)
            if not decision.matched:
                continue
            flags.append(flag)
            if decision.label == QualityLabel.REJECT:
                label = QualityLabel.REJECT

        return FilterDecision(
            label=label,
            metrics=_log(metrics, label, flags),
        )


def _invalid_pair_text(
    metrics: _Metrics,
    profile: TranslationQualityProfile,
) -> _Decision:
    return _reject(not metrics.source.valid or not metrics.target.valid)


def _target_extremely_short(
    metrics: _Metrics,
    profile: TranslationQualityProfile,
) -> _Decision:
    return _reject(metrics.char_ratio < profile.reject_min_ratio)


def _target_short(metrics: _Metrics, profile: TranslationQualityProfile) -> _Decision:
    return _reject(
        profile.reject_min_ratio <= metrics.char_ratio < profile.review_min_ratio,
    )


def _target_extremely_long(
    metrics: _Metrics,
    profile: TranslationQualityProfile,
) -> _Decision:
    return _reject(metrics.char_ratio > profile.reject_max_ratio)


def _target_long(metrics: _Metrics, profile: TranslationQualityProfile) -> _Decision:
    return _reject(
        profile.review_max_ratio < metrics.char_ratio <= profile.reject_max_ratio,
    )


def _identical_text(
    metrics: _Metrics,
    profile: TranslationQualityProfile,
) -> _Decision:
    return _reject(
        metrics.source.lang != metrics.target.lang
        and metrics.source.normalized.casefold() == metrics.target.normalized.casefold()
        and (
            metrics.source.script_chars >= profile.min_identical_script_chars
            or metrics.target.script_chars >= profile.min_identical_script_chars
        ),
    )


def _placeholder_mismatch(
    metrics: _Metrics,
    profile: TranslationQualityProfile,
) -> _Decision:
    return _reject(
        text.counts(metrics.source.placeholders) != text.counts(metrics.target.placeholders),
    )


def _html_tag_mismatch(
    metrics: _Metrics,
    profile: TranslationQualityProfile,
) -> _Decision:
    return _reject(
        text.counts(metrics.source.html_tags) != text.counts(metrics.target.html_tags),
    )


def _complex_numbers(
    metrics: _Metrics,
    profile: TranslationQualityProfile,
) -> _Decision:
    return _reject(metrics.complex_numbers)


def _number_value_mismatch(
    metrics: _Metrics,
    profile: TranslationQualityProfile,
) -> _Decision:
    has_numbers = len(metrics.source.numbers) > 0 or len(metrics.target.numbers) > 0
    return _reject(
        has_numbers
        and not metrics.complex_numbers
        and metrics.number_value_overlap < 1.0,
    )


def _number_surface_mismatch(
    metrics: _Metrics,
    profile: TranslationQualityProfile,
) -> _Decision:
    has_numbers = len(metrics.source.numbers) > 0 or len(metrics.target.numbers) > 0
    return _accept(
        has_numbers
        and metrics.number_value_overlap == 1.0
        and metrics.number_surface_overlap < 1.0,
    )


def _accept(matched: bool) -> _Decision:
    return _Decision(QualityLabel.ACCEPT, matched)


def _reject(matched: bool) -> _Decision:
    return _Decision(QualityLabel.REJECT, matched)


_RULES: tuple[tuple[str, _Rule], ...] = (
    ("invalid_pair_text", _invalid_pair_text),
    ("target_extremely_short", _target_extremely_short),
    ("target_short", _target_short),
    ("target_extremely_long", _target_extremely_long),
    ("target_long", _target_long),
    ("identical_text", _identical_text),
    ("placeholder_mismatch", _placeholder_mismatch),
    ("html_tag_mismatch", _html_tag_mismatch),
    ("complex_numbers", _complex_numbers),
    ("number_value_mismatch", _number_value_mismatch),
    ("number_surface_mismatch", _number_surface_mismatch),
)


def _log(
    metrics: _Metrics,
    label: QualityLabel,
    flags: list[str],
) -> Mapping[str, JsonValue]:
    return {
        "source": metrics.source.text,
        "target": metrics.target.text,
        "decision": label.value,
        "source_lang": metrics.source.lang.value,
        "target_lang": metrics.target.lang.value,
        "flags": flags,
    }


def _lang(name: str, value: Lang) -> Lang:
    if not isinstance(value, Lang):
        raise TypeError(f"{name} must be a Lang value.")
    if value == Lang.UND:
        raise ValueError(f"{name} must be explicit.")
    return value


__all__ = [
    "Bicleaner",
    "QualityLabel",
    "Scorer",
    "TranslationQuality",
    "TranslationQualityProfile",
]
