"""Atomic quality predicates for canonical machine-translation samples.

This module only owns source/target pair checks and optional pair-level model
rules. Single-text checks are provided by anydataset.quality.text, and cross-rule
label transitions are provided by anydataset.quality.rules.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import cached_property

from .._compat import Self
from .._immutable import Immutable
from .._validation import positive_float, positive_int
from ..filter import FilterDecision
from ..filter.types import JsonValue
from ..types import Lang, Preset, Role, Sample
from . import _text as text
from .rules import QualityLabel

Scorer = Callable[[str, str], float]

_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
)
_SCALED_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<number>[+-]?(?:\d{1,3}(?:[,\s]\d{3})+|\d+)(?:\.\d+)?)"
    r"[\s-]*(?P<scale>万多亿|万亿|亿|万|thousand|million|billion|trillion)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_ZH_CENTURY_DECADE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<century>\d{1,2})\s*世纪\s*"
    r"(?P<decade>\d{1,2})\s*年代"
)
_ZH_MONTH_RE = re.compile(r"(?<!\d)(?P<month>1[0-2]|0?[1-9])\s*月份?")
_ZH_YEAR_RE = re.compile(r"(?<!\d)\d{1,4}\s*年(?!代)")
_EN_MONTH_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b"
)
_EN_DECADE_RE = re.compile(r"(?<!\d)(?P<year>\d{3,4})s\b", re.IGNORECASE)
_ZH_UNSUPPORTED_COMPLEX_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:世纪|年代|年份?|月份?|日|万多亿|万亿|亿|万|兆|千|百)"
)
_EN_UNSUPPORTED_COMPLEX_RE = re.compile(
    r"(?:\b(?:early|mid|late)\b|\b(?:century|centuries|decade|decades)\b|"
    r"\d{3,4}s\b)",
    re.IGNORECASE,
)
_UNICODE_FRACTION_RE = re.compile(r"\d+\N{FRACTION SLASH}\d+")
_ASCII_RANGE_RE = re.compile(
    r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:-(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)+$"
)
_SCALE_MULTIPLIERS = {
    "万": Decimal(10_000),
    "亿": Decimal(100_000_000),
    "万亿": Decimal(1_000_000_000_000),
    "万多亿": Decimal(1_000_000_000_000),
    "thousand": Decimal(1_000),
    "million": Decimal(1_000_000),
    "billion": Decimal(1_000_000_000),
    "trillion": Decimal(1_000_000_000_000),
}
_EN_MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


@dataclass(init=False, unsafe_hash=True)
class TranslationQualityProfile(Immutable):
    source_lang: Lang
    target_lang: Lang
    review_min_ratio: float
    review_max_ratio: float
    reject_min_ratio: float
    reject_max_ratio: float
    min_identical_script_chars: int

    def __init__(
        self,
        source_lang: Lang,
        target_lang: Lang,
        review_min_ratio: float = 0.2,
        review_max_ratio: float = 6.0,
        reject_min_ratio: float = 0.05,
        reject_max_ratio: float = 20.0,
        min_identical_script_chars: int = 4,
    ) -> None:
        normalized_source_lang = _lang("source_lang", source_lang)
        normalized_target_lang = _lang("target_lang", target_lang)
        normalized_min_identical_script_chars = positive_int(
            "min_identical_script_chars",
            min_identical_script_chars,
        )
        normalized_reject_min_ratio = positive_float(
            "reject_min_ratio",
            reject_min_ratio,
        )
        normalized_review_min_ratio = positive_float(
            "review_min_ratio",
            review_min_ratio,
        )
        normalized_review_max_ratio = positive_float(
            "review_max_ratio",
            review_max_ratio,
        )
        normalized_reject_max_ratio = positive_float(
            "reject_max_ratio",
            reject_max_ratio,
        )
        if not (
            normalized_reject_min_ratio
            <= normalized_review_min_ratio
            <= normalized_review_max_ratio
            <= normalized_reject_max_ratio
        ):
            raise ValueError(
                "length ratios must satisfy reject_min_ratio <= review_min_ratio "
                "<= review_max_ratio <= reject_max_ratio."
            )

        self.source_lang = normalized_source_lang
        self.target_lang = normalized_target_lang
        self.review_min_ratio = normalized_review_min_ratio
        self.review_max_ratio = normalized_review_max_ratio
        self.reject_min_ratio = normalized_reject_min_ratio
        self.reject_max_ratio = normalized_reject_max_ratio
        self.min_identical_script_chars = normalized_min_identical_script_chars
        self.seal()


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
    def numbers(self) -> _NumberPair:
        if {
            self.source.expected_lang,
            self.target.expected_lang,
        } == {Lang.ZH, Lang.EN}:
            return _zh_en_numbers(self.source, self.target)
        return _NumberPair(
            source=text.counts(self.source.number_values),
            target=text.counts(self.target.number_values),
            unsupported_complex=(
                self.source.complex_numbers or self.target.complex_numbers
            ),
        )

    @cached_property
    def number_surface_overlap(self) -> float:
        return text.overlap(self.source.numbers, self.target.numbers)

    @cached_property
    def has_numbers(self) -> bool:
        return bool(
            self.numbers.source
            or self.numbers.target
            or self.source.numbers
            or self.target.numbers
        )

    @cached_property
    def placeholders(self) -> tuple[Counter[str], Counter[str]]:
        source = self.source.placeholders
        target = self.target.placeholders
        if {
            self.source.expected_lang,
            self.target.expected_lang,
        } == {Lang.ZH, Lang.EN}:
            source = tuple(value for value in source if not _dollar_number(value))
            target = tuple(value for value in target if not _dollar_number(value))
        return text.counts(source), text.counts(target)


@dataclass(frozen=True)
class _NumberPair:
    source: Counter[str]
    target: Counter[str]
    unsupported_complex: bool


@dataclass(frozen=True)
class _Decision:
    label: QualityLabel
    matched: bool


_Rule = Callable[[_Metrics, TranslationQualityProfile], _Decision]


@dataclass(init=False, unsafe_hash=True)
class Bicleaner(Immutable):
    scorer: Scorer
    source_lang: Lang
    target_lang: Lang
    min_score: float

    def __init__(
        self,
        scorer: Scorer,
        source_lang: Lang,
        target_lang: Lang,
        min_score: float = 0.6,
    ) -> None:
        if not callable(scorer):
            raise TypeError("bicleaner scorer must be callable.")
        self.scorer = scorer
        self.source_lang = _lang("source_lang", source_lang)
        self.target_lang = _lang("target_lang", target_lang)
        self.min_score = text.unit_ratio("bicleaner min_score", min_score)
        self.seal()

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
        if {source, target} != {Lang.ZH, Lang.EN}:
            raise ValueError(
                "WMT19 translation quality profile is only defined for the zh-en pair."
            )
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
    return _reject(metrics.placeholders[0] != metrics.placeholders[1])


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
    return _reject(metrics.numbers.unsupported_complex)


def _number_value_mismatch(
    metrics: _Metrics,
    profile: TranslationQualityProfile,
) -> _Decision:
    return _reject(
        metrics.has_numbers
        and not metrics.numbers.unsupported_complex
        and metrics.numbers.source != metrics.numbers.target,
    )


def _number_surface_mismatch(
    metrics: _Metrics,
    profile: TranslationQualityProfile,
) -> _Decision:
    return _accept(
        metrics.has_numbers
        and not metrics.numbers.unsupported_complex
        and metrics.numbers.source == metrics.numbers.target
        and metrics.number_surface_overlap < 1.0,
    )


def _zh_en_numbers(source: text.Metrics, target: text.Metrics) -> _NumberPair:
    if source.expected_lang == Lang.ZH:
        chinese_metrics, english_metrics = source, target
        source_is_chinese = True
    else:
        chinese_metrics, english_metrics = target, source
        source_is_chinese = False

    chinese = _numeric_text(chinese_metrics.normalized)
    english = _numeric_text(english_metrics.normalized)
    months = _zh_months(chinese) & _en_months(english)
    chinese_values, decades, chinese_unsupported = _zh_number_values(
        chinese,
        months,
    )
    english_values, english_unsupported = _en_number_values(
        english,
        months,
        decades,
    )
    unsupported = (
        chinese_unsupported
        or english_unsupported
        or _unsupported_number_tokens(chinese_metrics)
        or _unsupported_number_tokens(english_metrics)
    )
    if source_is_chinese:
        return _NumberPair(chinese_values, english_values, unsupported)
    return _NumberPair(english_values, chinese_values, unsupported)


def _zh_number_values(
    value: str,
    months: set[int],
) -> tuple[Counter[str], set[int], bool]:
    values: list[str] = []
    value_spans: list[tuple[int, int]] = []
    supported_spans: list[tuple[int, int]] = []
    decades: set[int] = set()

    if "世纪" in value and "年代" in value:
        for match in _ZH_CENTURY_DECADE_RE.finditer(value):
            century = int(match.group("century"))
            decade = int(match.group("decade"))
            if century <= 0 or decade >= 100:
                continue
            year = (century - 1) * 100 + decade
            values.append(str(year))
            decades.add(year)
            span = (match.start(), match.end())
            value_spans.append(span)
            supported_spans.append(span)

    for match in _SCALED_NUMBER_RE.finditer(value):
        number = _SPACE_RE.sub("", match.group("number")).replace(",", "")
        try:
            scaled = Decimal(number) * _SCALE_MULTIPLIERS[
                match.group("scale").lower()
            ]
        except InvalidOperation:
            continue
        values.append(_decimal(scaled))
        span = (match.start(), match.end())
        value_spans.append(span)
        supported_spans.append(span)

    if months:
        for match in _ZH_MONTH_RE.finditer(value):
            month = int(match.group("month"))
            if month not in months:
                continue
            values.append(str(month))
            span = (match.start(), match.end())
            value_spans.append(span)
            supported_spans.append(span)

    supported_spans.extend(
        (match.start(), match.end()) for match in _ZH_YEAR_RE.finditer(value)
    )
    remaining = _masked(value, value_spans)
    values.extend(_number(match.group(0)) for match in _NUMBER_RE.finditer(remaining))
    unsupported = _ZH_UNSUPPORTED_COMPLEX_RE.search(
        _masked(value, supported_spans)
    ) is not None
    return Counter(values), decades, unsupported


def _en_number_values(
    value: str,
    months: set[int],
    decades: set[int],
) -> tuple[Counter[str], bool]:
    values: list[str] = []
    value_spans: list[tuple[int, int]] = []
    supported_spans: list[tuple[int, int]] = []

    for match in _SCALED_NUMBER_RE.finditer(value):
        number = _SPACE_RE.sub("", match.group("number")).replace(",", "")
        try:
            scaled = Decimal(number) * _SCALE_MULTIPLIERS[
                match.group("scale").lower()
            ]
        except InvalidOperation:
            continue
        values.append(_decimal(scaled))
        span = (match.start(), match.end())
        value_spans.append(span)
        supported_spans.append(span)

    if months:
        for match in _EN_MONTH_RE.finditer(value):
            month = _en_month(value, match)
            if month is not None and month in months:
                values.append(str(month))

    if decades:
        for match in _EN_DECADE_RE.finditer(value):
            year = int(match.group("year"))
            if year not in decades:
                continue
            values.append(str(year))
            span = (match.start(), match.end())
            value_spans.append(span)
            supported_spans.append(span)

    remaining = _masked(value, value_spans)
    values.extend(_number(match.group(0)) for match in _NUMBER_RE.finditer(remaining))
    unsupported = _EN_UNSUPPORTED_COMPLEX_RE.search(
        _masked(value, supported_spans)
    ) is not None
    return Counter(values), unsupported


def _numeric_text(value: str) -> str:
    if "\N{FULLWIDTH HYPHEN-MINUS}" in value:
        value = value.replace("\N{FULLWIDTH HYPHEN-MINUS}", "\N{EN DASH}")
    normalized = value if value.isascii() else unicodedata.normalize("NFKC", value)
    if "\N{FRACTION SLASH}" in normalized:
        return _UNICODE_FRACTION_RE.sub(" ", normalized)
    return normalized


def _zh_months(value: str) -> set[int]:
    return {int(match.group("month")) for match in _ZH_MONTH_RE.finditer(value)}


def _en_months(value: str) -> set[int]:
    return {
        month
        for match in _EN_MONTH_RE.finditer(value)
        if (month := _en_month(value, match)) is not None
    }


def _en_month(value: str, match: re.Match[str]) -> int | None:
    name = match.group(0)
    if name != "May":
        return _EN_MONTHS[name]
    prefix = value[: match.start()].rstrip()
    suffix = value[match.end() :].lstrip(" ,")
    sentence_initial = not prefix or prefix[-1] in ".!?"
    if sentence_initial and (not suffix or not suffix[0].isdecimal()):
        return None
    return _EN_MONTHS[name]


def _number(value: str) -> str:
    value = value.replace(",", "")
    try:
        return _decimal(Decimal(value))
    except InvalidOperation:
        return value


def _decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _masked(value: str, spans: list[tuple[int, int]]) -> str:
    masked = list(value)
    for start, end in spans:
        masked[start:end] = " " * (end - start)
    return "".join(masked)


def _unsupported_number_tokens(metrics: text.Metrics) -> bool:
    return any(
        text.normalized_number(token) is None
        and _ASCII_RANGE_RE.fullmatch(token) is None
        for token in metrics.numbers
    )


def _dollar_number(value: str) -> bool:
    return value.startswith("$") and value[1:].isdecimal()


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
