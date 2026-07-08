"""Rule-based quality labels for canonical machine-translation samples.

The predicate reads source and target text from canonical `Sample` objects and
returns filter labels plus lightweight diagnostics. It does not load datasets,
build filter caches, or run neural quality-estimation models.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import auto
from functools import cached_property
from math import isfinite
from .._compat import Self, StrEnum
from ..filter import FilterDecision
from ..filter.types import JsonValue
from ..types import Modality, Preset, Role, Sample, TextItem, TextMeta, TextView

Scorer = Callable[[str, str], float]

_NUMBER_RE = re.compile(r"[+-]?\d+(?:[.,:/-]\d+)*")
_PLACEHOLDER_RE = re.compile(
    r"(\{[^{}]+\}|%\([^)]+\)[#0+\-]?\d*(?:\.\d+)?[diouxXeEfFgGcrs]"
    r"(?![A-Za-z])|%[#0+\-]?\d*(?:\.\d+)?[diouxXeEfFgGcrs]"
    r"(?![A-Za-z])|\$\d+)"
)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]{0,80}>")
_SPACE_RE = re.compile(r"\s+")
_COMPLEX_NUMBER_RE = re.compile(
    r"(\d+(?:[.,]\d+)?\s*(?:世纪|年代|年|月|日|万|亿|千|百)"
    r"|(?:early|mid|late)\b|(?:centur|decad)(?:y|ies|e|es)\b|\d{3,4}s\b)",
    re.IGNORECASE,
)
_GROUPED_NUMBER_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_DECIMAL_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

_LATIN_LANGS = frozenset(
    {
        "af",
        "ca",
        "cs",
        "da",
        "de",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "hr",
        "hu",
        "id",
        "it",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "sv",
        "tr",
        "vi",
    }
)
_CJK_LANGS = frozenset({"zh", "ja"})
_CYRILLIC_LANGS = frozenset({"be", "bg", "kk", "mk", "ru", "sr", "uk"})
_ARABIC_LANGS = frozenset({"ar", "fa", "ps", "ur"})


class Label(StrEnum):
    REJECT = auto()
    REVIEW = auto()
    USABLE = auto()
    CLEAN = auto()

    @property
    def rank(self) -> int:
        if self is Label.REJECT:
            return 0
        if self is Label.REVIEW:
            return 1
        if self is Label.USABLE:
            return 2
        if self is Label.CLEAN:
            return 3
        raise ValueError(f"Unsupported label: {self!r}.")

    def min(self, other: Self) -> Self:
        return self if self.rank <= other.rank else other


@dataclass
class Profile:
    source_lang: str
    target_lang: str
    min_chars: int = 1
    review_min_ratio: float = 0.2
    review_max_ratio: float = 6.0
    reject_min_ratio: float = 0.05
    reject_max_ratio: float = 20.0
    min_script_ratio: float = 0.45
    reject_script_ratio: float = 0.2
    min_script_chars: int = 4
    max_control_ratio: float = 0.02
    max_repeated_run: int = 24

    def __post_init__(self) -> None:
        self.source_lang = _lang_code(self.source_lang)
        self.target_lang = _lang_code(self.target_lang)


@dataclass
class _Pair:
    source: str
    target: str
    source_lang: str
    target_lang: str
    source_valid: bool
    target_valid: bool


@dataclass
class _Metrics(_Pair):
    scores: dict[str, float] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_pair(cls, pair: _Pair) -> Self:
        return cls(
            source=pair.source,
            target=pair.target,
            source_lang=pair.source_lang,
            target_lang=pair.target_lang,
            source_valid=pair.source_valid,
            target_valid=pair.target_valid,
        )

    @cached_property
    def normalized_source(self) -> str:
        return _normalize_space(self.source)

    @cached_property
    def normalized_target(self) -> str:
        return _normalize_space(self.target)

    @cached_property
    def source_chars(self) -> int:
        return _nonspace_len(self.normalized_source)

    @cached_property
    def target_chars(self) -> int:
        return _nonspace_len(self.normalized_target)

    @cached_property
    def char_ratio(self) -> float:
        return _length_ratio(self.source_chars, self.target_chars)

    @cached_property
    def source_control_ratio(self) -> float:
        return _control_ratio(self.normalized_source)

    @cached_property
    def target_control_ratio(self) -> float:
        return _control_ratio(self.normalized_target)

    @cached_property
    def source_repeated_run(self) -> int:
        return _longest_run(self.normalized_source)

    @cached_property
    def target_repeated_run(self) -> int:
        return _longest_run(self.normalized_target)

    @cached_property
    def source_script_ratio(self) -> float | None:
        return _script_ratio(self.normalized_source, self.source_lang)[0]

    @cached_property
    def target_script_ratio(self) -> float | None:
        return _script_ratio(self.normalized_target, self.target_lang)[0]

    @cached_property
    def source_script_chars(self) -> int:
        return _script_ratio(self.normalized_source, self.source_lang)[1]

    @cached_property
    def target_script_chars(self) -> int:
        return _script_ratio(self.normalized_target, self.target_lang)[1]

    @cached_property
    def source_numbers(self) -> tuple[str, ...]:
        return _numbers(self.normalized_source)

    @cached_property
    def target_numbers(self) -> tuple[str, ...]:
        return _numbers(self.normalized_target)

    @cached_property
    def source_number_values(self) -> tuple[str, ...]:
        return _number_values(self.source_numbers)

    @cached_property
    def target_number_values(self) -> tuple[str, ...]:
        return _number_values(self.target_numbers)

    @cached_property
    def number_value_overlap(self) -> float:
        return _overlap(self.source_number_values, self.target_number_values)

    @cached_property
    def number_surface_overlap(self) -> float:
        return _overlap(self.source_numbers, self.target_numbers)

    @cached_property
    def source_placeholders(self) -> tuple[str, ...]:
        return _placeholders(self.normalized_source)

    @cached_property
    def target_placeholders(self) -> tuple[str, ...]:
        return _placeholders(self.normalized_target)

    @cached_property
    def source_html_tags(self) -> tuple[str, ...]:
        return _html_tags(self.normalized_source)

    @cached_property
    def target_html_tags(self) -> tuple[str, ...]:
        return _html_tags(self.normalized_target)

    @cached_property
    def complex_numbers(self) -> bool:
        if _COMPLEX_NUMBER_RE.search(self.normalized_source) is not None:
            return True
        if _COMPLEX_NUMBER_RE.search(self.normalized_target) is not None:
            return True
        return any(
            _normalized_number(token) is None
            for token in self.source_numbers + self.target_numbers
        )


_RuleDecision = tuple[Label, bool]
_Rule = Callable[[_Metrics, Profile], _RuleDecision]
_Callback = Callable[[_Metrics, Profile, Label, tuple[str, ...]], Label]


@dataclass(frozen=True)
class Bicleaner:
    scorer: Scorer
    usable_score: float = 0.6
    high_score: float = 0.7

    def __post_init__(self) -> None:
        if not isfinite(self.usable_score) or not isfinite(self.high_score):
            raise ValueError("bicleaner thresholds must be finite.")
        if self.usable_score > self.high_score:
            raise ValueError("bicleaner usable_score must be <= high_score.")

    def __call__(
        self,
        metrics: _Metrics,
        profile: Profile,
        label: Label,
        flags: tuple[str, ...],
    ) -> Label:
        score = self.scorer(metrics.source, metrics.target)
        if not isfinite(score):
            raise ValueError("bicleaner scorer must return a finite score.")
        metrics.scores["bicleaner_score"] = score
        if score < self.usable_score:
            return Label.REJECT
        if label == Label.REJECT:
            return Label.REVIEW
        if score < self.high_score:
            return label.min(Label.USABLE)
        return label

    def flag(self, metrics: _Metrics) -> str:
        score = metrics.scores["bicleaner_score"]
        if score < self.usable_score:
            return "bicleaner_reject"
        if score < self.high_score:
            return "bicleaner_usable"
        return "bicleaner_high"


@dataclass(frozen=True)
class Predicate:
    profile: Profile
    callbacks: tuple[_Callback, ...] = ()

    @classmethod
    def from_preset(
        cls,
        preset: Preset,
        *,
        source_lang: str,
        target_lang: str,
        bicleaner: Scorer | None = None,
    ) -> Self:
        if preset != Preset.WMT19:
            raise ValueError("translation quality profile is only defined for WMT19.")

        source = _lang_code(source_lang)
        target = _lang_code(target_lang)
        if source != "zh" or target != "en":
            raise ValueError("WMT19 translation quality profile is only defined for zh-en.")
        callbacks = () if bicleaner is None else (Bicleaner(bicleaner),)
        return cls(Profile(source_lang=source, target_lang=target), callbacks=callbacks)

    def __call__(self, sample: Sample) -> FilterDecision:
        metrics = _Metrics.from_pair(_pair(sample, self.profile))
        label = Label.CLEAN
        flags: list[str] = []
        for flag, rule in _RULES:
            ceiling, matched = rule(metrics, self.profile)
            if matched:
                label = label.min(ceiling)
                flags.append(flag)

        for callback in self.callbacks:
            previous = label
            label = callback(metrics, self.profile, label, tuple(flags))
            if isinstance(callback, Bicleaner):
                flags.append(callback.flag(metrics))
            if label != previous:
                flags.append(f"{_callback_key(callback)}_{previous.value}_to_{label.value}")

        return FilterDecision(
            label=label,
            metrics=_log(metrics, label, flags),
        )


def _invalid_text(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return Label.REJECT, not metrics.source_valid or not metrics.target_valid


def _empty_text(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return (
        Label.REJECT,
        metrics.source_chars < profile.min_chars
        or metrics.target_chars < profile.min_chars,
    )


def _target_extremely_short(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return Label.REJECT, metrics.char_ratio < profile.reject_min_ratio


def _target_short(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return (
        Label.REVIEW,
        profile.reject_min_ratio <= metrics.char_ratio < profile.review_min_ratio,
    )


def _target_extremely_long(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return Label.REJECT, metrics.char_ratio > profile.reject_max_ratio


def _target_long(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return (
        Label.REVIEW,
        profile.review_max_ratio < metrics.char_ratio <= profile.reject_max_ratio,
    )


def _control_chars(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return (
        Label.REJECT,
        metrics.source_control_ratio > profile.max_control_ratio
        or metrics.target_control_ratio > profile.max_control_ratio,
    )


def _long_repeated_run(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return (
        Label.REJECT,
        metrics.source_repeated_run >= profile.max_repeated_run
        or metrics.target_repeated_run >= profile.max_repeated_run,
    )


def _identical_text(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return (
        Label.REJECT,
        metrics.source_lang != metrics.target_lang
        and metrics.normalized_source.casefold() == metrics.normalized_target.casefold()
        and (
            metrics.source_script_chars >= profile.min_script_chars
            or metrics.target_script_chars >= profile.min_script_chars
        ),
    )


def _source_script_mismatch(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    ratio = metrics.source_script_ratio
    if ratio is None or metrics.source_script_chars < profile.min_script_chars:
        return Label.REJECT, False
    return (
        Label.REJECT,
        ratio < profile.reject_script_ratio,
    )


def _source_script_low(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    ratio = metrics.source_script_ratio
    if ratio is None or metrics.source_script_chars < profile.min_script_chars:
        return Label.REVIEW, False
    return (
        Label.REVIEW,
        profile.reject_script_ratio <= ratio < profile.min_script_ratio,
    )


def _target_script_mismatch(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    ratio = metrics.target_script_ratio
    if ratio is None or metrics.target_script_chars < profile.min_script_chars:
        return Label.REJECT, False
    return (
        Label.REJECT,
        ratio < profile.reject_script_ratio,
    )


def _target_script_low(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    ratio = metrics.target_script_ratio
    if ratio is None or metrics.target_script_chars < profile.min_script_chars:
        return Label.REVIEW, False
    return (
        Label.REVIEW,
        profile.reject_script_ratio <= ratio < profile.min_script_ratio,
    )


def _placeholder_mismatch(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return (
        Label.REJECT,
        _counts(metrics.source_placeholders) != _counts(metrics.target_placeholders),
    )


def _html_tag_mismatch(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return (
        Label.REVIEW,
        _counts(metrics.source_html_tags) != _counts(metrics.target_html_tags),
    )


def _complex_numbers(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    return Label.REVIEW, metrics.complex_numbers


def _number_value_mismatch(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    has_numbers = len(metrics.source_numbers) > 0 or len(metrics.target_numbers) > 0
    return (
        Label.REJECT,
        has_numbers
        and not metrics.complex_numbers
        and metrics.number_value_overlap < 1.0,
    )


def _number_surface_mismatch(metrics: _Metrics, profile: Profile) -> _RuleDecision:
    has_numbers = len(metrics.source_numbers) > 0 or len(metrics.target_numbers) > 0
    return (
        Label.USABLE,
        has_numbers
        and metrics.number_value_overlap == 1.0
        and metrics.number_surface_overlap < 1.0,
    )


_RULES: tuple[tuple[str, _Rule], ...] = (
    ("invalid_text", _invalid_text),
    ("empty_text", _empty_text),
    ("target_extremely_short", _target_extremely_short),
    ("target_short", _target_short),
    ("target_extremely_long", _target_extremely_long),
    ("target_long", _target_long),
    ("control_chars", _control_chars),
    ("long_repeated_run", _long_repeated_run),
    ("identical_text", _identical_text),
    ("source_script_mismatch", _source_script_mismatch),
    ("source_script_low", _source_script_low),
    ("target_script_mismatch", _target_script_mismatch),
    ("target_script_low", _target_script_low),
    ("placeholder_mismatch", _placeholder_mismatch),
    ("html_tag_mismatch", _html_tag_mismatch),
    ("complex_numbers", _complex_numbers),
    ("number_value_mismatch", _number_value_mismatch),
    ("number_surface_mismatch", _number_surface_mismatch),
)


def _pair(sample: Sample, profile: Profile) -> _Pair:
    source, source_lang, source_valid = _text(sample, Role.SOURCE, profile.source_lang)
    target, target_lang, target_valid = _text(sample, Role.TARGET, profile.target_lang)
    return _Pair(
        source=source,
        target=target,
        source_lang=source_lang,
        target_lang=target_lang,
        source_valid=source_valid,
        target_valid=target_valid,
    )


def _text(sample: Sample, role: Role, fallback_lang: str) -> tuple[str, str, bool]:
    item = sample.get((role, Modality.TEXT))
    if not isinstance(item, TextItem):
        return "", fallback_lang, False

    text = item.views.get(TextView.TEXT)
    if not isinstance(text, str):
        return "", _item_lang(item, fallback_lang), False
    return text, _item_lang(item, fallback_lang), True


def _item_lang(item: TextItem, fallback: str) -> str:
    value = item.meta.get(TextMeta.LANG)
    if value is None:
        return fallback
    return _lang_code(str(value))


def _log(
    metrics: _Metrics,
    label: Label,
    flags: list[str],
) -> Mapping[str, JsonValue]:
    output: dict[str, JsonValue] = {
        "source": metrics.source,
        "target": metrics.target,
        "decision": label.value,
        "source_lang": metrics.source_lang,
        "target_lang": metrics.target_lang,
        "flags": flags,
    }
    if "bicleaner_score" in metrics.scores:
        output["bicleaner_score"] = round(metrics.scores["bicleaner_score"], 6)
    return output


def _normalize_space(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _nonspace_len(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def _length_ratio(source_len: int, target_len: int) -> float:
    if source_len == 0:
        return 0.0
    return target_len / source_len


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _NUMBER_RE.finditer(text))


def _number_values(tokens: Iterable[str]) -> tuple[str, ...]:
    return tuple(_number_value(token) for token in tokens)


def _number_value(token: str) -> str:
    normalized = _normalized_number(token)
    if normalized is None:
        return token
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return token
    return format(value.normalize(), "f")


def _normalized_number(token: str) -> str | None:
    if _GROUPED_NUMBER_RE.fullmatch(token) is not None:
        return token.replace(",", "")
    if _DECIMAL_RE.fullmatch(token) is not None:
        return token
    return None


def _placeholders(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _PLACEHOLDER_RE.finditer(text))


def _html_tags(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _HTML_TAG_RE.finditer(text))


def _overlap(source: Iterable[str], target: Iterable[str]) -> float:
    source_counts = Counter(source)
    target_counts = Counter(target)
    if len(source_counts) == 0 and len(target_counts) == 0:
        return 1.0

    union = source_counts | target_counts
    if len(union) == 0:
        return 1.0
    intersection = source_counts & target_counts
    return sum(intersection.values()) / sum(union.values())


def _script_ratio(text: str, lang: str) -> tuple[float | None, int]:
    script = _expected_script(lang)
    if script is None:
        return None, 0

    chars = [char for char in _script_text(text) if _is_script_letter(char)]
    if len(chars) == 0:
        return None, 0
    matches = sum(1 for char in chars if _matches_script(char, script))
    return matches / len(chars), len(chars)


def _script_text(text: str) -> str:
    without_tags = _HTML_TAG_RE.sub(" ", text)
    without_slots = _PLACEHOLDER_RE.sub(" ", without_tags)
    return _NUMBER_RE.sub(" ", without_slots)


def _expected_script(lang: str) -> str | None:
    code = _lang_code(lang)
    if code in _LATIN_LANGS:
        return "latin"
    if code in _CJK_LANGS:
        return "cjk"
    if code == "ko":
        return "hangul"
    if code in _CYRILLIC_LANGS:
        return "cyrillic"
    if code in _ARABIC_LANGS:
        return "arabic"
    if code == "hi":
        return "devanagari"
    if code == "th":
        return "thai"
    if code == "he":
        return "hebrew"
    return None


def _is_script_letter(char: str) -> bool:
    return char.isalpha() or _matches_script(char, "cjk")


def _matches_script(char: str, script: str) -> bool:
    codepoint = ord(char)
    name = unicodedata.name(char, "")
    if script == "cjk":
        return (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        )
    if script == "latin":
        return "LATIN" in name
    if script == "hangul":
        return "HANGUL" in name
    if script == "cyrillic":
        return "CYRILLIC" in name
    if script == "arabic":
        return "ARABIC" in name
    if script == "devanagari":
        return "DEVANAGARI" in name
    if script == "thai":
        return "THAI" in name
    if script == "hebrew":
        return "HEBREW" in name
    return False


def _control_ratio(text: str) -> float:
    if text == "":
        return 0.0
    control = 0
    for char in text:
        if char in "\t\n\r":
            continue
        if unicodedata.category(char).startswith("C"):
            control += 1
    return control / len(text)


def _longest_run(text: str) -> int:
    best = 0
    current = 0
    previous: str | None = None
    for char in text:
        if char == previous:
            current += 1
        else:
            current = 1
            previous = char
        best = max(best, current)
    return best


def _counts(values: Iterable[str]) -> Counter[str]:
    return Counter(values)


def _callback_key(callback: _Callback) -> str:
    if isinstance(callback, Bicleaner):
        return "bicleaner"
    return type(callback).__name__.lower()


def _lang_code(lang: str) -> str:
    return lang.lower().replace("_", "-").split("-", 1)[0]


__all__ = ["Bicleaner", "Label", "Predicate", "Profile", "Scorer"]
