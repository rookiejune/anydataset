"""Single-text quality metrics and lightweight rule findings.

This private module owns reusable text inspection logic for quality rules.
It does not decide task-level labels or compare source/target pairs.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import cached_property
from math import isfinite

from .._validation import positive_int
from ..types import Lang, Modality, Role, Sample, TextItem, TextMeta, TextView

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
        Lang.AF,
        Lang.CA,
        Lang.CS,
        Lang.DA,
        Lang.DE,
        Lang.EN,
        Lang.ES,
        Lang.ET,
        Lang.FI,
        Lang.FR,
        Lang.HR,
        Lang.HU,
        Lang.ID,
        Lang.IT,
        Lang.MS,
        Lang.NL,
        Lang.NO,
        Lang.PL,
        Lang.PT,
        Lang.RO,
        Lang.SK,
        Lang.SL,
        Lang.SV,
        Lang.TR,
        Lang.VI,
    }
)
_CJK_LANGS = frozenset({Lang.ZH, Lang.JA})
_CYRILLIC_LANGS = frozenset(
    {Lang.BE, Lang.BG, Lang.KK, Lang.MK, Lang.RU, Lang.SR, Lang.UK}
)
_ARABIC_LANGS = frozenset({Lang.AR, Lang.FA, Lang.PS, Lang.UR})


@dataclass
class TextQualityProfile:
    min_chars: int = 1
    min_script_ratio: float = 0.45
    reject_script_ratio: float = 0.2
    min_script_chars: int = 4
    max_control_ratio: float = 0.02
    max_repeated_run: int = 24

    def __post_init__(self) -> None:
        self.min_chars = positive_int("min_chars", self.min_chars)
        self.min_script_chars = positive_int(
            "min_script_chars",
            self.min_script_chars,
        )
        self.max_repeated_run = positive_int(
            "max_repeated_run",
            self.max_repeated_run,
        )
        self.reject_script_ratio = unit_ratio(
            "reject_script_ratio",
            self.reject_script_ratio,
        )
        self.min_script_ratio = unit_ratio(
            "min_script_ratio",
            self.min_script_ratio,
        )
        if self.reject_script_ratio > self.min_script_ratio:
            raise ValueError("reject_script_ratio must be <= min_script_ratio.")
        self.max_control_ratio = unit_ratio(
            "max_control_ratio",
            self.max_control_ratio,
        )


@dataclass
class Metrics:
    text: str
    lang: Lang
    expected_lang: Lang
    valid: bool

    @cached_property
    def normalized(self) -> str:
        return normalize_space(self.text)

    @cached_property
    def chars(self) -> int:
        return nonspace_len(self.normalized)

    @cached_property
    def control_ratio(self) -> float:
        return control_ratio(self.normalized)

    @cached_property
    def repeated_run(self) -> int:
        return longest_run(self.normalized)

    @cached_property
    def script_ratio(self) -> float | None:
        return script_ratio(self.normalized, self.lang)[0]

    @cached_property
    def script_chars(self) -> int:
        return script_ratio(self.normalized, self.lang)[1]

    @cached_property
    def numbers(self) -> tuple[str, ...]:
        return numbers(self.normalized)

    @cached_property
    def number_values(self) -> tuple[str, ...]:
        return number_values(self.numbers)

    @cached_property
    def placeholders(self) -> tuple[str, ...]:
        return placeholders(self.normalized)

    @cached_property
    def html_tags(self) -> tuple[str, ...]:
        return html_tags(self.normalized)

    @cached_property
    def complex_numbers(self) -> bool:
        if _COMPLEX_NUMBER_RE.search(self.normalized) is not None:
            return True
        return any(normalized_number(token) is None for token in self.numbers)


@dataclass(frozen=True)
class Finding:
    flag: str


def metrics(sample: Sample, role: Role, fallback_lang: Lang) -> Metrics:
    item = sample.get((role, Modality.TEXT))
    if not isinstance(item, TextItem):
        return Metrics(text="", lang=fallback_lang, expected_lang=fallback_lang, valid=False)

    value = item.views.get(TextView.TEXT)
    if not isinstance(value, str):
        return Metrics(
            text="",
            lang=item_lang(item, fallback_lang),
            expected_lang=fallback_lang,
            valid=False,
        )
    return Metrics(
        text=value,
        lang=item_lang(item, fallback_lang),
        expected_lang=fallback_lang,
        valid=True,
    )


def item_lang(item: TextItem, fallback: Lang) -> Lang:
    value = item.meta.get(TextMeta.LANG)
    if value is None:
        return fallback
    if not isinstance(value, Lang):
        raise TypeError("TextMeta.LANG must be a Lang value.")
    return value


def findings(metrics: Metrics, profile: TextQualityProfile) -> tuple[Finding, ...]:
    output: list[Finding] = []
    if metrics.lang != metrics.expected_lang:
        output.append(Finding("lang_mismatch"))
    if not metrics.valid:
        output.append(Finding("invalid_text"))
    if metrics.chars < profile.min_chars:
        output.append(Finding("empty_text"))
    if metrics.control_ratio > profile.max_control_ratio:
        output.append(Finding("control_chars"))
    if metrics.repeated_run >= profile.max_repeated_run:
        output.append(Finding("long_repeated_run"))

    ratio = metrics.script_ratio
    if ratio is None or metrics.script_chars < profile.min_script_chars:
        return tuple(output)
    if ratio < profile.reject_script_ratio:
        output.append(Finding("script_mismatch"))
    elif ratio < profile.min_script_ratio:
        output.append(Finding("script_low"))
    return tuple(output)


def normalize_space(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def nonspace_len(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def length_ratio(source_len: int, target_len: int) -> float:
    if source_len == 0:
        return 0.0
    return target_len / source_len


def numbers(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _NUMBER_RE.finditer(text))


def number_values(tokens: Iterable[str]) -> tuple[str, ...]:
    return tuple(number_value(token) for token in tokens)


def number_value(token: str) -> str:
    normalized = normalized_number(token)
    if normalized is None:
        return token
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return token
    return format(value.normalize(), "f")


def normalized_number(token: str) -> str | None:
    if _GROUPED_NUMBER_RE.fullmatch(token) is not None:
        return token.replace(",", "")
    if _DECIMAL_RE.fullmatch(token) is not None:
        return token
    return None


def placeholders(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _PLACEHOLDER_RE.finditer(text))


def html_tags(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _HTML_TAG_RE.finditer(text))


def overlap(source: Iterable[str], target: Iterable[str]) -> float:
    source_counts = Counter(source)
    target_counts = Counter(target)
    if len(source_counts) == 0 and len(target_counts) == 0:
        return 1.0

    union = source_counts | target_counts
    if len(union) == 0:
        return 1.0
    intersection = source_counts & target_counts
    return sum(intersection.values()) / sum(union.values())


def counts(values: Iterable[str]) -> Counter[str]:
    return Counter(values)


def script_ratio(text: str, lang: Lang) -> tuple[float | None, int]:
    script = expected_script(lang)
    if script is None:
        return None, 0

    chars = [char for char in script_text(text) if is_script_letter(char)]
    if len(chars) == 0:
        return None, 0
    matches = sum(1 for char in chars if matches_script(char, script))
    return matches / len(chars), len(chars)


def script_text(text: str) -> str:
    without_tags = _HTML_TAG_RE.sub(" ", text)
    without_slots = _PLACEHOLDER_RE.sub(" ", without_tags)
    return _NUMBER_RE.sub(" ", without_slots)


def expected_script(lang: Lang) -> str | None:
    if lang in _LATIN_LANGS:
        return "latin"
    if lang in _CJK_LANGS:
        return "cjk"
    if lang == Lang.KO:
        return "hangul"
    if lang in _CYRILLIC_LANGS:
        return "cyrillic"
    if lang in _ARABIC_LANGS:
        return "arabic"
    if lang == Lang.HI:
        return "devanagari"
    if lang == Lang.TH:
        return "thai"
    if lang == Lang.HE:
        return "hebrew"
    return None


def is_script_letter(char: str) -> bool:
    return char.isalpha() or matches_script(char, "cjk")


def matches_script(char: str, script: str) -> bool:
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


def control_ratio(text: str) -> float:
    if text == "":
        return 0.0
    control = 0
    for char in text:
        if char in "\t\n\r":
            continue
        if unicodedata.category(char).startswith("C"):
            control += 1
    return control / len(text)


def longest_run(text: str) -> int:
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


def unit_ratio(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result < 0 or result > 1:
        raise ValueError(f"{name} must be between 0 and 1.")
    return result
