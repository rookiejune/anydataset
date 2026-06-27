"""Debug translation-pair quality filtering with cached partitions.

The example provides a lightweight local TSV/CSV source and a picklable
predicate for canonical machine-translation samples. It outputs cached
`accept`, `review`, and `reject` partitions plus JSON metrics. Heavy semantic
models are intentionally outside this file; plug them into a later predicate
once the cheap rule layer is calibrated.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

from anydataset.types import SourceKey

from anydataset import (
    AnyDataset,
    FilterDecision,
    FilterRule,
    Modality,
    Role,
    Sample,
    Spec,
    TextItem,
    TextMeta,
    TextView,
    has_source,
    register_source,
)

TABLE_SOURCE = "translation_quality_table"

_NUMBER_RE = re.compile(r"[+-]?\d+(?:[.,:/-]\d+)*")
_PLACEHOLDER_RE = re.compile(
    r"(\{[^{}]+\}|%\([^)]+\)[#0+\-]?\d*(?:\.\d+)?[diouxXeEfFgGcrs](?![A-Za-z])|%[#0+\-]?\d*(?:\.\d+)?[diouxXeEfFgGcrs](?![A-Za-z])|\$\d+)"
)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]{0,80}>")
_SPACE_RE = re.compile(r"\s+")

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


@dataclass(frozen=True)
class QualityConfig:
    source_lang: str | None = None
    target_lang: str | None = None
    min_chars: int = 1
    review_min_ratio: float = 0.2
    review_max_ratio: float = 6.0
    reject_min_ratio: float = 0.05
    reject_max_ratio: float = 20.0
    min_script_ratio: float = 0.45
    reject_script_ratio: float = 0.2


@dataclass(frozen=True)
class TextPair:
    source: str
    target: str
    source_lang: str | None
    target_lang: str | None


@dataclass(frozen=True)
class TranslationQualityPredicate:
    config: QualityConfig

    def __call__(self, sample: Sample) -> FilterDecision:
        pair = text_pair(sample, self.config)
        decision = judge_pair(pair, self.config)
        return FilterDecision(
            label=decision["label"],
            metrics=decision["metrics"],
        )


@dataclass(frozen=True)
class BitextParser:
    source_column: str
    target_column: str
    source_lang: str | None
    target_lang: str | None

    def __call__(self, row: Mapping[str, str]) -> Sample:
        return parse_bitext_row(
            row,
            source_column=self.source_column,
            target_column=self.target_column,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        )


class BitextTableSource:
    def prepare(self, spec: Spec, cache_path: Path) -> BitextTableDataset:
        return BitextTableDataset(
            Path(spec.path),
            delimiter=str(spec.load_options.get("delimiter", "\t")),
            encoding=str(spec.load_options.get("encoding", "utf-8")),
            limit=_optional_int(spec.load_options.get("limit")),
        )


class BitextTableDataset:
    def __init__(
        self,
        path: Path,
        *,
        delimiter: str,
        encoding: str,
        limit: int | None,
    ) -> None:
        self.path = path
        self.delimiter = delimiter
        self.encoding = encoding
        self.limit = limit
        self._rows: tuple[Mapping[str, str], ...] | None = None

    def __len__(self) -> int:
        return len(self._load())

    def __getitem__(self, index: int) -> Mapping[str, str]:
        return self._load()[index]

    def _load(self) -> tuple[Mapping[str, str], ...]:
        if self._rows is not None:
            return self._rows

        rows: list[Mapping[str, str]] = []
        with self.path.open("r", encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            for index, row in enumerate(reader):
                if self.limit is not None and index >= self.limit:
                    break
                rows.append(dict(row))
        self._rows = tuple(rows)
        return self._rows


def main() -> None:
    args = parse_args()
    register_table_source(TABLE_SOURCE)
    dataset = AnyDataset(
        Spec(
            source=TABLE_SOURCE,
            path=str(args.input),
            load_options={
                "delimiter": delimiter(args.delimiter),
                "encoding": args.encoding,
                "limit": args.limit,
            },
        ),
        parse_fn=BitextParser(
            source_column=args.source_column,
            target_column=args.target_column,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        ),
        cache_root=args.dataset_cache_root,
    )
    config = QualityConfig(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        review_min_ratio=args.review_min_ratio,
        review_max_ratio=args.review_max_ratio,
        reject_min_ratio=args.reject_min_ratio,
        reject_max_ratio=args.reject_max_ratio,
        min_script_ratio=args.min_script_ratio,
        reject_script_ratio=args.reject_script_ratio,
    )
    rule = FilterRule(args.rule_name, TranslationQualityPredicate(config))
    result = rule.apply(
        dataset,
        metrics=True,
        num_workers=args.num_workers,
        cache_root=args.cache_root,
    )
    summary: dict[str, Any] = {
        "cache_path": str(result.cache_path),
        "metrics_path": None
        if result.metrics_path is None
        else str(result.metrics_path),
        "counts": dict(result.counts),
        "labels": result.labels,
    }
    if args.preview > 0:
        summary["preview"] = preview_metrics(result.iter_metrics(), args.preview)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cached translation-quality partitions for a local bitext table."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--source-lang")
    parser.add_argument("--target-lang")
    parser.add_argument("--delimiter", default="tab")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rule-name", default="mt_quality_rules_v1")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--dataset-cache-root", type=Path)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--preview", type=int, default=5)
    parser.add_argument("--review-min-ratio", type=float, default=0.2)
    parser.add_argument("--review-max-ratio", type=float, default=6.0)
    parser.add_argument("--reject-min-ratio", type=float, default=0.05)
    parser.add_argument("--reject-max-ratio", type=float, default=20.0)
    parser.add_argument("--min-script-ratio", type=float, default=0.45)
    parser.add_argument("--reject-script-ratio", type=float, default=0.2)
    return parser.parse_args()


def parse_bitext_row(
    row: Mapping[str, str],
    *,
    source_column: str,
    target_column: str,
    source_lang: str | None,
    target_lang: str | None,
) -> Sample:
    return {
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: row[source_column]},
            meta={} if source_lang is None else {TextMeta.LANG: source_lang},
        ),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: row[target_column]},
            meta={} if target_lang is None else {TextMeta.LANG: target_lang},
        ),
    }


def text_pair(sample: Sample, config: QualityConfig) -> TextPair:
    source = cast(TextItem, sample[Role.SOURCE, Modality.TEXT])
    target = cast(TextItem, sample[Role.TARGET, Modality.TEXT])
    source_text = source.views[TextView.TEXT]
    target_text = target.views[TextView.TEXT]
    if not isinstance(source_text, str) or not isinstance(target_text, str):
        raise TypeError("translation quality filter requires string text views.")
    return TextPair(
        source=source_text,
        target=target_text,
        source_lang=_text_lang(source, config.source_lang),
        target_lang=_text_lang(target, config.target_lang),
    )


def judge_pair(pair: TextPair, config: QualityConfig) -> dict[str, Any]:
    source = normalize_space(pair.source)
    target = normalize_space(pair.target)
    source_len = nonspace_len(source)
    target_len = nonspace_len(target)
    ratio = length_ratio(source_len, target_len)
    source_script = script_ratio(source, pair.source_lang)
    target_script = script_ratio(target, pair.target_lang)
    source_numbers = numbers(source)
    target_numbers = numbers(target)
    source_number_values = number_values(source_numbers)
    target_number_values = number_values(target_numbers)
    number_value_overlap = overlap(source_number_values, target_number_values)
    number_surface_overlap = overlap(source_numbers, target_numbers)
    source_tags = html_tags(source)
    target_tags = html_tags(target)
    source_slots = placeholders(source)
    target_slots = placeholders(target)

    reject_flags: list[str] = []
    review_flags: list[str] = []
    if source_len < config.min_chars:
        reject_flags.append("empty_source")
    if target_len < config.min_chars:
        reject_flags.append("empty_target")
    if ratio < config.reject_min_ratio:
        reject_flags.append("target_too_short")
    elif ratio < config.review_min_ratio:
        review_flags.append("target_short")
    if ratio > config.reject_max_ratio:
        reject_flags.append("target_too_long")
    elif ratio > config.review_max_ratio:
        review_flags.append("target_long")
    if different_langs(pair) and source.casefold() == target.casefold():
        reject_flags.append("identical_text")
    if control_ratio(source) > 0.02 or control_ratio(target) > 0.02:
        reject_flags.append("control_chars")
    if longest_run(source) >= 24 or longest_run(target) >= 24:
        review_flags.append("long_repeated_run")
    _script_flag(source_script, "source_script", config, reject_flags, review_flags)
    _script_flag(target_script, "target_script", config, reject_flags, review_flags)
    if source_tags != target_tags:
        review_flags.append("html_tag_mismatch")
    if source_slots != target_slots:
        reject_flags.append("placeholder_mismatch")
    if number_value_overlap < 1.0 and (source_numbers or target_numbers):
        reject_flags.append("number_value_mismatch")
    elif number_surface_overlap < 1.0 and (source_numbers or target_numbers):
        review_flags.append("number_surface_mismatch")

    label = "reject" if reject_flags else "review" if review_flags else "accept"
    flags = reject_flags + review_flags
    metrics = {
        "source_chars": source_len,
        "target_chars": target_len,
        "char_ratio": round(ratio, 6),
        "source_lang": pair.source_lang,
        "target_lang": pair.target_lang,
        "source_script_ratio": _rounded(source_script),
        "target_script_ratio": _rounded(target_script),
        "number_overlap": round(number_value_overlap, 6),
        "number_value_overlap": round(number_value_overlap, 6),
        "number_surface_overlap": round(number_surface_overlap, 6),
        "source_numbers": list(source_numbers),
        "target_numbers": list(target_numbers),
        "source_number_values": list(source_number_values),
        "target_number_values": list(target_number_values),
        "flags": flags,
        "quality_score": quality_score(reject_flags, review_flags),
    }
    return {"label": label, "metrics": metrics}


def register_table_source(source: SourceKey) -> None:
    if not has_source(source):
        register_source(source, BitextTableSource)


def delimiter(value: str) -> str:
    match value:
        case "tab" | "\\t":
            return "\t"
        case "comma" | ",":
            return ","
        case _:
            if len(value) != 1:
                raise ValueError("delimiter must be one character, `tab`, or `comma`.")
            return value


def preview_metrics(rows: Any, limit: int) -> list[Mapping[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        if index >= limit:
            break
        output.append(row)
    return output


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
    if not _is_plain_number(token):
        return token
    normalized = token.replace(",", ".")
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return token
    return format(value.normalize(), "f")


def placeholders(text: str) -> frozenset[str]:
    return frozenset(match.group(0) for match in _PLACEHOLDER_RE.finditer(text))


def html_tags(text: str) -> frozenset[str]:
    return frozenset(match.group(0).lower() for match in _HTML_TAG_RE.finditer(text))


def overlap(source: Iterable[str], target: Iterable[str]) -> float:
    source_counts = Counter(source)
    target_counts = Counter(target)
    if not source_counts and not target_counts:
        return 1.0
    union = source_counts | target_counts
    if not union:
        return 1.0
    intersection = source_counts & target_counts
    return sum(intersection.values()) / sum(union.values())


def script_ratio(text: str, lang: str | None) -> float | None:
    script = expected_script(lang)
    if script is None:
        return None
    cleaned = script_text(text)
    chars = [char for char in cleaned if is_script_letter(char)]
    if not chars:
        return None
    return sum(1 for char in chars if matches_script(char, script)) / len(chars)


def script_text(text: str) -> str:
    without_tags = _HTML_TAG_RE.sub(" ", text)
    without_slots = _PLACEHOLDER_RE.sub(" ", without_tags)
    without_numbers = _NUMBER_RE.sub(" ", without_slots)
    return without_numbers


def expected_script(lang: str | None) -> str | None:
    if lang is None:
        return None
    code = lang.lower().replace("_", "-").split("-", 1)[0]
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


def _is_plain_number(token: str) -> bool:
    value = token.replace(",", ".")
    if value.count(".") > 1:
        return False
    return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value))


def is_script_letter(char: str) -> bool:
    return char.isalpha() or matches_script(char, "cjk")


def matches_script(char: str, script: str) -> bool:
    codepoint = ord(char)
    name = unicodedata.name(char, "")
    match script:
        case "cjk":
            return (
                0x3400 <= codepoint <= 0x4DBF
                or 0x4E00 <= codepoint <= 0x9FFF
                or 0xF900 <= codepoint <= 0xFAFF
            )
        case "latin":
            return "LATIN" in name
        case "hangul":
            return "HANGUL" in name
        case "cyrillic":
            return "CYRILLIC" in name
        case "arabic":
            return "ARABIC" in name
        case "devanagari":
            return "DEVANAGARI" in name
        case "thai":
            return "THAI" in name
        case "hebrew":
            return "HEBREW" in name
        case _:
            return False


def control_ratio(text: str) -> float:
    if not text:
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


def different_langs(pair: TextPair) -> bool:
    if pair.source_lang is None or pair.target_lang is None:
        return False
    return pair.source_lang.lower() != pair.target_lang.lower()


def quality_score(reject_flags: list[str], review_flags: list[str]) -> float:
    score = 1.0 - 0.35 * len(reject_flags) - 0.12 * len(review_flags)
    return round(max(score, 0.0), 6)


def _script_flag(
    ratio: float | None,
    name: str,
    config: QualityConfig,
    reject_flags: list[str],
    review_flags: list[str],
) -> None:
    if ratio is None:
        return
    if ratio < config.reject_script_ratio:
        reject_flags.append(f"low_{name}")
    elif ratio < config.min_script_ratio:
        review_flags.append(f"low_{name}")


def _text_lang(item: TextItem, fallback: str | None) -> str | None:
    value = item.meta.get(TextMeta.LANG, fallback)
    if value is None:
        return None
    return str(value)


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    main()
