"""Debug translation-pair quality filtering with cached partitions.

The example provides a lightweight local TSV/CSV source and a picklable
predicate for canonical machine-translation samples. It outputs cached
`clean`, `usable`, `review`, and `reject` partitions plus JSON audit rows. Heavy semantic
models are intentionally outside this file; plug them into a later predicate
once the cheap rule layer is calibrated.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anydataset import (
    AnyDataset,
    FilterRule,
    has_source,
    register_source,
)
from anydataset.quality.translation import Predicate, Profile
from anydataset.types import (
    Modality,
    Role,
    Sample,
    SourceKey,
    Spec,
    TextItem,
    TextMeta,
    TextView,
)

TABLE_SOURCE = "translation_quality_table"


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


@dataclass(frozen=True)
class DatasetFactory:
    spec: Spec
    parser: BitextParser

    def __call__(self) -> AnyDataset:
        register_table_source(self.spec.source)
        return AnyDataset(self.spec, parse_fn=self.parser)


@dataclass(frozen=True)
class PredicateFactory:
    profile: Profile

    def __call__(self) -> Predicate:
        return Predicate(self.profile)


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
    dataset_factory = DatasetFactory(
        spec=Spec(
            source=TABLE_SOURCE,
            path=str(args.input),
            load_options={
                "delimiter": delimiter(args.delimiter),
                "encoding": args.encoding,
                "limit": args.limit,
            },
        ),
        parser=BitextParser(
            source_column=args.source_column,
            target_column=args.target_column,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        ),
    )
    if args.source_lang is None or args.target_lang is None:
        raise ValueError("source_lang and target_lang are required.")

    profile = Profile(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        review_min_ratio=args.review_min_ratio,
        review_max_ratio=args.review_max_ratio,
        reject_min_ratio=args.reject_min_ratio,
        reject_max_ratio=args.reject_max_ratio,
        min_script_ratio=args.min_script_ratio,
        reject_script_ratio=args.reject_script_ratio,
    )

    rule = FilterRule(args.rule_name, PredicateFactory(profile))
    result = rule.apply(
        dataset_factory=dataset_factory,
        metrics=True,
        device=args.device,
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
    parser.add_argument("--device", default="cpu")
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


def register_table_source(source: SourceKey) -> None:
    if not has_source(source):
        register_source(source, BitextTableSource)


def delimiter(value: str) -> str:
    if value in {"tab", "\\t"}:
        return "\t"
    if value in {"comma", ","}:
        return ","
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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    main()
