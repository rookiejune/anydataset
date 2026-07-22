"""Composable quality rule execution.

The chain owns cross-rule label transitions. Individual quality predicates only
report their own label, reasons, and metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import auto

from .._compat import StrEnum
from ..filter import FilterDecision
from ..filter.rules import label as filter_label
from ..filter.types import FilterOutput, JsonValue
from ..types import Sample


class QualityLabel(StrEnum):
    ACCEPT = auto()
    REVIEW = auto()
    REJECT = auto()


QualityRule = Callable[[Sample], FilterOutput]


@dataclass(frozen=True)
class Rule:
    name: str
    predicate: QualityRule

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("quality rule name must be a string.")
        if self.name == "":
            raise ValueError("quality rule name must not be empty.")
        if not callable(self.predicate):
            raise TypeError("quality rule predicate must be callable.")


@dataclass(frozen=True)
class QualityChain:
    rules: Sequence[Rule]

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if len(rules) == 0:
            raise ValueError("quality rule chain must not be empty.")
        names = [rule.name for rule in rules]
        if len(set(names)) != len(names):
            raise ValueError("quality rule names must be unique.")
        object.__setattr__(self, "rules", rules)

    def __call__(self, sample: Sample) -> FilterDecision:
        label = QualityLabel.ACCEPT
        rows: list[dict[str, JsonValue]] = []
        transitions: list[dict[str, str]] = []
        flags: list[str] = []

        for rule in self.rules:
            previous = label
            decision = _decision(rule.predicate(sample))
            current = _label(decision.label)
            label = _combine(label, current)
            metrics = dict(decision.metrics)
            rows.append(
                {
                    "rule": rule.name,
                    "label": current.value,
                    "metrics": metrics,
                }
            )
            flags.extend(_flags(rule.name, metrics))
            if label != previous:
                transitions.append(
                    {
                        "rule": rule.name,
                        "from": previous.value,
                        "to": label.value,
                    }
                )

        return FilterDecision(
            label=label,
            metrics={
                "decision": label.value,
                "rules": rows,
                "transitions": transitions,
                "flags": flags,
            },
        )


def _decision(output: FilterOutput) -> FilterDecision:
    if isinstance(output, FilterDecision):
        return output
    return FilterDecision(label=output, metrics={})


def _label(value: object) -> QualityLabel:
    normalized = filter_label(value)
    if normalized == QualityLabel.ACCEPT.value:
        return QualityLabel.ACCEPT
    if normalized == QualityLabel.REVIEW.value:
        return QualityLabel.REVIEW
    if normalized == QualityLabel.REJECT.value:
        return QualityLabel.REJECT
    raise ValueError(f"unsupported quality label: {normalized!r}.")


def _combine(previous: QualityLabel, current: QualityLabel) -> QualityLabel:
    if current == QualityLabel.REJECT:
        return QualityLabel.REJECT
    if current == QualityLabel.ACCEPT and previous == QualityLabel.REJECT:
        return QualityLabel.REVIEW
    if current == QualityLabel.REVIEW and previous == QualityLabel.ACCEPT:
        return QualityLabel.REVIEW
    return previous


def _flags(rule: str, metrics: Mapping[str, JsonValue]) -> list[str]:
    value = metrics.get("flags")
    if not isinstance(value, list):
        return []
    return [f"{rule}:{flag}" for flag in value if isinstance(flag, str)]


__all__ = ["QualityChain", "QualityLabel", "QualityRule", "Rule"]
