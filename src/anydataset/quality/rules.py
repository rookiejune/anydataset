"""Composable quality rule execution.

The chain owns cross-rule label transitions. Individual quality predicates only
report their own label, reasons, and metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto

from .._compat import StrEnum
from .._immutable import Immutable
from ..filter import FilterDecision
from ..filter.rules import label as filter_label
from ..filter.types import (
    BatchFilterPredicate,
    FilterOutput,
    JsonValue,
    validate_metrics,
)
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


@dataclass(init=False, unsafe_hash=True)
class QualityChain(Immutable):
    rules: tuple[Rule, ...]

    def __init__(self, rules: Sequence[Rule]) -> None:
        rules = tuple(rules)
        if len(rules) == 0:
            raise ValueError("quality rule chain must not be empty.")
        names = [rule.name for rule in rules]
        if len(set(names)) != len(names):
            raise ValueError("quality rule names must be unique.")
        self.rules = rules
        self.seal()

    def __call__(self, sample: Sample) -> FilterDecision:
        state = _ChainState()
        for rule in self.rules:
            state.apply(rule, rule.predicate(sample))
        return state.decision()

    def call_batch(self, samples: Sequence[Sample]) -> Sequence[FilterDecision]:
        samples = tuple(samples)
        if not samples:
            return ()
        states = [_ChainState() for _sample in samples]
        for rule in self.rules:
            outputs = _rule_outputs(rule.predicate, samples)
            for state, output in zip(states, outputs):
                state.apply(rule, output)
        return tuple(state.decision() for state in states)


@dataclass
class _ChainState:
    label: QualityLabel = QualityLabel.ACCEPT
    rows: list[dict[str, JsonValue]] = field(default_factory=list)
    transitions: list[dict[str, str]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def apply(self, rule: Rule, output: FilterOutput) -> None:
        previous = self.label
        decision = _decision(output)
        current = _label(decision.label)
        self.label = _combine(self.label, current)
        metrics = validate_metrics(decision.metrics)
        self.rows.append(
            {
                "rule": rule.name,
                "label": current.value,
                "metrics": metrics,
            }
        )
        self.flags.extend(_flags(rule.name, metrics))
        if self.label != previous:
            self.transitions.append(
                {
                    "rule": rule.name,
                    "from": previous.value,
                    "to": self.label.value,
                }
            )

    def decision(self) -> FilterDecision:
        return FilterDecision(
            label=self.label,
            metrics={
                "decision": self.label.value,
                "rules": self.rows,
                "transitions": self.transitions,
                "flags": self.flags,
            },
        )


def _rule_outputs(
    predicate: QualityRule,
    samples: Sequence[Sample],
) -> Sequence[FilterOutput]:
    if not isinstance(predicate, BatchFilterPredicate):
        return tuple(predicate(sample) for sample in samples)

    outputs = predicate.call_batch(samples)
    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
        raise TypeError("quality rule call_batch() must return an ordered sequence.")
    if len(outputs) != len(samples):
        raise ValueError(
            "quality rule call_batch() must return one output per input sample; "
            f"received {len(outputs)} outputs for {len(samples)} samples."
        )
    return outputs


def _decision(output: FilterOutput) -> FilterDecision:
    if isinstance(output, FilterDecision):
        return output
    return FilterDecision(label=output, metrics={})


def _label(value: object) -> QualityLabel:
    if not isinstance(value, (bool, str, Enum)):
        raise TypeError("quality label must be bool, str, or Enum.")
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
    if "flags" not in metrics:
        return []
    value = metrics["flags"]
    if not isinstance(value, list) or not all(isinstance(flag, str) for flag in value):
        raise TypeError(
            f"quality rule {rule!r} metrics 'flags' must be a list of strings."
        )
    return [f"{rule}:{flag}" for flag in value]


__all__ = ["QualityChain", "QualityLabel", "QualityRule", "Rule"]
