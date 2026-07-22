"""Atomic text quality predicates for canonical text items."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ..filter import FilterDecision
from ..types import Lang, Role, Sample
from . import _text
from ._text import TextQualityProfile
from .rules import QualityLabel

_DEFAULT_ACCEPTABILITY_MODELS = {
    Lang.EN: "textattack/roberta-base-CoLA",
}
_DEFAULT_GEC_MODELS = {
    Lang.ZH: "shibing624/mengzi-t5-base-chinese-correction",
}
_ACCEPT_LABELS = frozenset({"1", "LABEL_1", "ACCEPT", "ACCEPTABLE"})
_REJECT_LABELS = frozenset({"0", "LABEL_0", "REJECT", "UNACCEPTABLE"})


@dataclass(frozen=True)
class TextQuality:
    role: Role
    lang: Lang
    profile: TextQualityProfile = field(default_factory=TextQualityProfile)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", Role(self.role))
        object.__setattr__(self, "lang", _lang("text quality lang", self.lang))

    def __call__(self, sample: Sample) -> FilterDecision:
        metrics = _text.metrics(sample, self.role, self.lang)
        item_flags = [
            _flag(self.role, finding.flag)
            for finding in _text.findings(metrics, self.profile)
        ]
        label = QualityLabel.REJECT if item_flags else QualityLabel.ACCEPT
        return FilterDecision(
            label=label,
            metrics={
                "decision": label.value,
                "flags": item_flags,
                "items": [
                    {
                        "role": self.role.value,
                        "text": metrics.text,
                        "lang": metrics.lang.value,
                        "expected_lang": metrics.expected_lang.value,
                        "flags": item_flags,
                    }
                ],
            },
        )


@dataclass(frozen=True)
class TextAcceptability:
    role: Role
    lang: Lang
    min_score: float = 0.6
    model: str | None = None
    device: int | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", Role(self.role))
        object.__setattr__(self, "lang", _lang("text acceptability lang", self.lang))
        object.__setattr__(
            self,
            "min_score",
            _text.unit_ratio("text acceptability min_score", self.min_score),
        )
        object.__setattr__(
            self,
            "model",
            _model("text acceptability model", self.lang, self.model),
        )

    def __call__(self, sample: Sample) -> FilterDecision:
        metrics = _text.metrics(sample, self.role, self.lang)
        flags: list[str] = []
        score: float | None = None
        if metrics.lang != metrics.expected_lang:
            flags.append(_flag(self.role, "lang_mismatch"))
        if not metrics.valid:
            flags.append(_flag(self.role, "invalid_text"))

        if not flags:
            score = _text.unit_ratio(
                "text acceptability model output",
                _score(metrics.text, model=str(self.model), device=self.device),
            )
            if score < self.min_score:
                flags.append(_flag(self.role, "acceptability_low"))

        label = QualityLabel.REJECT if flags else QualityLabel.ACCEPT
        return FilterDecision(
            label=label,
            metrics={
                "decision": label.value,
                "flags": flags,
                "items": [
                    {
                        "role": self.role.value,
                        "text": metrics.text,
                        "lang": metrics.lang.value,
                        "expected_lang": metrics.expected_lang.value,
                        "acceptability_model": self.model,
                        "acceptability_score": (
                            None if score is None else round(score, 6)
                        ),
                        "flags": flags,
                    }
                ],
            },
        )


@dataclass(frozen=True)
class ChineseGEC:
    role: Role
    max_edit_ratio: float = 0.05
    max_edits: int | None = None
    model: str | None = None
    device: int | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", Role(self.role))
        object.__setattr__(
            self,
            "max_edit_ratio",
            _text.unit_ratio("ChineseGEC max_edit_ratio", self.max_edit_ratio),
        )
        object.__setattr__(
            self,
            "max_edits",
            _optional_non_negative_int("ChineseGEC max_edits", self.max_edits),
        )
        object.__setattr__(
            self,
            "model",
            _default_model("ChineseGEC model", Lang.ZH, self.model, _DEFAULT_GEC_MODELS),
        )

    def __call__(self, sample: Sample) -> FilterDecision:
        metrics = _text.metrics(sample, self.role, Lang.ZH)
        flags: list[str] = []
        corrected: str | None = None
        edit_count: int | None = None
        edit_ratio: float | None = None

        if metrics.lang != metrics.expected_lang:
            flags.append(_flag(self.role, "lang_mismatch"))
        if not metrics.valid:
            flags.append(_flag(self.role, "invalid_text"))

        if not flags:
            corrected = _text.normalize_space(
                _correct(metrics.text, model=str(self.model), device=self.device)
            )
            source = metrics.normalized
            edit_count = _edit_distance(source, corrected)
            edit_ratio = edit_count / max(_text.nonspace_len(source), 1)
            if self.max_edits is not None and edit_count > self.max_edits:
                flags.append(_flag(self.role, "gec_edit_count_high"))
            if edit_ratio > self.max_edit_ratio:
                flags.append(_flag(self.role, "gec_edit_ratio_high"))

        label = QualityLabel.REJECT if flags else QualityLabel.ACCEPT
        return FilterDecision(
            label=label,
            metrics={
                "decision": label.value,
                "flags": flags,
                "items": [
                    {
                        "role": self.role.value,
                        "text": metrics.text,
                        "corrected_text": corrected,
                        "lang": metrics.lang.value,
                        "expected_lang": metrics.expected_lang.value,
                        "gec_model": self.model,
                        "gec_edit_count": edit_count,
                        "gec_edit_ratio": (
                            None if edit_ratio is None else round(edit_ratio, 6)
                        ),
                        "flags": flags,
                    }
                ],
            },
        )


def _flag(role: Role, flag: str) -> str:
    return f"{role.value}_{flag}"


def _lang(name: str, value: Lang) -> Lang:
    if not isinstance(value, Lang):
        raise TypeError(f"{name} must be a Lang value.")
    if value == Lang.UND:
        raise ValueError(f"{name} must be explicit.")
    return value


def _model(name: str, lang: Lang, value: str | None) -> str:
    return _default_model(name, lang, value, _DEFAULT_ACCEPTABILITY_MODELS)


def _default_model(
    name: str,
    lang: Lang,
    value: str | None,
    defaults: dict[Lang, str],
) -> str:
    if value is not None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string or None.")
        if value == "":
            raise ValueError(f"{name} must not be empty.")
        return value
    try:
        return defaults[lang]
    except KeyError as exc:
        raise ValueError(
            f"{name} has no default model for {lang.value!r}."
        ) from exc


def _score(text: str, *, model: str, device: int | str | None) -> float:
    classifier = _classifier(model, device)
    try:
        output = classifier(text, truncation=True, top_k=None)
    except TypeError:
        output = classifier(text, truncation=True, return_all_scores=True)
    return _acceptability_score(model, output)


@lru_cache(maxsize=8)
def _classifier(model: str, device: int | str | None) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "TextAcceptability requires transformers; install "
            "anydataset[text] or transformers."
        ) from exc
    kwargs: dict[str, Any] = {"model": model}
    if device is not None:
        kwargs["device"] = device
    return pipeline("text-classification", **kwargs)


def _correct(text: str, *, model: str, device: int | str | None) -> str:
    return _generated_text(_corrector(model, device)(text, truncation=True))


@lru_cache(maxsize=8)
def _corrector(model: str, device: int | str | None) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "ChineseGEC requires transformers; install anydataset[text] "
            "or transformers."
        ) from exc
    kwargs: dict[str, Any] = {"model": model}
    if device is not None:
        kwargs["device"] = device
    return pipeline("text2text-generation", **kwargs)


def _acceptability_score(model: str, output: Any) -> float:
    rows = _rows(output)
    accept_scores = [
        _score_value(row)
        for row in rows
        if _label_value(row) in _ACCEPT_LABELS
    ]
    if accept_scores:
        return max(accept_scores)

    reject_scores = [
        _score_value(row)
        for row in rows
        if _label_value(row) in _REJECT_LABELS
    ]
    if reject_scores:
        return 1.0 - max(reject_scores)
    labels = ", ".join(sorted(_label_value(row) for row in rows))
    raise ValueError(
        f"text acceptability model {model!r} returned unsupported labels: {labels}."
    )


def _rows(output: Any) -> list[dict[str, Any]]:
    if isinstance(output, dict):
        return [output]
    if not isinstance(output, list) or len(output) == 0:
        raise TypeError("text acceptability model output must be a non-empty list.")
    first = output[0]
    if isinstance(first, list):
        return _rows(first)
    if not all(isinstance(row, dict) for row in output):
        raise TypeError("text acceptability model output rows must be mappings.")
    return output


def _label_value(row: dict[str, Any]) -> str:
    value = row.get("label")
    if not isinstance(value, str) or value == "":
        raise ValueError("text acceptability model output requires string labels.")
    return value.upper()


def _score_value(row: dict[str, Any]) -> float:
    return _text.unit_ratio("text acceptability label score", row.get("score"))


def _generated_text(output: Any) -> str:
    row = _rows(output)[0]
    for key in ("generated_text", "summary_text", "translation_text", "text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    raise ValueError("ChineseGEC model output requires generated text.")


def _edit_distance(source: str, target: str) -> int:
    if source == target:
        return 0
    previous = list(range(len(target) + 1))
    for source_index, source_char in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_char in enumerate(target, start=1):
            current.append(
                min(
                    previous[target_index] + 1,
                    current[target_index - 1] + 1,
                    previous[target_index - 1] + int(source_char != target_char),
                )
            )
        previous = current
    return previous[-1]


def _optional_non_negative_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


__all__ = [
    "ChineseGEC",
    "TextAcceptability",
    "TextQuality",
    "TextQualityProfile",
]
