"""Quality rules built on canonical samples.

This package provides reusable rule logic above `anydataset.filter`.
It does not own physical data loading, cache construction, or dataset presets.
"""

from __future__ import annotations

from .rules import QualityChain, QualityLabel, QualityRule, Rule
from .speech import SpeechQuality, SpeechQualityProfile
from .text import ChineseGEC, TextAcceptability, TextQuality, TextQualityProfile
from .translation import (
    Bicleaner,
    Scorer,
    TranslationQuality,
    TranslationQualityProfile,
)

__all__ = [
    "Bicleaner",
    "ChineseGEC",
    "QualityChain",
    "QualityLabel",
    "QualityRule",
    "Rule",
    "Scorer",
    "SpeechQuality",
    "SpeechQualityProfile",
    "TextAcceptability",
    "TextQuality",
    "TextQualityProfile",
    "TranslationQuality",
    "TranslationQualityProfile",
]
