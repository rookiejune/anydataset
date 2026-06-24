from __future__ import annotations

from enum import auto

from ..enums import AutoNameEnum


type ModalityRole = str | None


class ModalityKey(AutoNameEnum):
    AUDIO = auto()
    TEXT = auto()
