from __future__ import annotations

from enum import auto

from ...enums import AutoNameEnum


class TextKey(AutoNameEnum):
    CONTENT = auto()


class TextOptKey(AutoNameEnum):
    LANG = auto()
