from __future__ import annotations

from enum import auto

from ...enums import AutoNameEnum


class AudioKey(AutoNameEnum):
    VIEWS = auto()
    SAMPLE_RATE = auto()


class AudioOptKey(AutoNameEnum):
    DURATION = auto()
    LABEL = auto()
    LABELS = auto()


class AudioView(AutoNameEnum):
    WAVEFORM = auto()
    FILE = auto()
    LONGCAT = auto()
    DAC = auto()
