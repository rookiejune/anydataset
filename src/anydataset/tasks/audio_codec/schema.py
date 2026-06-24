from __future__ import annotations

from ...modalities import ModalityKey
from ...modalities.audio import AudioKey, AudioOptKey
from ...modalities.text import TextKey, TextOptKey


class AudioCodecKey:
    TEXT = ModalityKey.TEXT
    LANG = TextOptKey.LANG


type AudioCodecSampleKey = ModalityKey | AudioKey | AudioOptKey | TextKey | TextOptKey
