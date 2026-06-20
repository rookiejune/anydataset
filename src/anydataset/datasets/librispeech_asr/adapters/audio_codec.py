from __future__ import annotations

from anydataset.datasets.local_files.adapters.audio_codec import AudioCodecSampleAdapter


class LibriSpeechASRAudioCodecAdapter(AudioCodecSampleAdapter):
    def __init__(self, text_key: str = "text"):
        super().__init__(
            audio_key="audio",
            text_key=text_key,
        )
