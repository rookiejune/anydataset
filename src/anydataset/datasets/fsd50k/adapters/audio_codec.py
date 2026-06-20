from __future__ import annotations

from anydataset.datasets.local_files.adapters.audio_codec import AudioCodecSampleAdapter


class FSD50KAudioCodecAdapter(AudioCodecSampleAdapter):
    def __init__(self):
        super().__init__(
            audio_key="audio",
        )
