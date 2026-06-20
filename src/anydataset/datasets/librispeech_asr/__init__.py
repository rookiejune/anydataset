from .adapters.audio_codec import LibriSpeechASRAudioCodecAdapter
from .dataset import librispeech_asr_spec, register_task_adapters

__all__ = [
    "LibriSpeechASRAudioCodecAdapter",
    "librispeech_asr_spec",
    "register_task_adapters",
]
