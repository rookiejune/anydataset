from .adapters.audio_codec import FleursAudioCodecAdapter
from .dataset import fleurs_spec, register_task_adapters

__all__ = [
    "FleursAudioCodecAdapter",
    "fleurs_spec",
    "register_task_adapters",
]
