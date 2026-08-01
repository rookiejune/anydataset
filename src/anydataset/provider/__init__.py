from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .codec import CodecProvider
    from .longcat import LongCatProvider
    from .moss_tts import MossTTSProvider
    from .qwen_tts import QwenTTSProvider
    from .whisper import WhisperASRProvider

__all__ = [
    "CodecProvider",
    "LongCatProvider",
    "MossTTSProvider",
    "QwenTTSProvider",
    "WhisperASRProvider",
]

_PROVIDER_MODULES = {
    "CodecProvider": ".codec",
    "LongCatProvider": ".longcat",
    "MossTTSProvider": ".moss_tts",
    "QwenTTSProvider": ".qwen_tts",
    "WhisperASRProvider": ".whisper",
}


def __getattr__(name: str) -> Any:
    module_name = _PROVIDER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
