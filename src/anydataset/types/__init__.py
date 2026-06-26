from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from . import item
from .item import (
    AudioItem,
    AudioKey,
    AudioOptKey,
    AudioReq,
    AudioView,
    ImageItem,
    ImageKey,
    ImageOptKey,
    ImageReq,
    ImageView,
    Item,
    Key,
    Modality,
    OptKey,
    Ref,
    Reference,
    Requirement,
    Role,
    Sample,
    Schema,
    TextItem,
    TextKey,
    TextOptKey,
    TextReq,
    TextView,
    View,
)
from .preset import Preset

if TYPE_CHECKING:
    from ..dataset.collate import Batch


class Source(StrEnum):
    HF = "huggingface"
    HF_DISK = "huggingface_disk"
    LOCAL = "local_files"
    UNIFIED = "unified"


@dataclass(frozen=True)
class Spec:
    source: Source
    path: str
    split: str | None = None
    version: str | None = None
    load_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, Source):
            object.__setattr__(self, "source", Source(self.source))
        object.__setattr__(
            self, "load_options", MappingProxyType(dict(self.load_options))
        )

    @property
    def id(self) -> str:
        return _stable_hash(_identity_payload(self))

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        payload = _identity_payload(self)
        return {"id": self.id, **payload}


class Task(StrEnum):
    IMAGE_CLASSIFICATION = auto()
    AUDIO_CODEC = auto()
    MACHINE_TRANSLATION = auto()

    def schema(self) -> Schema:
        match self:
            case Task.IMAGE_CLASSIFICATION:
                return {
                    (Role.DEFAULT, Modality.IMAGE): ImageReq(
                        views=frozenset({ImageView.PIXEL}),
                        optional=frozenset({ImageOptKey.LABEL}),
                    )
                }
            case Task.AUDIO_CODEC:
                return {
                    (Role.DEFAULT, Modality.AUDIO): AudioReq(
                        views=frozenset({AudioView.WAVEFORM}),
                        required=frozenset({AudioKey.SAMPLE_RATE}),
                    )
                }
            case Task.MACHINE_TRANSLATION:
                req = TextReq(
                    views=frozenset({TextView.TEXT}),
                )
                return {
                    (Role.SOURCE, Modality.TEXT): req,
                    (Role.TARGET, Modality.TEXT): req,
                }

    def collate_fn(self) -> Callable[[Sequence[Sample]], Batch]:
        from ..dataset.collate import collate_fn

        return collate_fn(self.schema())


def _identity_payload(spec: Spec) -> dict[str, Any]:
    return {
        "source": spec.source.value,
        "path": spec.path,
        "split": spec.split,
        "version": spec.version,
        "load_options": _normalize(spec.load_options),
    }


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_normalize(child) for child in value]
    if isinstance(value, set | frozenset):
        return sorted(_normalize(child) for child in value)
    if isinstance(value, StrEnum):
        return value.value
    return value


__all__ = [
    "AudioItem",
    "AudioKey",
    "AudioOptKey",
    "AudioReq",
    "AudioView",
    "ImageItem",
    "ImageKey",
    "ImageOptKey",
    "ImageReq",
    "ImageView",
    "Item",
    "Key",
    "Modality",
    "OptKey",
    "Preset",
    "Ref",
    "Reference",
    "Requirement",
    "Role",
    "Sample",
    "Schema",
    "Source",
    "Spec",
    "Task",
    "TextItem",
    "TextKey",
    "TextOptKey",
    "TextReq",
    "TextView",
    "View",
    "item",
]
