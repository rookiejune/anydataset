from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from . import item
from .item import (
    AudioItem,
    AudioMeta,
    AudioReq,
    AudioView,
    ImageItem,
    ImageMeta,
    ImageReq,
    ImageView,
    Item,
    ItemTransform,
    Modality,
    Meta,
    Reference,
    Requirement,
    Role,
    Sample,
    Schema,
    TextItem,
    TextMeta,
    TextReq,
    TextView,
    Transforms,
    View,
)
from .preset import Preset

if TYPE_CHECKING:
    from ..dataset.collate import Batch


class Source(StrEnum):
    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,
        count: int,
        last_values: list[str],
    ) -> str:
        return name.lower().replace("_", "-")

    HF = auto()
    HF_DISK = auto()
    STORE = auto()


type SourceKey = Source | str


@dataclass(frozen=True)
class Spec:
    source: SourceKey
    path: str
    split: str | None = None
    version: str | None = None
    load_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, Source | str):
            raise TypeError("Spec.source must be a Source or string source key.")
        object.__setattr__(
            self, "load_options", MappingProxyType(dict(self.load_options))
        )

    @property
    def id(self) -> str:
        return _stable_hash(_identity_payload(self))

    @property
    def cache_relpath(self) -> Path:
        return Path(self.id)

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        payload = _identity_payload(self)
        return {"id": self.id, **payload}

    def __reduce__(self):
        return (
            type(self),
            (
                self.source,
                self.path,
                self.split,
                self.version,
                dict(self.load_options),
            ),
        )


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
                        meta=frozenset({ImageMeta.LABEL}),
                    )
                }
            case Task.AUDIO_CODEC:
                return {
                    (Role.DEFAULT, Modality.AUDIO): AudioReq(
                        views=frozenset({AudioView.WAVEFORM}),
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
        "source": source_key(spec.source),
        "path": spec.path,
        "split": spec.split,
        "version": spec.version,
        "load_options": _normalize(spec.load_options),
    }


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def source_key(source: SourceKey) -> str:
    if isinstance(source, Source):
        return source.value
    if not isinstance(source, str):
        raise TypeError("source key must be a Source or string source key.")
    if not source:
        raise ValueError("source key must not be empty.")
    return source


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
    "AudioMeta",
    "AudioReq",
    "AudioView",
    "ImageItem",
    "ImageMeta",
    "ImageReq",
    "ImageView",
    "Item",
    "ItemTransform",
    "Modality",
    "Meta",
    "Preset",
    "Reference",
    "Requirement",
    "Role",
    "Sample",
    "Schema",
    "Source",
    "SourceKey",
    "Spec",
    "Task",
    "TextItem",
    "TextMeta",
    "TextReq",
    "TextView",
    "Transforms",
    "View",
    "item",
]
