from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import auto
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, Union

from . import item
from .._compat import StrEnum
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


SourceKey = Union[Source, str]


@dataclass(frozen=True)
class Spec:
    source: SourceKey
    path: str
    split: str | None = None
    version: str | None = None
    load_options: Mapping[str, Any] = field(default_factory=dict)
    _identity: Mapping[str, Any] = field(init=False, repr=False, compare=False)
    _id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, (Source, str)):
            raise TypeError("Spec.source must be a Source or string source key.")
        frozen_options = _freeze_mapping(self.load_options)
        object.__setattr__(
            self, "load_options", frozen_options
        )
        identity = _identity_payload(self)
        object.__setattr__(self, "_identity", MappingProxyType(identity))
        object.__setattr__(self, "_id", _stable_hash(identity))

    @property
    def id(self) -> str:
        return self._id

    @property
    def cache_relpath(self) -> Path:
        return Path(self.id)

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, **self._identity}

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
        if self == Task.IMAGE_CLASSIFICATION:
            return {
                (Role.DEFAULT, Modality.IMAGE): ImageReq(
                    views=frozenset({ImageView.PIXEL}),
                    meta=frozenset({ImageMeta.LABEL}),
                )
            }
        if self == Task.AUDIO_CODEC:
            return {
                (Role.DEFAULT, Modality.AUDIO): AudioReq(
                    views=frozenset({AudioView.WAVEFORM}),
                )
            }
        if self == Task.MACHINE_TRANSLATION:
            req = TextReq(
                views=frozenset({TextView.TEXT}),
            )
            return {
                (Role.SOURCE, Modality.TEXT): req,
                (Role.TARGET, Modality.TEXT): req,
            }
        raise ValueError(f"Unsupported task: {self!r}.")

    def collate_fn(self) -> Callable[[Sequence[Sample]], Batch]:
        from ..dataset.collate import collate_fn

        return collate_fn(self.schema())


def _identity_payload(spec: Spec) -> dict[str, Any]:
    return {
        "source": source_key(spec.source),
        "path": spec.path,
        "split": spec.split,
        "version": spec.version,
        "load_options": _payload_value(spec.load_options),
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


def _payload_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _payload_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_payload_value(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_payload_value(child) for child in value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(child) for child in value)
    return value


def _freeze_mapping(value: Mapping[Any, Any]) -> MappingProxyType[str, Any]:
    return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})


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
