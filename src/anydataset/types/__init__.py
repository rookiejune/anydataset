from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Union

from . import item
from .._compat import StrEnum
from .._immutable import Immutable
from .._source_identity import (
    physical_load_options as _source_physical_load_options,
)
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
    SemanticAcousticView,
    TextItem,
    TextMeta,
    TextReq,
    TextView,
    Transforms,
    View,
)
from .language import Lang, remap_lang
from .preset import Preset


class Source(StrEnum):
    HF = "hf"
    HF_DISK = "hf-disk"
    HF_FILES = "hf-files"
    STORE = "store"


SourceKey = Union[Source, str]


_EMPTY_LOAD_OPTIONS: Mapping[str, Any] = MappingProxyType({})


@dataclass(init=False)
class Spec(Immutable):
    source: SourceKey
    path: str
    split: str | None
    version: str | None
    load_options: Mapping[str, Any]
    _identity_load_options: Mapping[str, Any] = field(
        init=False,
        repr=False,
    )
    _id: str = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        source: SourceKey,
        path: str,
        split: str | None = None,
        version: str | None = None,
        load_options: Mapping[str, Any] = _EMPTY_LOAD_OPTIONS,
        *,
        _identity_load_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(source, (Source, str)):
            raise TypeError("Spec.source must be a Source or string source key.")
        if not isinstance(path, str):
            raise TypeError("Spec.path must be a string.")
        if not path:
            raise ValueError("Spec.path must not be empty.")
        for name, value in (("split", split), ("version", version)):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"Spec.{name} must be a string or None.")
            if value == "":
                raise ValueError(f"Spec.{name} must not be empty.")
        if not isinstance(load_options, Mapping):
            raise TypeError("Spec.load_options must be a mapping.")

        self.source = source
        self.path = path
        self.split = split
        self.version = version
        self.load_options = _freeze_mapping(load_options)
        physical_load_options = (
            _source_physical_load_options(
                source_key(self.source),
                self.load_options,
            )
            if _identity_load_options is None
            else _identity_load_options
        )
        self._identity_load_options = _freeze_mapping(physical_load_options)
        identity = _identity_payload(self)
        self._id = _stable_hash(identity)
        self.seal()

    @property
    def id(self) -> str:
        return self._id

    @property
    def cache_relpath(self) -> Path:
        return Path(self.id)

    def __hash__(self) -> int:
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, **_identity_payload(self)}

    def __reduce__(self):
        return (
            _restore_spec,
            (
                self.source,
                self.path,
                self.split,
                self.version,
                _thaw(self.load_options),
                _thaw(self._identity_load_options),
            ),
        )


def _identity_payload(spec: Spec) -> dict[str, Any]:
    return {
        "source": source_key(spec.source),
        "path": spec.path,
        "split": spec.split,
        "version": spec.version,
        "load_options": _payload_value(spec._identity_load_options),
    }


def _restore_spec(
    source: SourceKey,
    path: str,
    split: str | None,
    version: str | None,
    load_options: Mapping[str, Any],
    identity_load_options: Mapping[str, Any],
) -> Spec:
    return Spec(
        source,
        path,
        split,
        version,
        load_options,
        _identity_load_options=identity_load_options,
    )


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


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw(child) for child in value)
    if isinstance(value, frozenset):
        return frozenset(_thaw(child) for child in value)
    return value


def _freeze_mapping(value: Mapping[Any, Any]) -> MappingProxyType[str, Any]:
    output: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise TypeError("Spec load option keys must be strings.")
        output[key] = _freeze(child)
    return MappingProxyType(output)


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
    "Lang",
    "Modality",
    "Meta",
    "Preset",
    "Reference",
    "Requirement",
    "Role",
    "Sample",
    "Schema",
    "SemanticAcousticView",
    "Source",
    "SourceKey",
    "Spec",
    "TextItem",
    "TextMeta",
    "TextReq",
    "TextView",
    "Transforms",
    "View",
    "item",
    "remap_lang",
]
