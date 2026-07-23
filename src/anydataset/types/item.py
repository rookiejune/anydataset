from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import auto
from typing import Any, Generic, TypeVar, Union

from .._compat import Self, StrEnum
from .language import Lang

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
ViewT = TypeVar("ViewT")
MetaT = TypeVar("MetaT")


def _select(
    values: Mapping[KeyT, ValueT],
    keys: Iterable[KeyT],
) -> Mapping[KeyT, ValueT]:
    return {key: values[key] for key in keys}


@dataclass(frozen=True)
class _Requirement(Generic[ViewT, MetaT]):
    views: frozenset[ViewT] = frozenset()
    meta: frozenset[MetaT] = frozenset()

    @classmethod
    def from_iter(
        cls,
        views: Iterable[ViewT],
        meta: Iterable[MetaT],
    ):
        return cls(
            views=frozenset(views),
            meta=frozenset(meta),
        )


@dataclass(frozen=True)
class _Item(Generic[ViewT, MetaT]):
    views: Mapping[ViewT, Any] = field(default_factory=dict)
    meta: Mapping[MetaT, Any] = field(default_factory=dict)

    def select_by(
        self,
        requirement,
    ) -> Self:
        return type(self)(
            views=_select(self.views, requirement.views),
            meta=_select(self.meta, requirement.meta),
        )


class AudioMeta(StrEnum):
    DURATION = auto()
    LABEL = auto()
    LABELS = auto()
    SPEAKER_ID = auto()


class AudioView(StrEnum):
    WAVEFORM = auto()
    FILE = auto()
    LONGCAT = auto()
    DAC = auto()
    STABLE = auto()
    UNICODEC = auto()
    SPEAKERS = auto()
    SPEAKER_LENGTHS = auto()


@dataclass(frozen=True)
class AudioItem(_Item[AudioView, AudioMeta]):
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "views",
            _enum_mapping("AudioItem.views", self.views, AudioView),
        )
        object.__setattr__(
            self,
            "meta",
            _enum_mapping("AudioItem.meta", self.meta, AudioMeta),
        )


class ImageMeta(StrEnum):
    LABEL = auto()


class ImageView(StrEnum):
    PIXEL = auto()


@dataclass(frozen=True)
class ImageItem(_Item[ImageView, ImageMeta]):
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "views",
            _enum_mapping("ImageItem.views", self.views, ImageView),
        )
        object.__setattr__(
            self,
            "meta",
            _enum_mapping("ImageItem.meta", self.meta, ImageMeta),
        )


class TextMeta(StrEnum):
    LANG = auto()
    SOURCE_INDEX = auto()


class TextView(StrEnum):
    TEXT = auto()
    SPEAKERS = auto()


@dataclass(frozen=True)
class TextItem(_Item[TextView, TextMeta]):
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "views",
            _enum_mapping("TextItem.views", self.views, TextView),
        )
        object.__setattr__(
            self,
            "meta",
            _text_meta_mapping("TextItem.meta", self.meta),
        )


@dataclass(frozen=True)
class AudioReq(
    _Requirement[
        AudioView,
        AudioMeta,
    ]
):
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "views",
            _enum_keys("AudioReq.views", self.views, AudioView),
        )
        object.__setattr__(
            self,
            "meta",
            _enum_keys("AudioReq.meta", self.meta, AudioMeta),
        )


@dataclass(frozen=True)
class ImageReq(
    _Requirement[
        ImageView,
        ImageMeta,
    ]
):
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "views",
            _enum_keys("ImageReq.views", self.views, ImageView),
        )
        object.__setattr__(
            self,
            "meta",
            _enum_keys("ImageReq.meta", self.meta, ImageMeta),
        )


@dataclass(frozen=True)
class TextReq(
    _Requirement[
        TextView,
        TextMeta,
    ]
):
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "views",
            _enum_keys("TextReq.views", self.views, TextView),
        )
        object.__setattr__(
            self,
            "meta",
            _enum_keys("TextReq.meta", self.meta, TextMeta),
        )


def _enum_mapping(name: str, value: object, key_type: type[KeyT]) -> Mapping[KeyT, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    output = dict(value)
    if any(not isinstance(key, key_type) for key in output):
        raise TypeError(f"{name} keys must be {key_type.__name__} values.")
    return output


def _text_meta_mapping(name: str, value: object) -> Mapping[TextMeta, Any]:
    output = _enum_mapping(name, value, TextMeta)
    lang = output.get(TextMeta.LANG)
    if lang is not None and not isinstance(lang, Lang):
        raise TypeError("TextMeta.LANG must be a Lang value.")
    source_index = output.get(TextMeta.SOURCE_INDEX)
    if source_index is not None and not _source_index_value(source_index):
        raise TypeError("TextMeta.SOURCE_INDEX must be an integer or sequence of integers.")
    return output


def _source_index_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if not isinstance(value, Sequence):
        return False
    return all(not isinstance(item, bool) and isinstance(item, int) for item in value)

def _enum_keys(name: str, value: object, key_type: type[KeyT]) -> frozenset[KeyT]:
    try:
        output = frozenset(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of {key_type.__name__} values.") from exc
    if any(not isinstance(key, key_type) for key in output):
        raise TypeError(f"{name} must contain {key_type.__name__} values.")
    return output


View = Union[AudioView, ImageView, TextView]
Meta = Union[AudioMeta, ImageMeta, TextMeta]
Item = Union[AudioItem, ImageItem, TextItem]
Requirement = Union[AudioReq, ImageReq, TextReq]


class Role(StrEnum):
    DEFAULT = auto()
    SOURCE = auto()
    TARGET = auto()


class Modality(StrEnum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()


Reference = tuple[Role, Modality]
ItemTransform = Callable[[Item], Item]
Transforms = Mapping[Reference, ItemTransform]
Schema = Mapping[Reference, Requirement]
Sample = Mapping[Reference, Item]
