from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union, cast

from typing_extensions import TypedDict

from .._compat import Self, StrEnum
from .._immutable import Immutable
from .language import Lang

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
ViewT = TypeVar("ViewT")
MetaT = TypeVar("MetaT")

if TYPE_CHECKING:
    from torch import Tensor


def _select(
    values: Mapping[KeyT, ValueT],
    keys: Iterable[KeyT],
) -> Mapping[KeyT, ValueT]:
    return {key: values[key] for key in keys}


def _restore_state(value: Immutable, state: object, kind: str) -> None:
    if not isinstance(state, dict):
        raise TypeError(f"invalid {kind} pickle state.")
    if any(not isinstance(name, str) for name in state):
        raise TypeError(f"invalid {kind} pickle state.")
    if not {"views", "meta"}.issubset(state):
        raise ValueError(f"{kind} pickle state is missing required fields.")
    for name, field_value in state.items():
        if name != "_immutable_sealed":
            setattr(value, name, field_value)


@dataclass(unsafe_hash=True)
class _Requirement(Immutable, Generic[ViewT, MetaT]):
    views: frozenset[ViewT] = frozenset()
    meta: frozenset[MetaT] = frozenset()

    def __post_init__(self) -> None:
        self.seal()

    def __setstate__(self, state: object) -> None:
        _restore_state(self, state, "requirement")
        self.__post_init__()

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


@dataclass
class _Item(Immutable, Generic[ViewT, MetaT]):
    views: Mapping[ViewT, Any] = field(default_factory=dict)
    meta: Mapping[MetaT, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.views = _mapping("item.views", self.views)
        self.meta = _mapping("item.meta", self.meta)
        self.seal()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(vars(self))
        state["views"] = dict(self.views)
        state["meta"] = dict(self.meta)
        return state

    def __setstate__(self, state: object) -> None:
        _restore_state(self, state, "item")
        self.__post_init__()

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
    BICODEC = auto()
    LONGCAT = auto()
    DAC = auto()
    STABLE = auto()
    UNICODEC = auto()
    SPEAKERS = auto()
    SPEAKER_LENGTHS = auto()


class SemanticAcousticView(TypedDict):
    """Structured semantic and acoustic unit tensors for one audio sample."""

    semantic: Tensor
    acoustic: Tensor


@dataclass
class AudioItem(_Item[AudioView, AudioMeta]):
    def __post_init__(self) -> None:
        self.views = _enum_mapping("AudioItem.views", self.views, AudioView)
        self.meta = _enum_mapping("AudioItem.meta", self.meta, AudioMeta)
        self.seal()


class ImageMeta(StrEnum):
    LABEL = auto()


class ImageView(StrEnum):
    PIXEL = auto()


@dataclass
class ImageItem(_Item[ImageView, ImageMeta]):
    def __post_init__(self) -> None:
        self.views = _enum_mapping("ImageItem.views", self.views, ImageView)
        self.meta = _enum_mapping("ImageItem.meta", self.meta, ImageMeta)
        self.seal()


class TextMeta(StrEnum):
    LANG = auto()
    SOURCE_INDEX = auto()


class TextView(StrEnum):
    TEXT = auto()
    SPEAKERS = auto()


@dataclass
class TextItem(_Item[TextView, TextMeta]):
    def __post_init__(self) -> None:
        self.views = _enum_mapping("TextItem.views", self.views, TextView)
        self.meta = _text_meta_mapping("TextItem.meta", self.meta)
        self.seal()


@dataclass(unsafe_hash=True)
class AudioReq(
    _Requirement[
        AudioView,
        AudioMeta,
    ]
):
    def __post_init__(self) -> None:
        self.views = _enum_keys("AudioReq.views", self.views, AudioView)
        self.meta = _enum_keys("AudioReq.meta", self.meta, AudioMeta)
        self.seal()


@dataclass(unsafe_hash=True)
class ImageReq(
    _Requirement[
        ImageView,
        ImageMeta,
    ]
):
    def __post_init__(self) -> None:
        self.views = _enum_keys("ImageReq.views", self.views, ImageView)
        self.meta = _enum_keys("ImageReq.meta", self.meta, ImageMeta)
        self.seal()


@dataclass(unsafe_hash=True)
class TextReq(
    _Requirement[
        TextView,
        TextMeta,
    ]
):
    def __post_init__(self) -> None:
        self.views = _enum_keys("TextReq.views", self.views, TextView)
        self.meta = _enum_keys("TextReq.meta", self.meta, TextMeta)
        self.seal()


def _enum_mapping(name: str, value: object, key_type: type[KeyT]) -> Mapping[KeyT, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    output = dict(value)
    if any(not isinstance(key, key_type) for key in output):
        raise TypeError(f"{name} keys must be {key_type.__name__} values.")
    return MappingProxyType(output)


def _mapping(name: str, value: object) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return MappingProxyType(dict(value))


def _text_meta_mapping(name: str, value: object) -> Mapping[TextMeta, Any]:
    output = _enum_mapping(name, value, TextMeta)
    lang = output.get(TextMeta.LANG)
    if lang is not None and not _lang_value(lang):
        raise TypeError("TextMeta.LANG must be a Lang value or sequence of Lang values.")
    source_index = output.get(TextMeta.SOURCE_INDEX)
    if source_index is not None and not _source_index_value(source_index):
        raise TypeError("TextMeta.SOURCE_INDEX must be an integer or sequence of integers.")
    return output


def _lang_value(value: object) -> bool:
    if isinstance(value, Lang):
        return True
    if isinstance(value, str):
        return False
    if not isinstance(value, Sequence):
        return False
    return all(isinstance(item, Lang) for item in value)


def _source_index_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if not isinstance(value, Sequence):
        return False
    return all(not isinstance(item, bool) and isinstance(item, int) for item in value)

def _enum_keys(name: str, value: object, key_type: type[KeyT]) -> frozenset[KeyT]:
    if not isinstance(value, Iterable):
        raise TypeError(f"{name} must be an iterable of {key_type.__name__} values.")
    try:
        output = frozenset(cast(Iterable[KeyT], value))
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

    def item_type(self) -> type[Item]:
        if self is Modality.AUDIO:
            return AudioItem
        if self is Modality.IMAGE:
            return ImageItem
        if self is Modality.TEXT:
            return TextItem
        raise ValueError(f"Unsupported modality: {self!r}.")

    def view_type(self) -> type[View]:
        if self is Modality.AUDIO:
            return AudioView
        if self is Modality.IMAGE:
            return ImageView
        if self is Modality.TEXT:
            return TextView
        raise ValueError(f"Unsupported modality: {self!r}.")

    def meta_type(self) -> type[Meta]:
        if self is Modality.AUDIO:
            return AudioMeta
        if self is Modality.IMAGE:
            return ImageMeta
        if self is Modality.TEXT:
            return TextMeta
        raise ValueError(f"Unsupported modality: {self!r}.")

    def requirement_type(self) -> type[Requirement]:
        if self is Modality.AUDIO:
            return AudioReq
        if self is Modality.IMAGE:
            return ImageReq
        if self is Modality.TEXT:
            return TextReq
        raise ValueError(f"Unsupported modality: {self!r}.")

    def item(
        self,
        *,
        views: Mapping[Any, Any],
        meta: Mapping[Any, Any],
    ) -> Item:
        return self.item_type()(views=views, meta=meta)

    def view(self, value: str) -> View:
        return self.view_type()(value)

    def requirement(
        self,
        views: Iterable[Any],
        meta: Iterable[Any],
    ) -> Requirement:
        return self.requirement_type().from_iter(views, meta)


Reference = tuple[Role, Modality]
ItemTransform = Callable[[Item], Item]
Transforms = Mapping[Reference, ItemTransform]
Schema = Mapping[Reference, Requirement]
Sample = Mapping[Reference, Item]
