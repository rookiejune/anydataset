from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Self


def _select[KeyT, ValueT](
    values: Mapping[KeyT, ValueT],
    keys: Iterable[KeyT],
) -> Mapping[KeyT, ValueT]:
    return {key: values[key] for key in keys}


@dataclass(frozen=True)
class _Requirement[ViewT, MetaT]:
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
class _Item[ViewT, MetaT]:
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
    # DURATION = auto()  # derived from waveform.size(-1) / sample_rate
    LABEL = auto()
    LABELS = auto()
    SPEAKER_ID = auto()


class AudioView(StrEnum):
    WAVEFORM = auto()
    FILE = auto()
    LONGCAT = auto()
    DAC = auto()


@dataclass(frozen=True)
class AudioItem(_Item[AudioView, AudioMeta]):
    pass


class ImageMeta(StrEnum):
    LABEL = auto()


class ImageView(StrEnum):
    PIXEL = auto()


@dataclass(frozen=True)
class ImageItem(_Item[ImageView, ImageMeta]):
    pass


class TextMeta(StrEnum):
    LANG = auto()


class TextView(StrEnum):
    TEXT = auto()


@dataclass(frozen=True)
class TextItem(_Item[TextView, TextMeta]):
    pass


@dataclass(frozen=True)
class AudioReq(
    _Requirement[
        AudioView,
        AudioMeta,
    ]
): ...


@dataclass(frozen=True)
class ImageReq(
    _Requirement[
        ImageView,
        ImageMeta,
    ]
): ...


@dataclass(frozen=True)
class TextReq(
    _Requirement[
        TextView,
        TextMeta,
    ]
): ...


type View = AudioView | ImageView | TextView
type Meta = AudioMeta | ImageMeta | TextMeta
type Item = AudioItem | ImageItem | TextItem
type Requirement = AudioReq | ImageReq | TextReq


class Role(StrEnum):
    DEFAULT = auto()
    SOURCE = auto()
    TARGET = auto()


class Modality(StrEnum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()


type Reference = tuple[Role, Modality]
type ItemTransform = Callable[[Item], Item]
type Transforms = Mapping[Reference, ItemTransform]
type Schema = Mapping[Reference, Requirement]
type Sample = Mapping[Reference, Item]
