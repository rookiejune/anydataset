from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Self


def _select[KeyT, ValueT](
    values: Mapping[KeyT, ValueT],
    keys: Iterable[KeyT],
) -> Mapping[KeyT, ValueT]:
    return {key: values[key] for key in keys}


@dataclass(frozen=True)
class _Requirement[ViewT, KeyT, OptKeyT]:
    views: frozenset[ViewT] = frozenset()
    required: frozenset[KeyT] = frozenset()
    optional: frozenset[OptKeyT] = frozenset()

    @classmethod
    def from_iter(
        cls,
        views: Iterable[ViewT],
        required: Iterable[KeyT],
        optional: Iterable[OptKeyT],
    ):
        return cls(
            views=frozenset(views),
            required=frozenset(required),
            optional=frozenset(optional),
        )


@dataclass(frozen=True)
class _Item[ViewT, KeyT, OptT]:
    views: Mapping[ViewT, Any] = field(default_factory=dict)
    required: Mapping[KeyT, Any] = field(default_factory=dict)
    optional: Mapping[OptT, Any] = field(default_factory=dict)

    def select_by(
        self,
        requirement,
    ) -> Self:
        return type(self)(
            views=_select(self.views, requirement.views),
            required=_select(self.required, requirement.required),
            optional=_select(self.optional, requirement.optional),
        )


class AudioKey(StrEnum):
    SAMPLE_RATE = auto()


class AudioOptKey(StrEnum):
    DURATION = auto()
    LABEL = auto()
    LABELS = auto()


class AudioView(StrEnum):
    WAVEFORM = auto()
    FILE = auto()
    LONGCAT = auto()
    DAC = auto()


@dataclass(frozen=True)
class AudioItem(_Item[AudioView, AudioKey, AudioOptKey]):
    pass


class ImageKey(StrEnum): ...


class ImageOptKey(StrEnum):
    LABEL = auto()


class ImageView(StrEnum):
    PIXEL = auto()


@dataclass(frozen=True)
class ImageItem(_Item[ImageView, ImageKey, ImageOptKey]):
    pass


class TextKey(StrEnum): ...


class TextOptKey(StrEnum):
    LANG = auto()


class TextView(StrEnum):
    TEXT = auto()


@dataclass(frozen=True)
class TextItem(_Item[TextView, TextKey, TextOptKey]):
    pass


@dataclass(frozen=True)
class AudioReq(
    _Requirement[
        AudioView,
        AudioKey,
        AudioOptKey,
    ]
): ...


@dataclass(frozen=True)
class ImageReq(
    _Requirement[
        ImageView,
        ImageKey,
        ImageOptKey,
    ]
): ...


@dataclass(frozen=True)
class TextReq(
    _Requirement[
        TextView,
        TextKey,
        TextOptKey,
    ]
): ...


type View = AudioView | ImageView | TextView
type Key = AudioKey | ImageKey | TextKey
type OptKey = AudioOptKey | ImageOptKey | TextOptKey
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
type Ref = Reference

type Schema = Mapping[Reference, Requirement]
type Sample = Mapping[Reference, Item]
