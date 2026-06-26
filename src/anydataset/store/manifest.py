from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..types.item import (
    AudioView,
    ImageView,
    Modality,
    Role,
    TextView,
    View,
)


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    sample_count: int
    split: str | None = None


type SampleItem = tuple[tuple[Role, Modality], Mapping[str, Any]]


@dataclass(frozen=True)
class SampleManifestEntry:
    sample_id: str
    sample_index: int
    items: tuple[SampleItem, ...] = ()

    def item(self, ref: tuple[Role, Modality]) -> SampleItem | None:
        for entry in self.items:
            if entry[0] == ref:
                return entry
        return None


@dataclass(frozen=True)
class ViewManifestEntry:
    role: Role
    modality: Modality
    view: View
    sample_id: str
    shard: str
    key: str


def view_from_dict(data: Mapping[str, Any]) -> tuple[Role, Modality, View]:
    modality = Modality(data["modality"])
    return (
        Role(data["role"]),
        modality,
        _view_from_str(modality, data["view"]),
    )


def _view_from_str(modality: Modality, value: str) -> View:
    match modality:
        case Modality.AUDIO:
            return AudioView(value)
        case Modality.IMAGE:
            return ImageView(value)
        case Modality.TEXT:
            return TextView(value)
    raise ValueError(f"Unsupported modality: {modality!r}.")


def string_key_dict(values: Mapping[Any, Any]) -> dict[str, Any]:
    return {
        key.value if isinstance(key, StrEnum) else str(key): value
        for key, value in values.items()
    }
