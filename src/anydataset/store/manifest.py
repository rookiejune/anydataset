from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._compat import StrEnum
from ..types.item import (
    AudioView,
    ImageView,
    Modality,
    Role,
    TextView,
    View,
)

STORE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    sample_count: int
    schema_version: int
    split: str | None = None


SampleItem = tuple[tuple[Role, Modality], Mapping[str, Any]]


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
    sample_index: int
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
    if modality is Modality.AUDIO:
        return AudioView(value)
    if modality is Modality.IMAGE:
        return ImageView(value)
    if modality is Modality.TEXT:
        return TextView(value)
    raise ValueError(f"Unsupported modality: {modality!r}.")


def string_key_dict(values: Mapping[Any, Any]) -> dict[str, Any]:
    return {
        key.value if isinstance(key, StrEnum) else str(key): value
        for key, value in values.items()
    }
