from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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

STORE_SCHEMA_VERSION = 3
LEGACY_STORE_SCHEMA_VERSION = 2

_BASE_DATASET_MANIFEST_FIELDS = frozenset(
    {"dataset_id", "sample_count", "schema_version", "split"}
)
_PROVENANCE_FIELDS = frozenset({"input_id", "provider_id"})


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    sample_count: int
    schema_version: int
    split: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)


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


def dataset_manifest_dict(manifest: DatasetManifest) -> dict[str, Any]:
    provenance = normalize_provenance(manifest.provenance)
    return {
        "dataset_id": manifest.dataset_id,
        "sample_count": manifest.sample_count,
        "schema_version": manifest.schema_version,
        "split": manifest.split,
        "provenance": provenance,
    }


def normalize_provenance(value: Mapping[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Store provenance must be a mapping.")
    unsupported = set(value) - _PROVENANCE_FIELDS
    if unsupported:
        raise ValueError(
            "Store provenance has unsupported field "
            f"{min(unsupported)!r}."
        )
    output: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Store provenance keys must be strings.")
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"Store provenance {key!r} must be a non-empty string."
            )
        output[key] = item
    return output
