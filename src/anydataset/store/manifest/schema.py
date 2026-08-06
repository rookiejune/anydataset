from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from ..._compat import StrEnum
from ..._immutable import Immutable
from ...types.item import (
    Modality,
    Role,
    View,
)

STORE_SCHEMA_VERSION = 3
LEGACY_STORE_SCHEMA_VERSION = 2

_BASE_DATASET_MANIFEST_FIELDS = frozenset(
    {"dataset_id", "sample_count", "schema_version", "split"}
)
_PROVENANCE_FIELDS = frozenset({"input_id", "provider_id", "output_id"})


@dataclass(init=False)
class DatasetManifest(Immutable):
    dataset_id: str
    sample_count: int
    schema_version: int
    split: str | None
    provenance: Mapping[str, str]

    def __init__(
        self,
        dataset_id: str,
        sample_count: int,
        schema_version: int,
        split: str | None = None,
        provenance: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(dataset_id, str):
            raise ValueError("Store dataset_id must be a string.")
        if type(sample_count) is not int or sample_count < 0:
            raise ValueError("Store sample_count must be a non-negative integer.")
        if type(schema_version) is not int or schema_version not in {
            LEGACY_STORE_SCHEMA_VERSION,
            STORE_SCHEMA_VERSION,
        }:
            raise ValueError(_unsupported_schema_version(schema_version))
        if split is not None and not isinstance(split, str):
            raise ValueError("Store split must be a string or None.")
        normalized = normalize_provenance(provenance)
        if schema_version == LEGACY_STORE_SCHEMA_VERSION and normalized:
            raise ValueError("Legacy store manifests cannot contain provenance.")

        self.dataset_id = dataset_id
        self.sample_count = sample_count
        self.schema_version = schema_version
        self.split = split
        self.provenance = MappingProxyType(normalized)
        self.seal()

    def __reduce__(self):
        return (
            DatasetManifest,
            (
                self.dataset_id,
                self.sample_count,
                self.schema_version,
                self.split,
                dict(self.provenance),
            ),
        )


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
    return modality.view(value)


def string_key_dict(values: Mapping[Any, Any]) -> dict[str, Any]:
    return {
        key.value if isinstance(key, StrEnum) else str(key): value
        for key, value in values.items()
    }


def dataset_manifest_dict(manifest: DatasetManifest) -> dict[str, Any]:
    data = {
        "dataset_id": manifest.dataset_id,
        "sample_count": manifest.sample_count,
        "schema_version": manifest.schema_version,
        "split": manifest.split,
    }
    if manifest.schema_version == STORE_SCHEMA_VERSION:
        data["provenance"] = normalize_provenance(manifest.provenance)
    return data


def normalize_provenance(value: Mapping[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Store provenance must be a mapping.")
    unsupported = set(value) - _PROVENANCE_FIELDS
    if unsupported:
        raise ValueError(
            f"Store provenance has unsupported field {min(unsupported)!r}."
        )
    output: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Store provenance keys must be strings.")
        if not isinstance(item, str) or not item:
            raise ValueError(f"Store provenance {key!r} must be a non-empty string.")
        output[key] = item
    return output


def dataset_manifest_from_dict(value: object) -> DatasetManifest:
    if not isinstance(value, Mapping):
        raise ValueError("Store dataset manifest must be a JSON object.")
    data = cast(Mapping[str, Any], value)
    version = data.get("schema_version")
    if type(version) is not int or version not in {
        LEGACY_STORE_SCHEMA_VERSION,
        STORE_SCHEMA_VERSION,
    }:
        raise ValueError(_unsupported_schema_version(version))

    required = _BASE_DATASET_MANIFEST_FIELDS
    if version == STORE_SCHEMA_VERSION:
        required = required | {"provenance"}
    fields = frozenset(data)
    missing = required - fields
    if missing:
        raise ValueError(
            f"Store dataset manifest is missing field {sorted(missing)[0]!r}."
        )
    unsupported = fields - required
    if unsupported:
        raise ValueError(
            f"Store dataset manifest has unsupported field {sorted(unsupported)[0]!r}."
        )

    return DatasetManifest(
        dataset_id=data["dataset_id"],
        sample_count=data["sample_count"],
        schema_version=version,
        split=data["split"],
        provenance=data["provenance"] if version == STORE_SCHEMA_VERSION else {},
    )


def _unsupported_schema_version(value: object) -> str:
    migration = (
        " Use anydataset.store.migrate_store(source, output) for a schema-v1 store."
        if value is None or (type(value) is int and value == 1)
        else ""
    )
    return (
        "Unsupported store schema_version: "
        f"{value!r}; expected {LEGACY_STORE_SCHEMA_VERSION} or "
        f"{STORE_SCHEMA_VERSION}.{migration}"
    )
