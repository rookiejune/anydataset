from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..modalities import ModalityKey, ViewRef


STORE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ViewSelection:
    ref: ViewRef
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ViewRef):
            raise TypeError("ref must be a ViewRef.")
        _validate_segment("revision", self.revision)
        object.__setattr__(self, "revision", str(self.revision))

    def to_dict(self) -> dict[str, Any]:
        data = view_ref_to_dict(self.ref)
        data["revision"] = self.revision
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ViewSelection:
        return cls(
            ref=view_ref_from_dict(data),
            revision=_required_str(data, "revision"),
        )


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    sample_count: int
    split: str | None = None
    views: tuple[ViewSelection, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = STORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_non_empty_str("dataset_id", self.dataset_id)
        if self.split is not None:
            _validate_non_empty_str("split", self.split)
        sample_count = _int_value(self.sample_count, "sample_count")
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative.")
        schema_version = _int_value(self.schema_version, "schema_version")
        views = tuple(self.views)
        for view in views:
            if not isinstance(view, ViewSelection):
                raise TypeError("views entries must be ViewSelection instances.")
        object.__setattr__(self, "dataset_id", str(self.dataset_id))
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "views", views)
        _validate_mapping("config", self.config)
        _validate_mapping("provenance", self.provenance)
        if schema_version != STORE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {STORE_SCHEMA_VERSION}.")
        object.__setattr__(self, "schema_version", schema_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "sample_count": self.sample_count,
            "views": [view.to_dict() for view in self.views],
            "config": dict(self.config),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetManifest:
        version = _required_int(data, "schema_version")
        if version != STORE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported store schema version: {version}.")
        views = _optional_sequence(data, "views")
        return cls(
            schema_version=version,
            dataset_id=_required_str(data, "dataset_id"),
            split=_optional_str(data, "split"),
            sample_count=_required_int(data, "sample_count"),
            views=tuple(ViewSelection.from_dict(view) for view in views),
            config=_optional_mapping(data, "config"),
            provenance=_optional_mapping(data, "provenance"),
        )


@dataclass(frozen=True)
class SampleManifestEntry:
    sample_id: str
    dataset_name: str
    sample_index: int | None = None
    source: Mapping[str, Any] = field(default_factory=dict)
    modality: ModalityKey | None = None
    role: str | None = None
    duration: float | None = None
    sample_rate: int | None = None
    label: Any = None
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_str("sample_id", self.sample_id)
        _validate_non_empty_str("dataset_name", self.dataset_name)
        if self.sample_index is not None:
            sample_index = _int_value(self.sample_index, "sample_index")
            if sample_index < 0:
                raise ValueError("sample_index must be non-negative.")
            object.__setattr__(self, "sample_index", sample_index)
        if self.modality is not None and not isinstance(self.modality, ModalityKey):
            raise TypeError("modality must be a ModalityKey.")
        if self.role is not None:
            _validate_segment("role", self.role)
            if self.role == "views":
                raise ValueError("role cannot be 'views'.")
            object.__setattr__(self, "role", str(self.role))
        if self.duration is not None:
            duration = _number_value(self.duration, "duration")
            if duration < 0:
                raise ValueError("duration must be non-negative.")
            object.__setattr__(self, "duration", duration)
        if self.sample_rate is not None:
            sample_rate = _int_value(self.sample_rate, "sample_rate")
            if sample_rate <= 0:
                raise ValueError("sample_rate must be positive.")
            object.__setattr__(self, "sample_rate", sample_rate)
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("text must be a string.")
        _validate_mapping("source", self.source)
        _validate_mapping("metadata", self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "sample_index": self.sample_index,
            "source": dict(self.source),
            "modality": self.modality.value if self.modality is not None else None,
            "role": self.role,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "label": self.label,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SampleManifestEntry:
        modality = data.get("modality")
        return cls(
            sample_id=_required_str(data, "sample_id"),
            dataset_name=_required_str(data, "dataset_name"),
            sample_index=_optional_int(data, "sample_index"),
            source=_optional_mapping(data, "source"),
            modality=ModalityKey(modality) if modality is not None else None,
            role=_optional_str(data, "role"),
            duration=_optional_float(data, "duration"),
            sample_rate=_optional_int(data, "sample_rate"),
            label=data.get("label"),
            text=_optional_str(data, "text"),
            metadata=_optional_mapping(data, "metadata"),
        )


@dataclass(frozen=True)
class ViewManifestEntry:
    ref: ViewRef
    revision: str
    sample_id: str
    shard: str
    key: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    checksum: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ViewRef):
            raise TypeError("ref must be a ViewRef.")
        _validate_segment("revision", self.revision)
        _validate_non_empty_str("sample_id", self.sample_id)
        _validate_segment("shard", self.shard)
        _validate_non_empty_str("key", self.key)
        if self.shape is not None:
            shape = tuple(_int_value(dim, "shape") for dim in self.shape)
            for dim in shape:
                if dim < 0:
                    raise ValueError("shape dimensions must be non-negative.")
            object.__setattr__(self, "shape", shape)
        if self.dtype is not None:
            _validate_non_empty_str("dtype", self.dtype)
        if self.checksum is not None:
            _validate_non_empty_str("checksum", self.checksum)
        _validate_mapping("provenance", self.provenance)

    def to_dict(self) -> dict[str, Any]:
        data = view_ref_to_dict(self.ref)
        data.update(
            {
                "revision": self.revision,
                "sample_id": self.sample_id,
                "shard": self.shard,
                "key": self.key,
                "shape": list(self.shape) if self.shape is not None else None,
                "dtype": self.dtype,
                "checksum": self.checksum,
                "provenance": dict(self.provenance),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ViewManifestEntry:
        shape = data.get("shape")
        if shape is not None:
            shape = tuple(_int_value(dim, "shape") for dim in _sequence_value(shape, "shape"))
        return cls(
            ref=view_ref_from_dict(data),
            revision=_required_str(data, "revision"),
            sample_id=_required_str(data, "sample_id"),
            shard=_required_str(data, "shard"),
            key=_required_str(data, "key"),
            shape=shape,
            dtype=_optional_str(data, "dtype"),
            checksum=_optional_str(data, "checksum"),
            provenance=_optional_mapping(data, "provenance"),
        )


def view_ref_to_dict(ref: ViewRef) -> dict[str, str | None]:
    if not isinstance(ref, ViewRef):
        raise TypeError("ref must be a ViewRef.")
    return {
        "modality": ref.modality.value,
        "role": ref.role,
        "view_key": ref.view_key,
    }


def view_ref_from_dict(data: Mapping[str, Any]) -> ViewRef:
    role = _optional_str(data, "role")
    return ViewRef(
        modality=ModalityKey(_required_str(data, "modality")),
        view_key=_required_str(data, "view_key"),
        role=role,
    )


def _validate_mapping(name: str, value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")


def _validate_non_empty_str(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value == "":
        raise ValueError(f"{name} must be non-empty.")


def _validate_segment(name: str, value: str) -> None:
    _validate_non_empty_str(name, value)
    if value in {".", ".."}:
        raise ValueError(f"{name} must be a non-empty path segment.")
    if "/" in value:
        raise ValueError(f"{name} cannot contain '/'.")


def _required_str(data: Mapping[str, Any], key: str) -> str:
    if key not in data:
        raise KeyError(key)
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string.")
    return value


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string.")
    return value


def _required_int(data: Mapping[str, Any], key: str) -> int:
    if key not in data:
        raise KeyError(key)
    return _int_value(data[key], key)


def _optional_int(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    return _int_value(value, key)


def _int_value(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    return value


def _optional_float(data: Mapping[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return _number_value(value, key)


def _number_value(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number.")
    return float(value)


def _optional_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping.")
    return dict(value)


def _optional_sequence(data: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    value = data.get(key, ())
    return tuple(_mapping_value(item, key) for item in _sequence_value(value, key))


def _mapping_value(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} entries must be mappings.")
    return value


def _sequence_value(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{name} must be a sequence.")
    return tuple(value)
