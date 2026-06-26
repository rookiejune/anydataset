from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..types.item import AudioView, ImageView, Modality, Role, TextView, View

STORE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ViewRef:
    modality: Modality
    view_key: View
    role: Role = Role.DEFAULT

    def __post_init__(self) -> None:
        if not isinstance(self.modality, Modality):
            raise TypeError("modality must be a Modality.")
        if not isinstance(self.role, Role):
            raise TypeError("role must be a Role.")
        _validate_view_key(self.modality, self.view_key)
        _validate_segment("view_key", self.view_key.value)
        _validate_segment("role", self.role.value)
        if self.role.value == "views":
            raise ValueError("role cannot be 'views'.")

    @property
    def sample_ref(self) -> tuple[Role, Modality]:
        return self.role, self.modality

    def path_parts(self) -> tuple[str, ...]:
        if self.role is Role.DEFAULT:
            return self.modality.value, "views", self.view_key.value
        return self.modality.value, self.role.value, "views", self.view_key.value


@dataclass(frozen=True)
class ViewSelection:
    ref: ViewRef
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ViewRef):
            raise TypeError("ref must be a ViewRef.")
        _validate_segment("revision", self.revision)

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
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative.")
        if self.schema_version != STORE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {STORE_SCHEMA_VERSION}.")
        if not isinstance(self.views, tuple):
            raise TypeError("views must be a tuple.")
        for view in self.views:
            if not isinstance(view, ViewSelection):
                raise TypeError("views entries must be ViewSelection instances.")
        _validate_mapping("config", self.config)
        _validate_mapping("provenance", self.provenance)

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
class SampleItemEntry:
    ref: tuple[Role, Modality]
    required: Mapping[str, Any] = field(default_factory=dict)
    optional: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_sample_ref(self.ref)
        _validate_string_mapping("required", self.required)
        _validate_string_mapping("optional", self.optional)

    def to_dict(self) -> dict[str, Any]:
        role, modality = self.ref
        return {
            "role": role.value,
            "modality": modality.value,
            "required": dict(self.required),
            "optional": dict(self.optional),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SampleItemEntry:
        return cls(
            ref=(
                Role(_required_str(data, "role")),
                Modality(_required_str(data, "modality")),
            ),
            required=_optional_mapping(data, "required"),
            optional=_optional_mapping(data, "optional"),
        )


@dataclass(frozen=True)
class SampleManifestEntry:
    sample_id: str
    dataset_name: str
    sample_index: int | None = None
    source: Mapping[str, Any] = field(default_factory=dict)
    items: tuple[SampleItemEntry, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_str("sample_id", self.sample_id)
        _validate_non_empty_str("dataset_name", self.dataset_name)
        if self.sample_index is not None and self.sample_index < 0:
            raise ValueError("sample_index must be non-negative.")
        _validate_mapping("source", self.source)
        _validate_mapping("metadata", self.metadata)
        if not isinstance(self.items, tuple):
            raise TypeError("items must be a tuple.")
        for item in self.items:
            if not isinstance(item, SampleItemEntry):
                raise TypeError("items entries must be SampleItemEntry instances.")
        refs = [item.ref for item in self.items]
        if len(set(refs)) != len(refs):
            raise ValueError("items cannot contain duplicate sample refs.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "sample_index": self.sample_index,
            "source": dict(self.source),
            "items": [item.to_dict() for item in self.items],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SampleManifestEntry:
        return cls(
            sample_id=_required_str(data, "sample_id"),
            dataset_name=_required_str(data, "dataset_name"),
            sample_index=_optional_int(data, "sample_index"),
            source=_optional_mapping(data, "source"),
            items=tuple(
                SampleItemEntry.from_dict(item)
                for item in _sequence_value(data["items"], "items")
            ),
            metadata=_optional_mapping(data, "metadata"),
        )

    def item(self, ref: tuple[Role, Modality]) -> SampleItemEntry | None:
        _validate_sample_ref(ref)
        for item in self.items:
            if item.ref == ref:
                return item
        return None


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
            if not isinstance(self.shape, tuple):
                raise TypeError("shape must be a tuple.")
            for dim in self.shape:
                if dim < 0:
                    raise ValueError("shape dimensions must be non-negative.")
        if self.dtype is not None:
            _validate_non_empty_str("dtype", self.dtype)
        if self.checksum is not None:
            _validate_non_empty_str("checksum", self.checksum)
        _validate_mapping("provenance", self.provenance)

    def to_dict(self) -> dict[str, Any]:
        data = view_ref_to_dict(self.ref)
        data |= {
            "revision": self.revision,
            "sample_id": self.sample_id,
            "shard": self.shard,
            "key": self.key,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "checksum": self.checksum,
            "provenance": dict(self.provenance),
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ViewManifestEntry:
        shape = data.get("shape")
        if shape is not None:
            shape = tuple(
                _int_value(dim, "shape") for dim in _sequence_value(shape, "shape")
            )
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


def view_ref_to_dict(ref: ViewRef) -> dict[str, str]:
    if not isinstance(ref, ViewRef):
        raise TypeError("ref must be a ViewRef.")
    return {
        "modality": ref.modality.value,
        "role": ref.role.value,
        "view_key": ref.view_key.value,
    }


def view_ref_from_dict(data: Mapping[str, Any]) -> ViewRef:
    modality = Modality(_required_str(data, "modality"))
    return ViewRef(
        modality=modality,
        view_key=_view_key_from_string(modality, _required_str(data, "view_key")),
        role=Role(_optional_str(data, "role") or Role.DEFAULT.value),
    )


def _validate_view_key(modality: Modality, view_key: View) -> None:
    match modality:
        case Modality.AUDIO:
            if not isinstance(view_key, AudioView):
                raise TypeError("audio ViewRef requires an AudioView.")
        case Modality.IMAGE:
            if not isinstance(view_key, ImageView):
                raise TypeError("image ViewRef requires an ImageView.")
        case Modality.TEXT:
            if not isinstance(view_key, TextView):
                raise TypeError("text ViewRef requires a TextView.")


def _view_key_from_string(modality: Modality, value: str) -> View:
    match modality:
        case Modality.AUDIO:
            return AudioView(value)
        case Modality.IMAGE:
            return ImageView(value)
        case Modality.TEXT:
            return TextView(value)
    raise ValueError(f"Unsupported modality: {modality!r}.")


def _validate_sample_ref(value: tuple[Role, Modality]) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("ref must be a (Role, Modality) tuple.")
    role, modality = value
    if not isinstance(role, Role):
        raise TypeError("sample ref role must be a Role.")
    if not isinstance(modality, Modality):
        raise TypeError("sample ref modality must be a Modality.")


def _validate_string_mapping(name: str, value: Mapping[str, Any]) -> None:
    _validate_mapping(name, value)
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings.")


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


def _optional_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping.")
    return dict(value)


def _optional_sequence(
    data: Mapping[str, Any], key: str
) -> tuple[Mapping[str, Any], ...]:
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
