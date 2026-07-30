from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

from ..types.item import (
    Item,
    Modality,
    Role,
    Sample,
    View,
)
from .manifest import SampleItem, SampleManifestEntry, string_key_dict


def sample_manifest_entry(
    sample: Sample,
    sample_id: str,
    sample_index: int,
) -> SampleManifestEntry:
    return SampleManifestEntry(
        sample_id=sample_id,
        sample_index=sample_index,
        items=tuple(item_entry(ref, item) for ref, item in sample.items()),
    )


def item_entry(ref: tuple[Role, Modality], item: Item) -> SampleItem:
    return ref, string_key_dict(item.meta)


def sample_view_refs(sample: Sample) -> tuple[tuple[Role, Modality, View], ...]:
    views: list[tuple[Role, Modality, View]] = []
    for (role, modality), item in sample.items():
        for view in item.views:
            views.append((role, modality, view))
    return tuple(views)


def explicit_views(
    value: object,
) -> tuple[tuple[Role, Modality, View], ...] | None:
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise TypeError("views must be a tuple of (Role, Modality, View) tuples.")

    views: list[tuple[Role, Modality, View]] = []
    seen: set[tuple[Role, Modality, View]] = set()
    for entry in value:
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise TypeError("views entries must be (Role, Modality, View) tuples.")
        role, modality, key = entry
        if not isinstance(role, Role):
            raise TypeError("store view role must be a Role.")
        if not isinstance(modality, Modality):
            raise TypeError("store view modality must be a Modality.")
        view_type = modality.view_type()
        if not isinstance(key, view_type):
            raise TypeError(
                f"{modality.value} store views must use {view_type.__name__} values."
            )
        view = role, modality, key
        if view in seen:
            raise ValueError(f"Duplicate store view {view_path(view)}.")
        seen.add(view)
        views.append(view)
    return tuple(views)


def validate_view_sets(
    sample: Sample,
    expected: dict[tuple[Role, Modality], frozenset[View]],
    sample_id: str,
) -> None:
    for ref, item in sample.items():
        views = frozenset(item.views)
        previous = expected.setdefault(ref, views)
        if views != previous:
            raise ValueError(
                f"Sample {sample_id} view set for {sample_ref_path(ref)} "
                f"does not match earlier samples."
            )


def sample_view_value(sample: Sample, view: tuple[Role, Modality, View]) -> Any:
    role, modality, key = view
    item = sample.get((role, modality))
    if item is None:
        return None
    item_type = modality.item_type()
    if not isinstance(item, item_type) or not isinstance(key, modality.view_type()):
        raise TypeError(
            f"{modality.value} store view must reference an {item_type.__name__}."
        )
    return cast(Mapping[View, Any], item.views).get(key)


def validate_item(modality: Modality, item: Item) -> None:
    item_type = modality.item_type()
    if not isinstance(item, item_type):
        raise TypeError(
            f"{modality.value} sample items must be {item_type.__name__} instances."
        )


def validate_sample(sample: Sample) -> None:
    for ref, item in sample.items():
        if not isinstance(ref, tuple) or len(ref) != 2:
            raise TypeError("sample keys must be (Role, Modality) tuples.")
        role, modality = ref
        if not isinstance(role, Role):
            raise TypeError("sample role keys must be Role instances.")
        if not isinstance(modality, Modality):
            raise TypeError("sample modality keys must be Modality instances.")
        validate_item(modality, item)


def sample_id_prefix(dataset_id: str) -> str:
    return slug(dataset_id)


def sample_id(dataset: str, index: int) -> str:
    return f"{index:012d}-{dataset}"


def slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-")
    return text or "sample"


def view_path(view: tuple[Role, Modality, View]) -> tuple[str, str, str]:
    role, modality, key = view
    return role.value, modality.value, key.value


def sample_ref_path(ref: tuple[Role, Modality]) -> tuple[str, str]:
    role, modality = ref
    return role.value, modality.value
