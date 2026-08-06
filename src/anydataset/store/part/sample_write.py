from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from ...dataset.universe import SampleIdentity
from ...types.item import (
    Item,
    Modality,
    Role,
    Sample,
    View,
)
from .._refs import sample_ref_path, view_path
from ..manifest.schema import SampleItem, SampleManifestEntry, string_key_dict

_SAMPLE_ID_SET_LIMIT = 1_000_000


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
    try:
        item = sample[role, modality]
    except KeyError as exc:
        ref = sample_ref_path((role, modality))
        raise KeyError(f"Sample is missing item {ref}.") from exc
    item_type = modality.item_type()
    if not isinstance(item, item_type) or not isinstance(key, modality.view_type()):
        raise TypeError(
            f"{modality.value} store view must reference an {item_type.__name__}."
        )
    try:
        return cast(Mapping[View, Any], item.views)[key]
    except KeyError as exc:
        raise KeyError(f"Sample item is missing view {view_path(view)}.") from exc


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


def inherited_sample_id(source: object, index: int) -> str | None:
    """Return a stable source identity when the input exposes that capability."""

    if not isinstance(source, SampleIdentity):
        return None
    value = source.sample_id(index)
    if not isinstance(value, str) or not value:
        raise ValueError("sample_id() must return a non-empty string.")
    return value


class UniqueSampleIds:
    """Validate sample IDs with bounded memory for large store writes."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._ids: set[str] | None = set()
        self._temporary: TemporaryDirectory[str] | None = None
        self._connection: sqlite3.Connection | None = None

    def add(self, value: str) -> None:
        if self._ids is not None:
            if value in self._ids:
                raise ValueError(f"Duplicate sample_id {value!r}.")
            self._ids.add(value)
            if len(self._ids) >= _SAMPLE_ID_SET_LIMIT:
                self._spill()
            return
        self._insert(value)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def _spill(self) -> None:
        ids = self._ids
        if ids is None:
            return
        self._temporary = TemporaryDirectory(
            prefix=".sample-ids-",
            dir=self._root,
        )
        self._connection = sqlite3.connect(
            Path(self._temporary.name) / "ids.sqlite"
        )
        self._connection.execute(
            "CREATE TABLE sample_ids (sample_id TEXT PRIMARY KEY)"
        )
        self._connection.executemany(
            "INSERT INTO sample_ids(sample_id) VALUES (?)",
            ((value,) for value in ids),
        )
        self._ids = None

    def _insert(self, value: str) -> None:
        connection = self._connection
        if connection is None:
            raise RuntimeError("sample id validation database is missing.")
        try:
            connection.execute(
                "INSERT INTO sample_ids(sample_id) VALUES (?)",
                (value,),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Duplicate sample_id {value!r}.") from exc


def slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-")
    return text or "sample"
