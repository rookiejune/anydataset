from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..cache import anydataset_home
from ..dataset.abc import AnyDataset, MapStyleABC, MergedDataset
from ..store.reader import StoreDataset
from ..types import Source, Spec
from .generations import (
    filter_cache_root,
    filter_generation_lock_path,
)
from .rules import rule_cache_key

if TYPE_CHECKING:
    from .api import FilterRule


FilterBase = MapStyleABC

_FILTER_VIEW_SCHEMA_VERSION = 2


def filter_base(dataset: object) -> FilterBase:
    if isinstance(dataset, MapStyleABC):
        return dataset
    raise TypeError("dataset must be a MapStyleABC.")


def filter_universe(dataset: FilterBase) -> FilterBase:
    dataset = filter_base(dataset)
    if _is_filtered_dataset(dataset):
        return filter_universe(dataset.base)
    return dataset


def filter_spec(dataset: FilterBase) -> Spec:
    if isinstance(dataset, AnyDataset):
        return dataset.spec
    if isinstance(dataset, StoreDataset):
        return Spec(
            source=Source.STORE,
            path=str(dataset.root),
            split=dataset.manifest.split,
        )
    if isinstance(dataset, MergedDataset):
        return filter_spec(dataset.left)
    if _is_filtered_dataset(dataset):
        return filter_spec(dataset.base)
    raise TypeError("dataset must be an AnyDataset, StoreDataset, or MergedDataset.")


def filter_identity(
    dataset: FilterBase,
    *,
    input_id: str | None = None,
) -> dict[str, Any]:
    if _is_filtered_dataset(dataset):
        identity = {
            "view_schema_version": _FILTER_VIEW_SCHEMA_VERSION,
            "kind": "filtered",
            "base": filter_identity(dataset.base, input_id=dataset.input_id),
            "rule": {"name": dataset.rule.name},
            "labels": list(dataset.labels),
            "cache_key": filter_cache_root(dataset.cache_path).name,
            "sample_count": len(dataset),
        }
        return _with_input_id(identity, input_id)
    if isinstance(dataset, MergedDataset):
        children = sorted(
            (
                dataset_identity(child, allow_external=input_id is not None)
                for child in merged_children(dataset)
            ),
            key=filter_identity_key,
        )
        identity = {
            "view_schema_version": _FILTER_VIEW_SCHEMA_VERSION,
            "kind": "merged",
            "children": children,
            "sample_count": len(dataset),
        }
        return _with_input_id(identity, input_id)
    spec = filter_spec(dataset)
    identity = {
        "kind": "physical",
        "type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "spec_id": spec.id,
        "spec": spec.to_dict(),
    }
    return _with_input_id(identity, input_id)


def dataset_identity(
    dataset: Any,
    *,
    allow_external: bool,
) -> dict[str, Any]:
    if (
        isinstance(dataset, (AnyDataset, StoreDataset, MergedDataset))
        or _is_filtered_dataset(dataset)
    ):
        return filter_identity(dataset)
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TypeError("merged dataset inputs must be map-style datasets.")
    if not allow_external:
        dataset_type = f"{type(dataset).__module__}.{type(dataset).__qualname__}"
        raise TypeError(
            "input_id is required when a merged dataset contains an external "
            f"map-style child: {dataset_type}."
        )
    return {
        "view_schema_version": _FILTER_VIEW_SCHEMA_VERSION,
        "kind": "map_style",
        "type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "sample_count": len(dataset),
    }


def merged_children(dataset: MergedDataset) -> tuple[Any, ...]:
    children: list[Any] = []
    for child in (dataset.left, dataset.right):
        if isinstance(child, MergedDataset):
            children.extend(merged_children(child))
            continue
        children.append(child)
    return tuple(children)


def metadata(
    identity: Mapping[str, Any],
    base_count: int,
    rule: FilterRule,
) -> dict[str, Any]:
    output = {
        "schema_version": 5,
        "base": {
            "identity": dict(identity),
            "identity_id": filter_identity_key(identity),
            "sample_count": base_count,
        },
        "rule": {"name": rule.name},
    }
    if identity.get("kind") == "physical":
        output["base"]["spec_id"] = identity["spec_id"]
    else:
        output["base"]["view"] = dict(identity)
    return output


def filter_path(
    rule: FilterRule,
    identity: Mapping[str, Any],
) -> Path:
    return (
        anydataset_home()
        / "cache"
        / "filters"
        / filter_identity_key(identity)
        / rule_cache_key(rule.name)
    )


def filter_lock_path(
    rule: FilterRule,
    identity: Mapping[str, Any],
) -> Path:
    return filter_generation_lock_path(filter_path(rule, identity))


def filter_identity_key(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _with_input_id(
    identity: dict[str, Any],
    input_id: str | None,
) -> dict[str, Any]:
    if input_id is not None:
        identity["input_id"] = input_id
    return identity


def _is_filtered_dataset(dataset: object) -> bool:
    from .api import FilteredDataset

    return isinstance(dataset, FilteredDataset)
