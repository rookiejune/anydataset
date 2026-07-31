from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from typing_extensions import TypeGuard

from ...cache import anydataset_home
from ...dataset.abc import AnyDataset, MapStyleABC
from ...store.reader import StoreDataset, read_store_manifest
from ...types import Source, Spec
from .generations import (
    filter_cache_root,
    filter_generation_lock_path,
)
from ..rules import rule_cache_key, rule_identity

if TYPE_CHECKING:
    from ..api import FilteredDataset, FilterRule

    FilterBase = Union[AnyDataset, StoreDataset, FilteredDataset]
else:
    FilterBase = MapStyleABC

_FILTER_VIEW_SCHEMA_VERSION = 3


def filter_base(dataset: object) -> FilterBase:
    from ..api import FilteredDataset

    if isinstance(dataset, (AnyDataset, StoreDataset, FilteredDataset)):
        return dataset
    raise TypeError("dataset must be an AnyDataset, StoreDataset, or FilteredDataset.")


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
    if _is_filtered_dataset(dataset):
        return filter_spec(dataset.base)
    raise TypeError("dataset must be an AnyDataset, StoreDataset, or FilteredDataset.")


def filter_identity(
    dataset: FilterBase,
    *,
    input_id: str | None = None,
) -> dict[str, Any]:
    if _is_filtered_dataset(dataset):
        cache_path = Path(dataset.cache_path)
        identity = {
            "view_schema_version": _FILTER_VIEW_SCHEMA_VERSION,
            "kind": "filtered",
            "base": filter_identity(dataset.base, input_id=dataset.input_id),
            "rule": rule_identity(
                dataset.rule.name,
                dataset.rule.rule_id,
                dataset.rule.version,
                dataset.rule.content_id,
            ),
            "labels": list(dataset.labels),
            "cache_key": filter_cache_root(cache_path).name,
            "generation": cache_path.name,
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
    provenance = _store_provenance(dataset, spec)
    if provenance:
        identity["provenance"] = dict(provenance)
    selection = _store_selection(dataset)
    if selection is not None:
        identity["views"] = selection
    return _with_input_id(identity, input_id)


def _store_provenance(
    dataset: FilterBase,
    spec: Spec,
) -> Mapping[str, str]:
    if isinstance(dataset, StoreDataset):
        return dataset.manifest.provenance
    if isinstance(dataset, AnyDataset) and spec.source == Source.STORE:
        return read_store_manifest(
            spec.path,
            legacy_policy="reject",
        ).provenance
    return {}


def _store_selection(dataset: FilterBase) -> list[list[str]] | None:
    if isinstance(dataset, StoreDataset):
        views = tuple(dataset.views)
    elif isinstance(dataset, AnyDataset) and dataset.spec.source == Source.STORE:
        if dataset.selected_store_views is None:
            return None
        views = dataset.selected_store_views
    else:
        return None
    return [[role.value, modality.value, view.value] for role, modality, view in views]


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
        "rule": rule_identity(
            rule.name,
            rule.rule_id,
            rule.version,
            rule.content_id,
        ),
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
        / rule_cache_key(
            rule.name,
            rule.rule_id,
            rule.version,
            rule.content_id,
        )
    )


def filter_lock_path(
    rule: FilterRule,
    identity: Mapping[str, Any],
) -> Path:
    return filter_generation_lock_path(filter_path(rule, identity))


def filter_identity_key(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def _with_input_id(
    identity: dict[str, Any],
    input_id: str | None,
) -> dict[str, Any]:
    if input_id is not None:
        identity["input_id"] = input_id
    return identity


def _is_filtered_dataset(dataset: object) -> TypeGuard[FilteredDataset]:
    from ..api import FilteredDataset

    return isinstance(dataset, FilteredDataset)
