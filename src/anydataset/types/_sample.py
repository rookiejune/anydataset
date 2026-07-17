from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .item import Item, Sample, Schema, View


def select(sample: Sample, schema: Schema) -> Sample:
    return {
        reference: sample[reference].select_by(requirement)
        for reference, requirement in schema.items()
    }


def merge(left: Sample, right: Sample, *, context: str) -> Sample:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise TypeError(f"{context} samples must be mappings.")

    result = dict(left)
    for ref, item in right.items():
        current = result.get(ref)
        if current is None:
            result[ref] = item
            continue
        result[ref] = merge_items(current, item, ref=ref, context=context)
    return result


def merge_items(
    left: Item,
    right: Item,
    *,
    ref: object,
    context: str,
) -> Item:
    if type(left) is not type(right):
        raise TypeError(f"{context} item {ref!r} has incompatible types.")

    conflicts = set(left.views) & set(right.views)
    if conflicts:
        view = min(conflicts, key=lambda value: value.value)
        raise ValueError(f"{context} item {ref!r} view conflict for {view!r}.")

    meta = dict(left.meta)
    for key, value in right.meta.items():
        if key in meta and not values_equal(meta[key], value):
            raise ValueError(
                f"{context} item {ref!r} metadata conflict for {key!r}."
            )
        meta[key] = value

    return type(left)(views={**left.views, **right.views}, meta=meta)


def values_equal(left: Any, right: Any) -> bool:
    equal = left == right
    if isinstance(equal, bool):
        return equal
    try:
        return bool(equal)
    except (TypeError, ValueError, RuntimeError):
        return left is right
