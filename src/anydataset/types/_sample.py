from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .item import Item, Sample, Schema


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
    if left is right:
        return True

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        try:
            return torch.equal(left, right)
        except (TypeError, RuntimeError):
            return False

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if len(left) != len(right):
            return False
        return all(
            key in right and values_equal(value, right[key])
            for key, value in left.items()
        )

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(
            values_equal(l_value, r_value) for l_value, r_value in zip(left, right)
        )

    left_shape = getattr(left, "shape", None)
    right_shape = getattr(right, "shape", None)
    if (left_shape is None) != (right_shape is None) or left_shape != right_shape:
        return False

    try:
        equal = left == right
    except (TypeError, ValueError, RuntimeError):
        return False
    if isinstance(equal, bool):
        return equal
    if equal is NotImplemented:
        return False

    reduce = getattr(equal, "all", None)
    if callable(reduce):
        try:
            equal = reduce()
        except (TypeError, ValueError, RuntimeError):
            return False
    try:
        return bool(equal)
    except (TypeError, ValueError, RuntimeError):
        return False
