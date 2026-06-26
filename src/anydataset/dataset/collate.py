from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from numbers import Number
from pathlib import Path
from typing import Any

import torch

from ..types import item


type FieldKey = (
    item.AudioView
    | item.ImageView
    | item.TextView
    | item.AudioKey
    | item.ImageKey
    | item.TextKey
    | item.AudioOptKey
    | item.ImageOptKey
    | item.TextOptKey
)


_MISSING = object()


class FieldGroup(StrEnum):
    VIEWS = auto()
    REQUIRED = auto()
    OPTIONAL = auto()


@dataclass(frozen=True)
class FieldRef:
    ref: item.Reference
    group: FieldGroup
    key: FieldKey


@dataclass(frozen=True)
class Batch:
    sample: item.Sample
    masks: Mapping[FieldRef, torch.Tensor]


def collate_fn(
    schema: item.Schema,
) -> Callable[[Sequence[item.Sample]], Batch]:
    def collate(samples: Sequence[item.Sample]) -> Batch:
        return _collate_samples(samples, schema)

    return collate


def _collate_samples(
    samples: Sequence[item.Sample],
    schema: item.Schema,
) -> Batch:
    if not samples:
        raise ValueError("Cannot collate an empty sample batch.")

    sample: dict[item.Reference, item.Item] = {}
    masks: dict[FieldRef, torch.Tensor] = {}
    for ref, requirement in schema.items():
        items = [_sample_item(row, ref) for row in samples]
        item, item_masks = _collate_item(ref, items, requirement)
        sample[ref] = item
        masks.update(item_masks)
    return Batch(sample=sample, masks=masks)


def _sample_item(
    sample: item.Sample,
    ref: item.Reference,
) -> item.Item:
    sample_item = sample[ref]
    match ref[1]:
        case item.Modality.AUDIO:
            if not isinstance(sample_item, item.AudioItem):
                raise TypeError(f"{ref!r} requires AudioItem samples.")
        case item.Modality.IMAGE:
            if not isinstance(sample_item, item.ImageItem):
                raise TypeError(f"{ref!r} requires ImageItem samples.")
        case item.Modality.TEXT:
            if not isinstance(sample_item, item.TextItem):
                raise TypeError(f"{ref!r} requires TextItem samples.")
        case _:
            raise TypeError(f"Unsupported sample reference: {ref!r}.")
    return sample_item


def _collate_item(
    ref: item.Reference,
    items: Sequence[item.Item],
    requirement: item.Requirement,
) -> tuple[item.Item, dict[FieldRef, torch.Tensor]]:
    views, view_masks = _collate_group(
        ref,
        items,
        FieldGroup.VIEWS,
        requirement.views,
    )
    required, required_masks = _collate_group(
        ref,
        items,
        FieldGroup.REQUIRED,
        requirement.required,
    )
    optional, optional_masks = _collate_group(
        ref,
        items,
        FieldGroup.OPTIONAL,
        requirement.optional,
    )

    masks = view_masks | required_masks | optional_masks
    match ref[1]:
        case item.Modality.AUDIO:
            return item.AudioItem(
                views=views,
                required=required,
                optional=optional,
            ), masks
        case item.Modality.IMAGE:
            return item.ImageItem(
                views=views,
                required=required,
                optional=optional,
            ), masks
        case item.Modality.TEXT:
            return item.TextItem(
                views=views,
                required=required,
                optional=optional,
            ), masks
    raise TypeError(f"Unsupported sample reference: {ref!r}.")


def _collate_group(
    ref: item.Reference,
    items: Sequence[item.Item],
    group: FieldGroup,
    keys: frozenset[Any],
) -> tuple[dict[Any, Any], dict[FieldRef, torch.Tensor]]:
    fields: dict[Any, Any] = {}
    masks: dict[FieldRef, torch.Tensor] = {}
    for key in keys:
        values = _field_values(items, group, key)
        if values is None:
            continue

        field = FieldRef(ref=ref, group=group, key=key)
        value, mask = _collate_values(values, field)
        fields[key] = value
        if mask is not None:
            masks[field] = mask
    return fields, masks


def _field_values(
    items: Sequence[item.Item],
    group: FieldGroup,
    key: Any,
) -> list[Any] | None:
    values: list[Any] = []
    missing = 0
    for item in items:
        mapping = _field_mapping(item, group)
        if key in mapping:
            values.append(mapping[key])
            continue
        if group is FieldGroup.OPTIONAL:
            values.append(_MISSING)
            missing += 1
            continue
        raise KeyError(f"Sample item is missing {group.value} field {key!r}.")

    if missing == 0:
        return values
    if missing == len(items):
        return None
    return values


def _field_mapping(
    item: item.Item,
    group: FieldGroup,
) -> Mapping[Any, Any]:
    match group:
        case FieldGroup.VIEWS:
            return item.views
        case FieldGroup.REQUIRED:
            return item.required
        case FieldGroup.OPTIONAL:
            return item.optional
    raise TypeError(f"Unsupported field group: {group!r}.")


def _collate_values(
    values: Sequence[Any],
    field: FieldRef,
) -> tuple[Any, torch.Tensor | None]:
    tensors = [
        None if value is _MISSING else _as_tensor(value)
        for value in values
    ]
    present_values = [value for value in values if value is not _MISSING]
    present = [
        tensor
        for value, tensor in zip(values, tensors, strict=True)
        if value is not _MISSING and tensor is not None
    ]
    if len(present) == len(present_values):
        return _batch_tensors(tensors, field)
    if present:
        raise TypeError(f"Cannot collate mixed tensor and non-tensor values for {field!r}.")

    if any(value is _MISSING for value in values):
        return [None if value is _MISSING else value for value in values], None
    if all(tensor is not None for tensor in tensors):
        return _batch_tensors(tensors, field)
    if any(tensor is not None for tensor in tensors):
        raise TypeError(f"Cannot collate mixed tensor and non-tensor values for {field!r}.")
    return list(values), None


def _as_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Number):
        return torch.as_tensor(value)
    if isinstance(value, str | bytes | bytearray | Path | Mapping):
        return None
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _batch_tensors(
    tensors: Sequence[torch.Tensor | None],
    field: FieldRef,
) -> tuple[torch.Tensor, torch.Tensor]:
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        raise ValueError(f"Cannot collate field with no tensor values for {field!r}.")

    device = present[0].device
    for tensor in tensors:
        if tensor is None:
            continue
        if tensor.device != device:
            raise ValueError(f"Cannot collate tensors from different devices for {field!r}.")

    dtype = _promote_dtype(present)
    tensors = [None if tensor is None else tensor.to(dtype=dtype) for tensor in tensors]
    shapes = [tuple(tensor.shape) for tensor in tensors if tensor is not None]
    if all(shape == shapes[0] for shape in shapes):
        shape = shapes[0]
        batch = torch.zeros((len(tensors), *shape), dtype=dtype, device=device)
        mask = torch.zeros((len(tensors), *shape), dtype=torch.bool, device=device)
        for index, tensor in enumerate(tensors):
            if tensor is None:
                continue
            batch[index] = tensor
            mask[index] = True
        return batch, mask

    rank = len(shapes[0])
    prefix = shapes[0][:-1]
    if rank == 0 or any(len(shape) != rank or shape[:-1] != prefix for shape in shapes):
        raise ValueError(f"Only the last tensor dimension may vary for {field!r}.")

    max_len = max(shape[-1] for shape in shapes)
    batch_shape = (len(tensors), *prefix, max_len)
    batch = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.zeros(batch_shape, dtype=torch.bool, device=device)
    for index, tensor in enumerate(tensors):
        if tensor is None:
            continue
        length = tensor.shape[-1]
        slices = (index, *[slice(None)] * len(prefix), slice(0, length))
        batch[slices] = tensor
        mask[slices] = True
    return batch, mask


def _promote_dtype(tensors: Sequence[torch.Tensor]) -> torch.dtype:
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype
