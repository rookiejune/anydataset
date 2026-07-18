from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import auto
from typing import Any, Union

import torch

from .._compat import StrEnum
from ..types import item

FieldKey = Union[
    item.AudioView,
    item.ImageView,
    item.TextView,
    item.AudioMeta,
    item.ImageMeta,
    item.TextMeta,
]


class FieldGroup(StrEnum):
    VIEWS = auto()
    META = auto()


@dataclass(frozen=True)
class FieldRef:
    ref: item.Reference
    group: FieldGroup
    key: FieldKey


@dataclass(frozen=True)
class Batch:
    sample: item.Sample
    masks: Mapping[FieldRef, torch.Tensor]

    def lengths(self, field: FieldRef) -> torch.Tensor:
        return field_lengths(self, field)


@dataclass(frozen=True)
class _Collator:
    schema: item.Schema

    def __call__(self, samples: Sequence[item.Sample]) -> Batch:
        return _collate_samples(samples, self.schema)


def field_lengths(batch: Batch, field: FieldRef) -> torch.Tensor:
    try:
        mask = batch.masks[field]
    except KeyError as exc:
        raise KeyError(f"Batch has no mask for {field!r}.") from exc
    if mask.ndim < 2:
        raise ValueError(f"Cannot derive sequence lengths from mask for {field!r}.")
    if mask.ndim > 2:
        dims = tuple(range(1, mask.ndim - 1))
        mask = mask.any(dim=dims)
    return mask.to(torch.int64).sum(dim=-1)


def collate_fn(
    schema: item.Schema,
) -> Callable[[Sequence[item.Sample]], Batch]:
    return _Collator(dict(schema))


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
    if ref[1] is item.Modality.AUDIO:
        if not isinstance(sample_item, item.AudioItem):
            raise TypeError(f"{ref!r} requires AudioItem samples.")
    elif ref[1] is item.Modality.IMAGE:
        if not isinstance(sample_item, item.ImageItem):
            raise TypeError(f"{ref!r} requires ImageItem samples.")
    elif ref[1] is item.Modality.TEXT:
        if not isinstance(sample_item, item.TextItem):
            raise TypeError(f"{ref!r} requires TextItem samples.")
    else:
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
    meta, meta_masks = _collate_group(
        ref,
        items,
        FieldGroup.META,
        requirement.meta,
    )

    masks = view_masks | meta_masks
    if ref[1] is item.Modality.AUDIO:
        return item.AudioItem(
            views=views,
            meta=meta,
        ), masks
    if ref[1] is item.Modality.IMAGE:
        return item.ImageItem(
            views=views,
            meta=meta,
        ), masks
    if ref[1] is item.Modality.TEXT:
        return item.TextItem(
            views=views,
            meta=meta,
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
    mappings = tuple(_field_mapping(_item, group) for _item in items)
    for key in keys:
        values = [mapping[key] for mapping in mappings]

        field = FieldRef(ref=ref, group=group, key=key)
        value, mask = _collate_values(values, field)
        fields[key] = value
        if mask is not None:
            masks[field] = mask
    return fields, masks


def _field_mapping(
    item: item.Item,
    group: FieldGroup,
) -> Mapping[Any, Any]:
    if group is FieldGroup.VIEWS:
        return item.views
    if group is FieldGroup.META:
        return item.meta
    raise TypeError(f"Unsupported field group: {group!r}.")


def _collate_values(
    values: Sequence[Any],
    field: FieldRef,
) -> tuple[Any, torch.Tensor | None]:
    if _is_waveform_field(field):
        return _collate_waveforms(values, field)
    if _is_codec_field(field):
        return _collate_codec_codes(values, field)

    if field.group is FieldGroup.VIEWS:
        mappings = [value for value in values if isinstance(value, Mapping)]
        if len(mappings) == len(values):
            return _collate_mappings(mappings, field)
        if mappings:
            raise TypeError(
                f"Cannot collate mixed mapping and non-mapping values for {field!r}."
            )

    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if len(tensors) == len(values):
        return _batch_tensors(tensors, field)
    if tensors:
        raise TypeError(
            f"Cannot collate mixed tensor and non-tensor values for {field!r}."
        )
    return list(values), None


def _is_waveform_field(field: FieldRef) -> bool:
    return (
        field.group is FieldGroup.VIEWS
        and field.ref[1] is item.Modality.AUDIO
        and field.key == item.AudioView.WAVEFORM
    )


def _is_codec_field(field: FieldRef) -> bool:
    return (
        field.group is FieldGroup.VIEWS
        and field.ref[1] is item.Modality.AUDIO
        and field.key
        in {
            item.AudioView.LONGCAT,
            item.AudioView.DAC,
            item.AudioView.STABLE,
            item.AudioView.UNICODEC,
        }
    )


def _collate_codec_codes(
    values: Sequence[Any],
    field: FieldRef,
) -> tuple[torch.Tensor, torch.Tensor]:
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError(f"Codec view values must be tensors for {field!r}.")

    tensors = list(values)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError(
            f"Codec view values must have shape [frame, codebook] for {field!r}."
        )
    if any(
        tensor.dtype == torch.bool
        or tensor.is_floating_point()
        or tensor.is_complex()
        for tensor in tensors
    ):
        raise TypeError(f"Codec view values must contain integer ids for {field!r}.")

    dtype = tensors[0].dtype
    if any(tensor.dtype != dtype for tensor in tensors):
        raise TypeError(f"Codec view values must share one dtype for {field!r}.")
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError(f"Codec view values must share one device for {field!r}.")

    codebooks = tensors[0].shape[1]
    if codebooks == 0 or any(tensor.shape[1] != codebooks for tensor in tensors):
        raise ValueError(
            f"Codec view values must share one non-empty codebook axis for {field!r}."
        )

    lengths = torch.tensor(
        [tensor.shape[0] for tensor in tensors],
        dtype=torch.int64,
        device=tensors[0].device,
    )
    max_length = int(lengths.max().item())
    batch = tensors[0].new_zeros((len(tensors), max_length, codebooks))
    mask = torch.zeros(
        (len(tensors), max_length),
        dtype=torch.bool,
        device=tensors[0].device,
    )
    for index, tensor in enumerate(tensors):
        length = tensor.shape[0]
        batch[index, :length] = tensor
        mask[index, :length] = True
    return batch, mask


def _collate_mappings(
    values: Sequence[Mapping[Any, Any]],
    field: FieldRef,
) -> tuple[dict[Any, Any], torch.Tensor | None]:
    keys = _mapping_keys(values, field)
    _validate_sample_mapping_lengths(values, field)

    fields: dict[Any, Any] = {}
    mask: torch.Tensor | None = None
    for key in keys:
        value, value_mask = _collate_values(
            [mapping[key] for mapping in values],
            field,
        )
        fields[key] = value
        mask = _merge_mapping_mask(mask, value_mask, field)
    return fields, mask


def _mapping_keys(
    values: Sequence[Mapping[Any, Any]],
    field: FieldRef,
) -> tuple[Any, ...]:
    if not values:
        raise ValueError(f"Cannot collate field with no mapping values for {field!r}.")

    keys = tuple(values[0])
    expected = set(keys)
    for value in values:
        if set(value) != expected:
            raise ValueError(f"Cannot collate mappings with different keys for {field!r}.")
    return keys


def _validate_sample_mapping_lengths(
    values: Sequence[Mapping[Any, Any]],
    field: FieldRef,
) -> None:
    for value in values:
        lengths = {
            entry.shape[-1]
            for entry in value.values()
            if isinstance(entry, torch.Tensor) and entry.ndim > 0
        }
        if len(lengths) > 1:
            raise ValueError(
                f"Mapping tensor values in a single sample must share the same "
                f"last dimension for {field!r}."
            )


def _merge_mapping_mask(
    current: torch.Tensor | None,
    value: torch.Tensor | None,
    field: FieldRef,
) -> torch.Tensor | None:
    if value is None:
        return current
    mask = _mapping_time_mask(value)
    if current is None:
        return mask
    if not torch.equal(current, mask):
        raise ValueError(f"Mapping values produced incompatible masks for {field!r}.")
    return current


def _mapping_time_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim <= 2:
        return mask
    dims = tuple(range(1, mask.ndim - 1))
    return mask.any(dim=dims)


def _collate_waveforms(
    values: Sequence[tuple[torch.Tensor, int]],
    field: FieldRef,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    waveforms = [
        waveform if isinstance(waveform, torch.Tensor) else torch.as_tensor(waveform)
        for waveform, _ in values
    ]
    batch, mask = _batch_tensors(waveforms, field)
    rates = torch.tensor(
        [sample_rate for _, sample_rate in values],
        dtype=torch.int64,
        device=batch.device,
    )
    return (batch, rates), mask


def _batch_tensors(
    tensors: Sequence[torch.Tensor],
    field: FieldRef,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError(f"Cannot collate field with no tensor values for {field!r}.")

    shapes = [tuple(tensor.shape) for tensor in tensors]
    if all(shape == shapes[0] for shape in shapes):
        batch = torch.stack(tuple(tensors))
        mask = torch.ones(batch.shape, dtype=torch.bool, device=batch.device)
        return batch, mask

    rank = len(shapes[0])
    prefix = shapes[0][:-1]
    if rank == 0 or any(len(shape) != rank or shape[:-1] != prefix for shape in shapes):
        raise ValueError(f"Only the last tensor dimension may vary for {field!r}.")

    max_len = max(shape[-1] for shape in shapes)
    padded: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for tensor in tensors:
        length = tensor.shape[-1]
        if length < max_len:
            padding = tensor.new_zeros((*prefix, max_len - length))
            tensor = torch.cat((tensor, padding), dim=-1)
        padded.append(tensor)

        mask = tensor.new_zeros((*prefix, max_len), dtype=torch.bool)
        slices = (*[slice(None)] * len(prefix), slice(0, length))
        mask[slices] = True
        masks.append(mask)

    return torch.stack(tuple(padded)), torch.stack(tuple(masks))
