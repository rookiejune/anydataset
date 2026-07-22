from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Union

from .types import remap_lang
from .types.item import (
    AudioItem,
    AudioMeta,
    AudioView,
    ImageItem,
    ImageMeta,
    ImageView,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
    Sample,
)
from ._validation import positive_int


@dataclass(frozen=True)
class AudioMap:
    fields: AudioFields


@dataclass(frozen=True)
class ImageMap:
    fields: ImageFields


@dataclass(frozen=True)
class Labels:
    name: str


@dataclass(frozen=True)
class TextMap:
    fields: TextFields
    values: Mapping[TextMeta, Any] | None = None


FieldPath = Union[str, tuple[str, ...]]
AudioField = Union[AudioView, AudioMeta, Labels]
ImageField = Union[ImageView, ImageMeta]
TextField = Union[TextView, TextMeta]
AudioFields = Mapping[FieldPath, AudioField]
ImageFields = Mapping[FieldPath, ImageField]
TextFields = Mapping[FieldPath, TextField]
ItemMap = Union[AudioMap, ImageMap, TextMap]


def labels(name: str) -> Labels:
    return Labels(name=name)


def audio_map(fields: AudioFields) -> AudioMap:
    return AudioMap(fields=fields)


def image_map(fields: ImageFields) -> ImageMap:
    return ImageMap(fields=fields)


def text_map(
    fields: TextFields,
    *,
    values: Mapping[TextMeta, Any] | None = None,
) -> TextMap:
    return TextMap(fields=fields, values=values)


def sample_from_row(
    row: Mapping[str, Any],
    *,
    items: Mapping[tuple[Role, Modality], ItemMap] | None = None,
    audio: AudioFields | None = None,
    image: ImageFields | None = None,
    text: TextFields | None = None,
    text_values: Mapping[TextMeta, Any] | None = None,
) -> Sample:
    sample: dict[tuple[Role, Modality], Any] = {}
    if items is not None:
        for reference, item_map in items.items():
            _add_item(sample, reference, _load_item(row, reference, item_map))
    if audio is not None:
        _add_item(
            sample,
            (Role.DEFAULT, Modality.AUDIO),
            load_audio(row, audio),
        )
    if image is not None:
        _add_item(
            sample,
            (Role.DEFAULT, Modality.IMAGE),
            load_image(row, image),
        )
    if text is not None:
        _add_item(
            sample,
            (Role.DEFAULT, Modality.TEXT),
            load_text(
                row,
                text,
                values=text_values,
            ),
        )
    return sample


def load_audio(row: Mapping[str, Any], fields: AudioFields) -> AudioItem:
    views: dict[AudioView, Any] = {}
    meta: dict[AudioMeta, Any] = {}
    label_values: dict[str, Any] = {}

    for field, key in fields.items():
        value = _value(row, field)
        if isinstance(key, AudioView):
            if key == AudioView.WAVEFORM:
                waveform, sample_rate = _audio(value)
                if sample_rate is None:
                    raise ValueError("audio waveform views require sample_rate.")
                _assign(views, key, (waveform, sample_rate), target="audio view")
                continue
            _assign(views, key, value, target="audio view")
        elif isinstance(key, AudioMeta):
            _assign(meta, key, value, target="audio metadata")
        elif isinstance(key, Labels):
            _assign(label_values, key.name, value, target="audio label")
        else:
            raise TypeError(f"Unsupported audio field key: {key!r}.")

    if label_values:
        if AudioMeta.LABELS in meta:
            raise ValueError("Use either direct labels field or label field mappings.")
        meta[AudioMeta.LABELS] = label_values
    return AudioItem(views=views, meta=meta)


def load_image(row: Mapping[str, Any], fields: ImageFields) -> ImageItem:
    views: dict[ImageView, Any] = {}
    meta: dict[ImageMeta, Any] = {}

    for field, key in fields.items():
        value = _value(row, field)
        if isinstance(key, ImageView):
            _assign(views, key, value, target="image view")
        elif isinstance(key, ImageMeta):
            _assign(meta, key, value, target="image metadata")
        else:
            raise TypeError(f"Unsupported image field key: {key!r}.")
    return ImageItem(views=views, meta=meta)


def load_text(
    row: Mapping[str, Any],
    fields: TextFields,
    *,
    values: Mapping[TextMeta, Any] | None = None,
) -> TextItem:
    views: dict[TextView, Any] = {}
    meta: dict[TextMeta, Any] = _text_values(values)

    for field, key in fields.items():
        value = _value(row, field)
        if isinstance(key, TextView):
            _assign(views, key, value, target="text view")
        elif isinstance(key, TextMeta):
            _assign(meta, key, _text_value(key, value), target="text metadata")
        else:
            raise TypeError(f"Unsupported text field key: {key!r}.")
    return TextItem(views=views, meta=meta)


def _text_values(values: Mapping[TextMeta, Any] | None) -> dict[TextMeta, Any]:
    if values is None:
        return {}
    return {key: _text_value(key, value) for key, value in values.items()}


def _text_value(key: TextMeta, value: Any) -> Any:
    if key == TextMeta.LANG:
        return remap_lang(value)
    return value


def _load_item(
    row: Mapping[str, Any],
    reference: tuple[Role, Modality],
    item_map: ItemMap,
) -> AudioItem | ImageItem | TextItem:
    _, modality = reference
    if modality is Modality.AUDIO:
        if not isinstance(item_map, AudioMap):
            raise TypeError(f"{reference!r} requires AudioMap.")
        return load_audio(row, item_map.fields)
    if modality is Modality.IMAGE:
        if not isinstance(item_map, ImageMap):
            raise TypeError(f"{reference!r} requires ImageMap.")
        return load_image(row, item_map.fields)
    if modality is Modality.TEXT:
        if not isinstance(item_map, TextMap):
            raise TypeError(f"{reference!r} requires TextMap.")
        return load_text(row, item_map.fields, values=item_map.values)
    raise TypeError(f"Unsupported sample reference: {reference!r}.")


def _add_item(
    sample: dict[tuple[Role, Modality], Any],
    reference: tuple[Role, Modality],
    item: AudioItem | ImageItem | TextItem,
) -> None:
    if reference in sample:
        raise ValueError(f"Duplicate sample reference: {reference!r}.")
    sample[reference] = item


def _value(row: Mapping[str, Any], field: FieldPath) -> Any:
    if isinstance(field, str):
        return row[field]
    if not field:
        raise ValueError("Field path must not be empty.")

    value: Any = row
    for key in field:
        value = value[key]
    return value


def _audio(value: Any) -> tuple[Any, int | None]:
    if isinstance(value, Mapping):
        return value["array"], _maybe_int(value.get("sampling_rate"))

    decoded = _maybe_decode_audio(value)
    if decoded is not None:
        return decoded
    return value, None


def _maybe_decode_audio(audio: Any) -> tuple[Any, int] | None:
    get_all_samples = getattr(audio, "get_all_samples", None)
    if get_all_samples is None:
        return None

    samples = get_all_samples()
    data = getattr(samples, "data")
    sample_rate = getattr(samples, "sample_rate")
    return data, int(sample_rate)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("audio sampling_rate must be a positive integer.")
    if isinstance(value, str):
        try:
            result = int(value)
        except ValueError as exc:
            raise TypeError(
                "audio sampling_rate must be a positive integer."
            ) from exc
    else:
        try:
            result = operator.index(value)
        except TypeError as exc:
            raise TypeError(
                "audio sampling_rate must be a positive integer."
            ) from exc
    return positive_int("audio sampling_rate", result)


def _assign(mapping: dict[Any, Any], key: Any, value: Any, *, target: str) -> None:
    if key in mapping:
        raise ValueError(f"Duplicate {target} target: {key!r}.")
    mapping[key] = value
