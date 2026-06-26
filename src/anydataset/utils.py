from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .types import Preset, Source, SourceKey, Spec, source_key
from .types.item import (
    AudioItem,
    AudioKey,
    AudioOptKey,
    AudioView,
    ImageItem,
    ImageOptKey,
    ImageView,
    Modality,
    Role,
    TextItem,
    TextOptKey,
    TextView,
    Sample,
)

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
    values: Mapping[TextOptKey, Any] | None = None


type FieldPath = str | tuple[str, ...]
type AudioField = AudioView | AudioKey | AudioOptKey | Labels
type ImageField = ImageView | ImageOptKey
type TextField = TextView | TextOptKey
type AudioFields = Mapping[FieldPath, AudioField]
type ImageFields = Mapping[FieldPath, ImageField]
type TextFields = Mapping[FieldPath, TextField]
type ItemMap = AudioMap | ImageMap | TextMap


def labels(name: str) -> Labels:
    return Labels(name=name)


def audio_map(fields: AudioFields) -> AudioMap:
    return AudioMap(fields=fields)


def image_map(fields: ImageFields) -> ImageMap:
    return ImageMap(fields=fields)


def text_map(
    fields: TextFields,
    *,
    values: Mapping[TextOptKey, Any] | None = None,
) -> TextMap:
    return TextMap(fields=fields, values=values)


def resolve_dataset(dataset: str | Preset | Spec) -> Spec:
    if isinstance(dataset, Spec):
        return dataset
    if isinstance(dataset, Preset):
        return dataset.spec()
    if isinstance(dataset, str):
        return _resolve_shorthand(dataset)
    raise TypeError("dataset must be a string, Preset or Spec.")


def sample_from_row(
    row: Mapping[str, Any],
    *,
    items: Mapping[tuple[Role, Modality], ItemMap] | None = None,
    audio: AudioFields | None = None,
    image: ImageFields | None = None,
    text: TextFields | None = None,
    text_values: Mapping[TextOptKey, Any] | None = None,
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
    required: dict[AudioKey, Any] = {}
    optional: dict[AudioOptKey, Any] = {}
    label_values: dict[str, Any] = {}

    for field, key in fields.items():
        value = _value(row, field)
        if isinstance(key, AudioView):
            if key == AudioView.WAVEFORM:
                waveform, sample_rate = _audio(value)
                views[key] = waveform
                if sample_rate is not None:
                    required.setdefault(AudioKey.SAMPLE_RATE, sample_rate)
                continue
            views[key] = value
        elif isinstance(key, AudioKey):
            required[key] = _maybe_int(value)
        elif isinstance(key, AudioOptKey):
            optional[key] = value
        elif isinstance(key, Labels):
            label_values[key.name] = value
        else:
            raise TypeError(f"Unsupported audio field key: {key!r}.")

    if label_values:
        if AudioOptKey.LABELS in optional:
            raise ValueError("Use either direct labels field or label field mappings.")
        optional[AudioOptKey.LABELS] = label_values
    if AudioKey.SAMPLE_RATE not in required:
        raise ValueError("audio samples require sample_rate.")
    return AudioItem(views=views, required=required, optional=optional)


def load_image(row: Mapping[str, Any], fields: ImageFields) -> ImageItem:
    views: dict[ImageView, Any] = {}
    optional: dict[ImageOptKey, Any] = {}

    for field, key in fields.items():
        value = _value(row, field)
        if isinstance(key, ImageView):
            views[key] = value
        elif isinstance(key, ImageOptKey):
            optional[key] = value
        else:
            raise TypeError(f"Unsupported image field key: {key!r}.")
    return ImageItem(views=views, optional=optional)


def load_text(
    row: Mapping[str, Any],
    fields: TextFields,
    *,
    values: Mapping[TextOptKey, Any] | None = None,
) -> TextItem:
    views: dict[TextView, Any] = {}
    optional: dict[TextOptKey, Any] = dict(values or {})

    for field, key in fields.items():
        value = _value(row, field)
        if isinstance(key, TextView):
            views[key] = value
        elif isinstance(key, TextOptKey):
            optional[key] = value
        else:
            raise TypeError(f"Unsupported text field key: {key!r}.")
    return TextItem(views=views, optional=optional)


def _load_item(
    row: Mapping[str, Any],
    reference: tuple[Role, Modality],
    item_map: ItemMap,
) -> AudioItem | ImageItem | TextItem:
    _, modality = reference
    match modality:
        case Modality.AUDIO:
            if not isinstance(item_map, AudioMap):
                raise TypeError(f"{reference!r} requires AudioMap.")
            return load_audio(row, item_map.fields)
        case Modality.IMAGE:
            if not isinstance(item_map, ImageMap):
                raise TypeError(f"{reference!r} requires ImageMap.")
            return load_image(row, item_map.fields)
        case Modality.TEXT:
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


def _resolve_shorthand(shorthand: str) -> Spec:
    source, body = _split_source_prefix(shorthand)
    if source is not None:
        path, split = _split_name_and_split(body)
        if not path:
            raise ValueError(
                f"{source_key(source)} dataset shorthand must include a path."
            )
        return Spec(source=source, path=path, split=split)

    name, split = _split_name_and_split(shorthand)
    try:
        preset = Preset(name)
    except ValueError as exc:
        raise KeyError(
            f"Unknown dataset preset {name!r}. Use a registered source shorthand "
            "such as `hf://`, `hf-disk://` or `unified://` for raw specs."
        ) from exc
    return preset.spec(split=split)


def _split_source_prefix(shorthand: str) -> tuple[SourceKey | None, str]:
    if shorthand.startswith("hf://"):
        return Source.HF, shorthand[len("hf://") :]
    if shorthand.startswith("unified://"):
        return Source.UNIFIED, shorthand[len("unified://") :]
    if "://" in shorthand:
        from .dataset.source import has_source

        source, body = shorthand.split("://", 1)
        if has_source(source):
            return source, body
    return None, shorthand


def _split_name_and_split(value: str) -> tuple[str, str | None]:
    if ":" not in value:
        return value, None
    name, split = value.rsplit(":", 1)
    return name, split or None


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
    return int(value)
