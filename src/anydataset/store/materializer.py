from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..types.item import (
    AudioItem,
    AudioView,
    ImageItem,
    ImageView,
    Item,
    Sample,
    TextItem,
    TextView,
    View,
)
from ..view import Provider, ViewProvider
from .writer import DatasetWriter


@dataclass
class ViewMaterializer:
    output_dir: str | Path
    dataset_id: str
    split: str | None = None
    copy_inputs: bool = False

    def write(self, dataset: Iterable[Sample], provider: Provider) -> Path:
        return DatasetWriter(
            self.output_dir,
            dataset_id=self.dataset_id,
            split=self.split,
        ).write(self._samples(dataset, provider))

    def _samples(self, dataset: Iterable[Sample], provider: Provider):
        for sample in dataset:
            yield {
                ref: _with_provider(item, provider, copy_inputs=self.copy_inputs)
                for ref, item in sample.items()
            }


def _with_provider(
    item: Item,
    provider: Provider,
    *,
    copy_inputs: bool,
) -> Item:
    match item:
        case AudioItem():
            provider = _audio_provider(provider)
            return _with_view(
                item,
                provider.output,
                provider(item.views),
                copy_inputs=copy_inputs,
            )
        case ImageItem():
            provider = _image_provider(provider)
            return _with_view(
                item,
                provider.output,
                provider(item.views),
                copy_inputs=copy_inputs,
            )
        case TextItem():
            provider = _text_provider(provider)
            return _with_view(
                item,
                provider.output,
                provider(item.views),
                copy_inputs=copy_inputs,
            )
    raise TypeError(f"Unsupported materializer item: {type(item).__name__}.")


def _with_view(
    item: Item,
    view: View,
    value: Any,
    *,
    copy_inputs: bool,
) -> Item:
    match item:
        case AudioItem():
            if not isinstance(view, AudioView):
                raise TypeError("audio item materializer output must be an AudioView.")
            return AudioItem(
                views=_views(item.views, view, value, copy_inputs=copy_inputs),
                meta=item.meta,
            )
        case ImageItem():
            if not isinstance(view, ImageView):
                raise TypeError("image item materializer output must be an ImageView.")
            return ImageItem(
                views=_views(item.views, view, value, copy_inputs=copy_inputs),
                meta=item.meta,
            )
        case TextItem():
            if not isinstance(view, TextView):
                raise TypeError("text item materializer output must be a TextView.")
            return TextItem(
                views=_views(item.views, view, value, copy_inputs=copy_inputs),
                meta=item.meta,
            )
    raise TypeError(f"Unsupported materializer item: {type(item).__name__}.")


def _audio_provider(provider: Provider) -> ViewProvider[AudioView]:
    if not isinstance(provider.output, AudioView):
        raise TypeError("audio item materializer output must be an AudioView.")
    return cast(ViewProvider[AudioView], provider)


def _image_provider(provider: Provider) -> ViewProvider[ImageView]:
    if not isinstance(provider.output, ImageView):
        raise TypeError("image item materializer output must be an ImageView.")
    return cast(ViewProvider[ImageView], provider)


def _text_provider(provider: Provider) -> ViewProvider[TextView]:
    if not isinstance(provider.output, TextView):
        raise TypeError("text item materializer output must be a TextView.")
    return cast(ViewProvider[TextView], provider)


def _views[ViewT](
    original: Mapping[ViewT, Any],
    view: ViewT,
    value: Any,
    *,
    copy_inputs: bool,
) -> dict[ViewT, Any]:
    values = dict(original) if copy_inputs else {}
    values[view] = value
    return values
