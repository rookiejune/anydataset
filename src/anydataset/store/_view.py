"""Materialize new views for existing sample modalities.

The module validates provider output types and returns sparse samples containing
only the generated views for matching input modalities.
"""

from __future__ import annotations

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
from ._types import output_modality, views


def with_view_provider(
    sample: Sample,
    provider: Provider,
) -> Sample:
    modality = output_modality(provider.output)
    return {
        ref: _with_provider(item, provider)
        for ref, item in sample.items()
        if ref[1] is modality
    }


def _with_provider(
    item: Item,
    provider: Provider,
) -> Item:
    match item:
        case AudioItem():
            provider = _audio_provider(provider)
            return with_view(item, provider.output, provider(item.views))
        case ImageItem():
            provider = _image_provider(provider)
            return with_view(item, provider.output, provider(item.views))
        case TextItem():
            provider = _text_provider(provider)
            return with_view(item, provider.output, provider(item.views))
    raise TypeError(f"Unsupported materializer item: {type(item).__name__}.")


def with_view(
    item: Item,
    view: View,
    value: Any,
) -> Item:
    match item:
        case AudioItem():
            if not isinstance(view, AudioView):
                raise TypeError("audio item materializer output must be an AudioView.")
            return AudioItem(
                views=views(view, value),
                meta=item.meta,
            )
        case ImageItem():
            if not isinstance(view, ImageView):
                raise TypeError("image item materializer output must be an ImageView.")
            return ImageItem(
                views=views(view, value),
                meta=item.meta,
            )
        case TextItem():
            if not isinstance(view, TextView):
                raise TypeError("text item materializer output must be a TextView.")
            return TextItem(
                views=views(view, value),
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
