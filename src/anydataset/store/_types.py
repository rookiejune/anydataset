"""Shared materializer provider and item helpers.

The module keeps provider type aliases and view-to-modality conversion logic
used by view, modality, and batch materialization helpers.
"""

from __future__ import annotations

from typing import Any

from ..types.item import AudioView, ImageView, Modality, TextView, View
from ..view import ModalityProvider, Provider

type ModalityProviderLike = (
    ModalityProvider[AudioView]
    | ModalityProvider[ImageView]
    | ModalityProvider[TextView]
)
type MaterializerProvider = Provider | ModalityProviderLike


def output_modality(view: View) -> Modality:
    if isinstance(view, AudioView):
        return Modality.AUDIO
    if isinstance(view, ImageView):
        return Modality.IMAGE
    if isinstance(view, TextView):
        return Modality.TEXT
    raise TypeError("materializer output must be an AudioView, ImageView, or TextView.")


def views[ViewT](view: ViewT, value: Any) -> dict[ViewT, Any]:
    return {view: value}
