from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .types.item import AudioView, ImageView, TextView, View

type ViewMap = (
    Mapping[AudioView, Any]
    | Mapping[ImageView, Any]
    | Mapping[TextView, Any]
)


type ViewTransform[ViewT: View] = Callable[[Mapping[ViewT, Any]], Any]


class ViewProvider[ViewT: View](Protocol):
    output: ViewT

    def __call__(self, views: Mapping[ViewT, Any]) -> Any: ...


type Provider = (
    ViewProvider[AudioView]
    | ViewProvider[ImageView]
    | ViewProvider[TextView]
)


@dataclass
class FunctionViewProvider[ViewT: View]:
    output: ViewT
    transform_fn: ViewTransform[ViewT]

    def __post_init__(self) -> None:
        if not callable(self.transform_fn):
            raise TypeError("transform_fn must be callable.")

    def __call__(self, views: Mapping[ViewT, Any]) -> Any:
        return self.transform_fn(views)
