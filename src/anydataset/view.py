from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .dataset.collate import Batch
from .types.item import AudioView, ImageView, Reference, Role, TextView, View

type ViewMap = (
    Mapping[AudioView, Any]
    | Mapping[ImageView, Any]
    | Mapping[TextView, Any]
)


type ViewTransform[ViewT: View] = Callable[[Mapping[ViewT, Any]], Any]
type BatchOutput = Sequence[Any] | Mapping[Reference, Sequence[Any]]
type BatchViewTransform = Callable[[Batch], BatchOutput]
type ModalityTransform = Callable[[ViewMap], Any]
type BatchModalityTransform = Callable[[Batch], BatchOutput]


class ViewProvider[ViewT: View](Protocol):
    output: ViewT

    def __call__(self, views: Mapping[ViewT, Any]) -> Any: ...


class BatchViewProvider[ViewT: View](Protocol):
    output: ViewT

    def __call__(self, views: Mapping[ViewT, Any]) -> Any: ...

    def call_batch(self, batch: Batch) -> BatchOutput: ...


class ModalityProvider[ViewT: View](Protocol):
    output: ViewT
    reference_role: Role | None

    def __call__(self, views: ViewMap) -> Any: ...


class BatchModalityProvider[ViewT: View](Protocol):
    output: ViewT
    reference_role: Role | None

    def __call__(self, views: ViewMap) -> Any: ...

    def call_batch(self, batch: Batch) -> BatchOutput: ...


type Provider = (
    ViewProvider[AudioView]
    | ViewProvider[ImageView]
    | ViewProvider[TextView]
)


@dataclass
class FunctionViewProvider[ViewT: View]:
    output: ViewT
    transform_fn: ViewTransform[ViewT]
    batch_transform_fn: BatchViewTransform | None = None

    def __post_init__(self) -> None:
        if not callable(self.transform_fn):
            raise TypeError("transform_fn must be callable.")
        if self.batch_transform_fn is not None and not callable(self.batch_transform_fn):
            raise TypeError("batch_transform_fn must be callable.")

    def __call__(self, views: Mapping[ViewT, Any]) -> Any:
        return self.transform_fn(views)

    def call_batch(self, batch: Batch) -> BatchOutput:
        if self.batch_transform_fn is None:
            raise TypeError("batch_transform_fn is not configured.")
        return self.batch_transform_fn(batch)


@dataclass
class FunctionModalityProvider[ViewT: View]:
    output: ViewT
    transform_fn: ModalityTransform
    batch_transform_fn: BatchModalityTransform | None = None

    def __post_init__(self) -> None:
        if not callable(self.transform_fn):
            raise TypeError("transform_fn must be callable.")
        if self.batch_transform_fn is not None and not callable(self.batch_transform_fn):
            raise TypeError("batch_transform_fn must be callable.")

    def __call__(self, views: ViewMap) -> Any:
        return self.transform_fn(views)

    def call_batch(self, batch: Batch) -> BatchOutput:
        if self.batch_transform_fn is None:
            raise TypeError("batch_transform_fn is not configured.")
        return self.batch_transform_fn(batch)
