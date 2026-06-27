from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .._sharding import validate_shard
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
from .parts import DatasetPartWriter, commit_store_parts
from .writer import DEFAULT_MAX_SHARD_SAMPLES, DatasetWriter


@dataclass
class ViewMaterializer:
    output_dir: str | Path
    dataset_id: str
    split: str | None = None
    copy_inputs: bool = False
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES

    def write(self, dataset: Iterable[Sample], provider: Provider) -> Path:
        return DatasetWriter(
            self.output_dir,
            dataset_id=self.dataset_id,
            split=self.split,
            max_shard_samples=self.max_shard_samples,
        ).write(self._samples(dataset, provider))

    def write_part(
        self,
        dataset: Any,
        provider: Provider,
        *,
        parts_dir: str | Path,
        num_shards: int,
        shard_id: int,
    ) -> Path:
        return DatasetPartWriter(
            Path(parts_dir) / f"part-{shard_id:05d}",
            dataset_id=self.dataset_id,
            split=self.split,
            shard_id=shard_id,
            num_shards=num_shards,
            max_shard_samples=self.max_shard_samples,
        ).write(
            (
                (index, _with_providers(sample, provider, copy_inputs=self.copy_inputs))
                for index, sample in iter_indexed_shard(dataset, num_shards, shard_id)
            )
        )

    def commit_parts(self, parts_dir: str | Path) -> Path:
        return commit_store_parts(
            self.output_dir,
            parts_dir,
            dataset_id=self.dataset_id,
            split=self.split,
        )

    def _samples(self, dataset: Iterable[Sample], provider: Provider):
        for sample in dataset:
            yield _with_providers(sample, provider, copy_inputs=self.copy_inputs)


def iter_indexed_shard(
    dataset: Any,
    num_shards: int,
    shard_id: int,
) -> Iterator[tuple[int, Sample]]:
    validate_shard(num_shards, shard_id)
    iter_indexed = getattr(dataset, "iter_indexed_shard", None)
    if callable(iter_indexed):
        yield from iter_indexed(num_shards, shard_id)
        return

    iter_shard = getattr(dataset, "iter_shard", None)
    if callable(iter_shard):
        for local_index, sample in enumerate(iter_shard(num_shards, shard_id)):
            yield shard_id + local_index * num_shards, sample
        return

    native_shard = getattr(dataset, "shard", None)
    if callable(native_shard):
        for local_index, sample in enumerate(
            native_shard(num_shards=num_shards, index=shard_id)
        ):
            yield shard_id + local_index * num_shards, sample
        return

    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        for index in range(shard_id, len(dataset), num_shards):
            yield index, dataset[index]
        return

    for index, sample in enumerate(dataset):
        if index % num_shards == shard_id:
            yield index, sample


def _with_providers(
    sample: Sample,
    provider: Provider,
    *,
    copy_inputs: bool,
) -> Sample:
    return {
        ref: _with_provider(item, provider, copy_inputs=copy_inputs)
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
