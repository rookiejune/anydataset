from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..._runtime.sharding import iter_map_style_shard
from ... import types


class HuggingFaceSource:
    def prepare(self, spec: types.Spec, cache_path: Path) -> Any:
        return _prepare_hf(spec, cache_path)

    def iter_shard(
        self,
        dataset: Any,
        *,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Any]]:
        yield from iter_map_style_shard(dataset, num_shards, shard_id)


class HuggingFaceDiskSource:
    def prepare(self, spec: types.Spec, cache_path: Path) -> Any:
        return _prepare_hf_disk(spec)

    def iter_shard(
        self,
        dataset: Any,
        *,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Any]]:
        yield from iter_map_style_shard(dataset, num_shards, shard_id)


def _prepare_hf(spec: types.Spec, cache_path: Path) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets support requires `pip install anydataset[huggingface]`."
        ) from exc

    if spec.split is None:
        raise ValueError("huggingface source requires Spec.split.")
    load_kwargs = dict(spec.load_options)
    if load_kwargs.get("streaming") is True:
        raise ValueError(
            "Hugging Face streaming is not supported. Omit `streaming` or set "
            "`streaming=False`, or use Source.HF_DISK for local map-style data."
        )
    config_name = load_kwargs.pop("config_name", None)
    if config_name is not None:
        if "name" in load_kwargs:
            raise ValueError("Use either `config_name` or `name`, not both.")
        load_kwargs["name"] = config_name
    return load_dataset(
        spec.path,
        split=spec.split,
        cache_dir=str(cache_path),
        **load_kwargs,
    )


def _prepare_hf_disk(spec: types.Spec) -> Any:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets support requires `pip install anydataset[huggingface]`."
        ) from exc

    dataset = load_from_disk(spec.path, **dict(spec.load_options))
    if not isinstance(dataset, DatasetDict):
        if spec.split is not None:
            raise ValueError(
                "huggingface_disk split is only supported for DatasetDict inputs."
            )
        return dataset

    if spec.split is None:
        raise ValueError("huggingface_disk DatasetDict specs must set split.")
    if spec.split not in dataset:
        raise KeyError(f"HuggingFace disk dataset is missing split {spec.split!r}.")
    return dataset[spec.split]
