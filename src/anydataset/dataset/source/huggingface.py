from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ... import types


class HuggingFaceSource:
    def prepare(self, spec: types.Spec, cache_path: Path) -> Any:
        return prepare_hf(spec, cache_path)


class HuggingFaceDiskSource:
    def prepare(self, spec: types.Spec, cache_path: Path) -> Any:
        return prepare_hf_disk(spec)

    def iter_indexed_shard(
        self,
        dataset: Any,
        *,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Any]]:
        for sample_index in range(shard_id, len(dataset), num_shards):
            yield sample_index, dataset[sample_index]


def prepare_hf(spec: types.Spec, cache_path: Path) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets support requires `pip install anydataset[huggingface]`."
        ) from exc

    split = spec.split or "train"
    load_kwargs = dict(spec.load_options)
    config_name = load_kwargs.pop("config_name", None)
    if config_name is not None:
        if "name" in load_kwargs:
            raise ValueError("Use either `config_name` or `name`, not both.")
        load_kwargs["name"] = config_name
    return load_dataset(
        spec.path,
        split=split,
        cache_dir=str(cache_path),
        **load_kwargs,
    )


def prepare_hf_disk(spec: types.Spec) -> Any:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets support requires `pip install anydataset[huggingface]`."
        ) from exc

    dataset = load_from_disk(spec.path, **dict(spec.load_options))
    if not isinstance(dataset, DatasetDict):
        return dataset

    if spec.split is None:
        raise ValueError("huggingface_disk DatasetDict specs must set split.")
    if spec.split not in dataset:
        raise KeyError(f"HuggingFace disk dataset is missing split {spec.split!r}.")
    return dataset[spec.split]
