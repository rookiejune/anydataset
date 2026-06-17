from __future__ import annotations

from typing import Any, Iterator

from .base import DatasetAdapter
from anydatasets.cache import CacheManifest
from anydatasets.registry import DatasetSpec


class HuggingFaceAdapter(DatasetAdapter):
    def prepare(self, spec: DatasetSpec, cache: CacheManifest) -> Any:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "HuggingFace datasets support requires `pip install anydatasets[huggingface]`."
            ) from exc

        split = spec.split or "train"
        return load_dataset(
            spec.path,
            split=split,
            cache_dir=str(cache.cache_path),
            **dict(spec.options),
        )

    def iter_samples(self, manifest: Any) -> Iterator[dict]:
        for row in manifest:
            yield dict(row)
