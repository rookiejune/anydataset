from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from anydataset.datasets.base import DatasetAdapter

if TYPE_CHECKING:
    from anydataset.api.cache import CacheManifest
    from anydataset.api.spec import DatasetSpec


class HuggingFaceDataset(DatasetAdapter):
    def prepare(self, spec: DatasetSpec, cache: CacheManifest) -> Any:
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
            cache_dir=str(cache.cache_path),
            **load_kwargs,
        )

    def iter_samples(self, manifest: Any) -> Iterator[dict]:
        for row in manifest:
            yield dict(row)

    def iter_indexed_samples(
        self,
        manifest: Any,
        num_shards: int = 1,
        shard_id: int = 0,
    ) -> Iterator[tuple[int, dict]]:
        yield from super().iter_indexed_samples(
            manifest,
            num_shards=num_shards,
            shard_id=shard_id,
        )
