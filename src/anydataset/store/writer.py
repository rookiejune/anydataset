"""Public single-process and parallel writer for canonical sample stores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._io.atomic import replace_dir
from .._validation import non_negative_int, optional_positive_int, positive_int
from ..dataset.write import DatasetFactory, ordered_samples, write_dataset_parts
from ..types.item import Modality, Role, Sample, View
from .config import DEFAULT_MAX_SHARD_SAMPLES
from .part.writer import write_indexed_samples
from .payload.groups import write_payload_groups
from .part.sample_write import explicit_views
from .manifest.schema import normalize_provenance
from .paths import dataset_ready_path


@dataclass
class DatasetWriter:
    output_dir: str | Path
    dataset_id: str | None = None
    split: str | None = None
    views: tuple[tuple[Role, Modality, View], ...] | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES
    provenance: Mapping[str, str] | None = None
    num_shards: int = 1
    num_workers: int = 0
    prefetch_factor: int | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.dataset_id is None:
            self.dataset_id = self.output_dir.expanduser().name or "dataset"
        self.views = explicit_views(self.views)
        self.max_shard_samples = positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )
        self.provenance = normalize_provenance(self.provenance)
        self.num_shards = positive_int("num_shards", self.num_shards)
        self.num_workers = non_negative_int("num_workers", self.num_workers)
        self.prefetch_factor = optional_positive_int(
            "prefetch_factor",
            self.prefetch_factor,
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if "provenance" not in state:
            self.provenance = None
        if "num_shards" not in state:
            self.num_shards = 1
        if "num_workers" not in state:
            self.num_workers = 0
        if "prefetch_factor" not in state:
            self.prefetch_factor = None
        self.__post_init__()

    def write(
        self,
        samples: Any | None = None,
        *,
        dataset_factory: DatasetFactory | None = None,
    ) -> Path:
        if dataset_factory is None:
            if samples is None:
                raise TypeError("write requires samples or dataset_factory.")
            if self.num_shards > 1 or self.num_workers > 0:
                raise TypeError(
                    "dataset_factory is required when num_shards or num_workers "
                    "is greater than one."
                )
            return self._write_single(samples)

        if samples is not None:
            raise TypeError(
                "write accepts either samples or dataset_factory, not both."
            )
        if self.num_shards == 1 and self.num_workers == 0:
            return self._write_single(dataset_factory())
        dataset_id, provenance = self._metadata()
        return write_dataset_parts(
            self.output_dir,
            dataset_id=dataset_id,
            split=self.split,
            views=self.views,
            max_shard_samples=self.max_shard_samples,
            num_shards=self.num_shards,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            provenance=provenance,
            dataset_factory=dataset_factory,
        )

    def _write_single(self, samples: Any) -> Path:
        return replace_dir(
            self.output_dir,
            lambda tmp: self._write_to_tmp(tmp, ordered_samples(samples)),
        )

    def _write_to_tmp(self, root: Path, samples: Iterable[Sample]) -> Path:
        dataset_id, provenance = self._metadata()
        sample_count, written_views = write_indexed_samples(
            root,
            enumerate(samples),
            dataset_id=dataset_id,
            split=self.split,
            views=self.views,
            max_shard_samples=self.max_shard_samples,
            provenance=provenance,
        )
        write_payload_groups(root, written_views, sample_count)
        dataset_ready_path(root).touch()
        return root

    def _metadata(self) -> tuple[str, Mapping[str, str]]:
        if self.dataset_id is None or self.provenance is None:
            raise RuntimeError("writer metadata was not initialized.")
        return self.dataset_id, self.provenance
