from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .._io.atomic import replace_dir
from .._sharding import validate_shard
from .._validation import positive_int, validate_path_segment
from ..types.item import Modality, Role, Sample, View
from ._config import DEFAULT_MAX_SHARD_SAMPLES
from ._sample_write import (
    explicit_views,
    sample_id,
    sample_id_prefix,
    sample_manifest_entry,
    sample_view_refs,
    sample_view_value,
    validate_sample,
    validate_view_sets,
    view_path,
)
from .jsonio import write_json
from .manifest import (
    DatasetManifest,
    STORE_SCHEMA_VERSION,
    dataset_manifest_dict,
    normalize_provenance,
)
from .manifestio import sample_manifest_writer
from .paths import dataset_json_path, dataset_ready_path
from .viewwriter import ViewWriter

IndexedSample = tuple[int, Sample]


def write_indexed_samples(
    root: Path,
    samples: Iterable[IndexedSample],
    *,
    dataset_id: str,
    split: str | None,
    views: tuple[tuple[Role, Modality, View], ...] | None,
    max_shard_samples: int,
    provenance: Mapping[str, str],
    shard_prefix: str = "",
) -> tuple[int, tuple[tuple[Role, Modality, View], ...]]:
    sinks: dict[tuple[Role, Modality, View], ViewWriter] = {}
    sample_views: dict[tuple[Role, Modality], frozenset[View]] = {}
    sample_manifest = sample_manifest_writer(root)
    sample_count = 0
    previous_index: int | None = None
    prefix = sample_id_prefix(dataset_id)

    try:
        for sample_index, sample in samples:
            if previous_index is not None and sample_index <= previous_index:
                raise ValueError("Materialized sample indexes must be increasing.")
            previous_index = sample_index
            if not isinstance(sample, Mapping):
                raise TypeError("Store writers expect Sample mappings.")
            current_sample_id = sample_id(prefix, sample_index)
            validate_sample(sample)
            selected = views if views is not None else sample_view_refs(sample)
            if not selected:
                raise ValueError(f"Sample {current_sample_id} has no views.")
            if views is None:
                validate_view_sets(sample, sample_views, current_sample_id)
            sample_manifest.write(
                sample_manifest_entry(sample, current_sample_id, sample_index)
            )
            sample_count += 1
            for view in selected:
                value = sample_view_value(sample, view)
                if value is None:
                    if views is not None:
                        raise KeyError(
                            f"Sample {current_sample_id} is missing view {view_path(view)}."
                        )
                    continue
                sink = sinks.get(view)
                if sink is None:
                    sink = ViewWriter(
                        root=root,
                        view=view,
                        max_shard_samples=max_shard_samples,
                        shard_prefix=shard_prefix,
                    )
                    sinks[view] = sink
                sink.write(sample_index, value)

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            schema_version=STORE_SCHEMA_VERSION,
            split=split,
            sample_count=sample_count,
            provenance=provenance,
        )
        write_json(dataset_json_path(root), dataset_manifest_dict(manifest))
        sample_manifest.close()
        for sink in sinks.values():
            sink.close()
        return sample_count, tuple(sinks)
    except Exception:
        sample_manifest.abort()
        for sink in sinks.values():
            sink.abort()
        raise


def write_dataset_part(
    root: Path,
    samples: Iterable[IndexedSample],
    *,
    dataset_id: str,
    split: str | None,
    views: tuple[tuple[Role, Modality, View], ...] | None,
    max_shard_samples: int,
    provenance: Mapping[str, str],
    shard_id: int,
    num_shards: int,
    shard_prefix: str,
) -> tuple[int, tuple[tuple[Role, Modality, View], ...]]:
    result = write_indexed_samples(
        root,
        samples,
        dataset_id=dataset_id,
        split=split,
        views=views,
        max_shard_samples=max_shard_samples,
        provenance=provenance,
        shard_prefix=shard_prefix,
    )
    sample_count, _written_views = result
    write_json(
        part_json_path(root),
        {
            "dataset_id": dataset_id,
            "split": split,
            "num_shards": num_shards,
            "shard_id": shard_id,
            "sample_count": sample_count,
        },
    )
    return result


@dataclass
class DatasetPartWriter:
    output_dir: str | Path
    dataset_id: str
    shard_id: int
    num_shards: int
    split: str | None = None
    views: tuple[tuple[Role, Modality, View], ...] | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES
    shard_prefix: str | None = None
    provenance: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        validate_shard(self.num_shards, self.shard_id)
        self.views = explicit_views(self.views)
        self.max_shard_samples = positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )
        self.provenance = normalize_provenance(self.provenance)

    def write(self, samples: Iterable[IndexedSample]) -> Path:
        return replace_dir(
            self.output_dir, lambda tmp: self._write_to_tmp(tmp, samples)
        )

    def _write_to_tmp(self, root: Path, samples: Iterable[IndexedSample]) -> Path:
        provenance = self.provenance
        if provenance is None:
            raise RuntimeError("part writer provenance was not initialized.")
        write_dataset_part(
            root,
            samples,
            dataset_id=self.dataset_id,
            split=self.split,
            views=self.views,
            max_shard_samples=self.max_shard_samples,
            provenance=provenance,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            shard_prefix=self._shard_prefix(),
        )
        dataset_ready_path(root).touch()
        return root

    def _shard_prefix(self) -> str:
        if self.shard_prefix is not None:
            return self.shard_prefix
        return f"part-{self.shard_id:05d}-"


@dataclass
class DatasetFragmentWriter:
    output_dir: str | Path
    dataset_id: str
    fragment_id: str
    split: str | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES
    provenance: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        validate_path_segment("fragment_id", self.fragment_id)
        self.max_shard_samples = positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )
        self.provenance = normalize_provenance(self.provenance)

    def write(self, samples: Sequence[IndexedSample]) -> Path:
        if not samples:
            raise ValueError("DatasetFragmentWriter.write requires samples.")
        ordered = tuple(sorted(samples, key=lambda item: item[0]))
        indexes = tuple(index for index, _ in ordered)
        if len(set(indexes)) != len(indexes):
            raise ValueError("Dataset fragment sample indexes must be unique.")
        return replace_dir(
            self.output_dir,
            lambda tmp: self._write_to_tmp(tmp, ordered, indexes),
        )

    def _write_to_tmp(
        self,
        root: Path,
        samples: tuple[IndexedSample, ...],
        indexes: tuple[int, ...],
    ) -> Path:
        provenance = self.provenance
        if provenance is None:
            raise RuntimeError("fragment writer provenance was not initialized.")
        write_dataset_part(
            root,
            samples,
            dataset_id=self.dataset_id,
            split=self.split,
            views=None,
            max_shard_samples=self.max_shard_samples,
            provenance=provenance,
            shard_id=0,
            num_shards=1,
            shard_prefix=f"fragment-{self.fragment_id}-",
        )
        write_json(
            fragment_json_path(root),
            {
                "dataset_id": self.dataset_id,
                "split": self.split,
                "fragment_id": self.fragment_id,
                "sample_count": len(indexes),
                "sample_indexes": list(indexes),
            },
        )
        dataset_ready_path(root).touch()
        return root


def part_json_path(root: str | Path) -> Path:
    return Path(root) / "part.json"


def fragment_json_path(root: str | Path) -> Path:
    return Path(root) / "fragment.json"
