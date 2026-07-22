from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .._io.atomic import replace_dir
from .._validation import positive_int
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
    STORE_SCHEMA_VERSION,
    DatasetManifest,
    dataset_manifest_dict,
)
from .manifestio import sample_manifest_writer
from .paths import dataset_json_path, dataset_ready_path
from .viewwriter import ViewWriter


@dataclass
class DatasetWriter:
    output_dir: str | Path
    dataset_id: str
    split: str | None = None
    views: tuple[tuple[Role, Modality, View], ...] | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.views = explicit_views(self.views)
        self.max_shard_samples = positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )

    def write(self, samples: Iterable[Sample]) -> Path:
        return replace_dir(
            self.output_dir, lambda tmp: self._write_to_tmp(tmp, samples)
        )

    def _write_to_tmp(self, root: Path, samples: Iterable[Sample]) -> Path:
        sinks: dict[tuple[Role, Modality, View], ViewWriter] = {}
        sample_views: dict[tuple[Role, Modality], frozenset[View]] = {}
        sample_manifest = sample_manifest_writer(root)
        sample_count = 0
        prefix = sample_id_prefix(self.dataset_id)

        try:
            for index, sample in enumerate(samples):
                if not isinstance(sample, Mapping):
                    raise TypeError("DatasetWriter.write expects Sample mappings.")
                current_sample_id = sample_id(prefix, index)
                validate_sample(sample)
                views = (
                    self.views if self.views is not None else sample_view_refs(sample)
                )
                if not views:
                    raise ValueError(f"Sample {current_sample_id} has no views.")
                if self.views is None:
                    validate_view_sets(sample, sample_views, current_sample_id)
                sample_manifest.write(
                    sample_manifest_entry(sample, current_sample_id, index)
                )
                sample_count += 1
                for view in views:
                    value = sample_view_value(sample, view)
                    if value is None:
                        if self.views is not None:
                            raise KeyError(
                                f"Sample {current_sample_id} is missing view {view_path(view)}."
                            )
                        continue
                    sink = sinks.get(view)
                    if sink is None:
                        sink = ViewWriter(
                            root=root,
                            view=view,
                            max_shard_samples=self.max_shard_samples,
                        )
                        sinks[view] = sink
                    sink.write(index, value)

            manifest = DatasetManifest(
                dataset_id=self.dataset_id,
                schema_version=STORE_SCHEMA_VERSION,
                split=self.split,
                sample_count=sample_count,
            )
            write_json(dataset_json_path(root), dataset_manifest_dict(manifest))
            sample_manifest.close()
            for sink in sinks.values():
                sink.close()
            dataset_ready_path(root).touch()
            return root
        except Exception:
            sample_manifest.abort()
            for sink in sinks.values():
                sink.abort()
            raise
