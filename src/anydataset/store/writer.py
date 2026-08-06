"""Public single-process and parallel writer for canonical sample stores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._io.atomic import replace_dir
from .._validation import non_negative_int, optional_positive_int, positive_int
from ._pickle_state import decode_pickle_state, validate_pickle_fields
from .part.dispatch import DatasetFactory, ordered_sample_records, write_dataset_parts
from ..types.item import Modality, Role, Sample, View
from .config import DEFAULT_MAX_SHARD_BYTES, DEFAULT_MAX_SHARD_SAMPLES
from .part.writer import write_sample_records
from .payload.groups import write_payload_groups
from .part.sample_write import explicit_views
from .manifest.schema import normalize_provenance
from .paths import dataset_ready_path

DATASET_WRITER_PICKLE_VERSION = 2

_DATASET_WRITER_PICKLE_FIELDS = frozenset(
    {
        "output_dir",
        "dataset_id",
        "split",
        "views",
        "max_shard_samples",
        "max_shard_bytes",
        "provenance",
        "num_shards",
        "num_workers",
        "prefetch_factor",
    }
)
_DATASET_WRITER_PICKLE_V1_FIELDS = _DATASET_WRITER_PICKLE_FIELDS - {
    "max_shard_bytes"
}
_DATASET_WRITER_LEGACY_REQUIRED_FIELDS = frozenset(
    {"output_dir", "dataset_id", "split", "views", "max_shard_samples"}
)
_DATASET_WRITER_LEGACY_OPTIONAL_FIELDS = (
    _DATASET_WRITER_PICKLE_V1_FIELDS - _DATASET_WRITER_LEGACY_REQUIRED_FIELDS
)


@dataclass
class DatasetWriter:
    output_dir: str | Path
    dataset_id: str | None = None
    split: str | None = None
    views: tuple[tuple[Role, Modality, View], ...] | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES
    max_shard_bytes: int | None = DEFAULT_MAX_SHARD_BYTES
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
        self.max_shard_bytes = optional_positive_int(
            "max_shard_bytes",
            self.max_shard_bytes,
        )
        self.provenance = normalize_provenance(self.provenance)
        self.num_shards = positive_int("num_shards", self.num_shards)
        self.num_workers = non_negative_int("num_workers", self.num_workers)
        self.prefetch_factor = optional_positive_int(
            "prefetch_factor",
            self.prefetch_factor,
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "pickle_schema_version": DATASET_WRITER_PICKLE_VERSION,
            "output_dir": self.output_dir,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "views": self.views,
            "max_shard_samples": self.max_shard_samples,
            "max_shard_bytes": self.max_shard_bytes,
            "provenance": self.provenance,
            "num_shards": self.num_shards,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
        }

    def __setstate__(self, state: object) -> None:
        version, values = _dataset_writer_pickle_state(state)
        if version == 0:
            values = _migrate_dataset_writer_pickle_v0(values)
        elif version == 1:
            values = _migrate_dataset_writer_pickle_v1(values)
        else:
            validate_pickle_fields(
                values,
                kind="DatasetWriter",
                required=_DATASET_WRITER_PICKLE_FIELDS,
            )
        _validate_dataset_writer_pickle_state(values)
        restored = DatasetWriter(
            output_dir=values["output_dir"],
            dataset_id=values["dataset_id"],
            split=values["split"],
            views=values["views"],
            max_shard_samples=values["max_shard_samples"],
            max_shard_bytes=values["max_shard_bytes"],
            provenance=values["provenance"],
            num_shards=values["num_shards"],
            num_workers=values["num_workers"],
            prefetch_factor=values["prefetch_factor"],
        )
        self.__dict__.clear()
        self.__dict__.update(restored.__dict__)

    def write(
        self,
        samples: Iterable[Sample] | None = None,
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
            max_shard_bytes=self.max_shard_bytes,
            num_shards=self.num_shards,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            provenance=provenance,
            dataset_factory=dataset_factory,
        )

    def _write_single(self, samples: Any) -> Path:
        return replace_dir(
            self.output_dir,
            lambda tmp: self._write_to_tmp(tmp, samples),
        )

    def _write_to_tmp(self, root: Path, samples: Any) -> Path:
        dataset_id, provenance = self._metadata()
        sample_count, written_views = write_sample_records(
            root,
            ordered_sample_records(samples),
            dataset_id=dataset_id,
            split=self.split,
            views=self.views,
            max_shard_samples=self.max_shard_samples,
            max_shard_bytes=self.max_shard_bytes,
            provenance=provenance,
        )
        write_payload_groups(root, written_views, sample_count)
        dataset_ready_path(root).touch()
        return root

    def _metadata(self) -> tuple[str, Mapping[str, str]]:
        if self.dataset_id is None or self.provenance is None:
            raise RuntimeError("writer metadata was not initialized.")
        return self.dataset_id, self.provenance


def _migrate_dataset_writer_pickle_v0(
    state: dict[str, Any],
) -> dict[str, Any]:
    validate_pickle_fields(
        state,
        kind="DatasetWriter",
        required=_DATASET_WRITER_LEGACY_REQUIRED_FIELDS,
        optional=_DATASET_WRITER_LEGACY_OPTIONAL_FIELDS,
    )
    values = dict(state)
    values.setdefault("provenance", None)
    values.setdefault("num_shards", 1)
    values.setdefault("num_workers", 0)
    values.setdefault("prefetch_factor", None)
    values.setdefault("max_shard_bytes", None)
    return values


def _migrate_dataset_writer_pickle_v1(
    state: dict[str, Any],
) -> dict[str, Any]:
    validate_pickle_fields(
        state,
        kind="DatasetWriter",
        required=_DATASET_WRITER_PICKLE_V1_FIELDS,
    )
    return {**state, "max_shard_bytes": None}


def _dataset_writer_pickle_state(state: object) -> tuple[int, dict[str, Any]]:
    if (
        isinstance(state, dict)
        and type(state.get("pickle_schema_version")) is int
        and state.get("pickle_schema_version") == 1
    ):
        values = dict(state)
        values.pop("pickle_schema_version")
        return 1, values
    legacy, values = decode_pickle_state(
        state,
        kind="DatasetWriter",
        current_version=DATASET_WRITER_PICKLE_VERSION,
    )
    return (0 if legacy else DATASET_WRITER_PICKLE_VERSION), values


def _validate_dataset_writer_pickle_state(
    state: Mapping[str, Any],
) -> None:
    output_dir = state["output_dir"]
    if not isinstance(output_dir, (str, Path)):
        raise TypeError(
            "DatasetWriter pickle field 'output_dir' must be a string or Path."
        )
    dataset_id = state["dataset_id"]
    if dataset_id is not None and not isinstance(dataset_id, str):
        raise TypeError(
            "DatasetWriter pickle field 'dataset_id' must be a string or None."
        )
    split = state["split"]
    if split is not None and not isinstance(split, str):
        raise TypeError(
            "DatasetWriter pickle field 'split' must be a string or None."
        )
