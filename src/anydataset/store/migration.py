from __future__ import annotations

import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .._io.atomic import replace_dir
from .._io.parquet import parquet_schema, pyarrow
from ..types.item import Modality, Role, View
from ._integrity import validate_store_payloads
from .jsonio import read_json, write_json
from .manifest import (
    DatasetManifest,
    STORE_SCHEMA_VERSION,
    SampleManifestEntry,
    ViewManifestEntry,
    dataset_manifest_dict,
    view_from_dict,
)
from .manifestio import (
    read_samples_manifest,
    sample_manifest_writer,
    view_manifest_writer,
)
from .paths import (
    dataset_json_path,
    dataset_ready_path,
    samples_parquet_path,
    view_manifest_parquet_path,
    view_ready_path,
    view_shard_path,
)
from .reader import read_store_dataset, read_store_views

_V1_DATASET_FIELDS = frozenset({"dataset_id", "sample_count", "split"})
_V1_VIEW_SCHEMA = (
    ("modality", "string"),
    ("role", "string"),
    ("view", "string"),
    ("sample_id", "string"),
    ("shard", "string"),
    ("key", "string"),
)


def migrate_store(source: str | Path, output: str | Path) -> Path:
    source_root = Path(source).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    _validate_paths(source_root, output_root)
    manifest = _v1_manifest(source_root)
    views = read_store_views(source_root)

    def write(root: Path) -> Path:
        samples, ref_counts = _write_samples(source_root, root, manifest.sample_count)
        for view in views:
            shards: set[str] = set()
            writer = view_manifest_writer(root, view)
            try:
                for entry in _v1_view_entries(
                    source_root,
                    view,
                    samples,
                    ref_counts.get(view[:2], 0),
                    shards,
                ):
                    writer.write(entry)
                writer.close()
            except Exception:
                writer.abort()
                raise
            _copy_shards(source_root, root, view, shards)
            view_ready_path(root, view).touch()

        write_json(dataset_json_path(root), dataset_manifest_dict(manifest))
        dataset_ready_path(root).touch()
        read_store_dataset(root, preload=True)
        validate_store_payloads((root,))
        return root

    return replace_dir(output_root, write)


def _validate_paths(source: Path, output: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("Store migration output must differ from its source.")
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Store migration output cannot be inside its source.")
    if not dataset_ready_path(source).is_file():
        raise ValueError(f"Store dataset is not ready: {source}")
    if not dataset_json_path(source).is_file():
        raise FileNotFoundError(dataset_json_path(source))
    if not samples_parquet_path(source).is_file():
        raise FileNotFoundError(samples_parquet_path(source))


def _v1_manifest(root: Path) -> DatasetManifest:
    data = read_json(dataset_json_path(root))
    if not isinstance(data, Mapping):
        raise ValueError("Store dataset manifest must be a JSON object.")
    version = data.get("schema_version")
    if type(version) is int and version == STORE_SCHEMA_VERSION:
        raise ValueError("Store already uses schema_version 2.")
    if "schema_version" in data and (type(version) is not int or version != 1):
        raise ValueError(f"Unsupported source store schema_version: {version!r}.")
    fields = frozenset(data) - {"schema_version"}
    if fields != _V1_DATASET_FIELDS:
        missing = _V1_DATASET_FIELDS - fields
        if missing:
            raise ValueError(
                f"Store schema v1 manifest is missing field {min(missing)!r}."
            )
        unsupported = fields - _V1_DATASET_FIELDS
        raise ValueError(
            f"Store schema v1 manifest has unsupported field {min(unsupported)!r}."
        )
    dataset_id = data["dataset_id"]
    if not isinstance(dataset_id, str):
        raise ValueError("Store dataset_id must be a string.")
    sample_count = data["sample_count"]
    if type(sample_count) is not int or sample_count < 0:
        raise ValueError("Store sample_count must be a non-negative integer.")
    split = data["split"]
    if split is not None and not isinstance(split, str):
        raise ValueError("Store split must be a string or None.")
    return DatasetManifest(
        dataset_id=dataset_id,
        sample_count=sample_count,
        schema_version=STORE_SCHEMA_VERSION,
        split=split,
    )


def _write_samples(
    source: Path,
    output: Path,
    expected_count: int,
) -> tuple[
    dict[str, tuple[int, frozenset[tuple[Role, Modality]]]],
    dict[tuple[Role, Modality], int],
]:
    samples: dict[str, tuple[int, frozenset[tuple[Role, Modality]]]] = {}
    ref_counts: dict[tuple[Role, Modality], int] = {}
    writer = sample_manifest_writer(output)
    count = 0
    try:
        for count, sample in enumerate(read_samples_manifest(source), start=1):
            _validate_sample(sample, count - 1, samples)
            refs = frozenset(ref for ref, _meta in sample.items)
            samples[sample.sample_id] = sample.sample_index, refs
            for ref in refs:
                ref_counts[ref] = ref_counts.get(ref, 0) + 1
            writer.write(sample)
        if count != expected_count:
            raise ValueError(
                "Store schema v1 sample manifest row count must match "
                "dataset.json sample_count."
            )
        writer.close()
    except Exception:
        writer.abort()
        raise
    return samples, ref_counts


def _validate_sample(
    sample: SampleManifestEntry,
    expected_index: int,
    samples: Mapping[str, object],
) -> None:
    if not isinstance(sample.sample_id, str) or not sample.sample_id:
        raise ValueError("Store schema v1 sample_id must be a non-empty string.")
    if sample.sample_id in samples:
        raise ValueError(f"Duplicate sample_id {sample.sample_id!r}.")
    if sample.sample_index != expected_index:
        raise ValueError(
            f"Sample manifest row {expected_index} has sample_index "
            f"{sample.sample_index}."
        )
    refs: set[tuple[Role, Modality]] = set()
    for ref, meta in sample.items:
        if ref in refs:
            raise ValueError(f"Duplicate sample item ref {ref!r}.")
        if not isinstance(meta, Mapping):
            raise ValueError("Store schema v1 item metadata must be a mapping.")
        refs.add(ref)


def _v1_view_entries(
    root: Path,
    view: tuple[Role, Modality, View],
    samples: Mapping[str, tuple[int, frozenset[tuple[Role, Modality]]]],
    expected_count: int,
    shards: set[str],
) -> Iterator[ViewManifestEntry]:
    path = view_manifest_parquet_path(root, view)
    pa, pq = pyarrow()
    parquet = pq.ParquetFile(path)
    expected = parquet_schema(pa, _V1_VIEW_SCHEMA)
    if not parquet.schema_arrow.equals(expected, check_metadata=False):
        raise ValueError(
            "Store schema v1 view manifest schema does not match expected fields."
        )

    count = 0
    previous: int | None = None
    for batch in parquet.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            count += 1
            entry = _v1_view_entry(row, view, samples)
            if previous is not None and entry.sample_index <= previous:
                raise ValueError(
                    "Store schema v1 view entries must be ordered by sample_index."
                )
            previous = entry.sample_index
            shards.add(entry.shard)
            yield entry
    if count != expected_count:
        raise ValueError(
            f"View {_view_path(view)} sample count {count} "
            f"does not match item count {expected_count}."
        )


def _v1_view_entry(
    row: Mapping[str, Any],
    view: tuple[Role, Modality, View],
    samples: Mapping[str, tuple[int, frozenset[tuple[Role, Modality]]]],
) -> ViewManifestEntry:
    try:
        row_view = view_from_dict(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Store schema v1 view manifest has an invalid view ref."
        ) from exc
    if row_view != view:
        raise ValueError("View manifest entry ref must match its path.")
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Store schema v1 view sample_id must be a non-empty string.")
    sample = samples.get(sample_id)
    if sample is None:
        raise ValueError(
            f"Store schema v1 view references unknown sample_id {sample_id!r}."
        )
    sample_index, refs = sample
    if view[:2] not in refs:
        raise ValueError(
            f"View {_view_path(view)} has an entry for sample_index {sample_index} "
            "without a matching sample item."
        )
    shard = row.get("shard")
    if not _path_name(shard):
        raise ValueError(f"Store schema v1 has invalid shard name {shard!r}.")
    key = row.get("key")
    if not _path_name(key):
        raise ValueError(f"Store schema v1 has invalid payload key {key!r}.")
    return ViewManifestEntry(
        role=view[0],
        modality=view[1],
        view=view[2],
        sample_index=sample_index,
        shard=shard,
        key=key,
    )


def _copy_shards(
    source: Path,
    output: Path,
    view: tuple[Role, Modality, View],
    shards: set[str],
) -> None:
    for shard in sorted(shards):
        source_path = view_shard_path(source, view, shard)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        output_path = view_shard_path(output, view, shard)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)


def _view_path(view: tuple[Role, Modality, View]) -> tuple[str, str, str]:
    role, modality, key = view
    return role.value, modality.value, key.value


def _path_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and value not in {"", ".", ".."}
        and Path(value).name == value
    )
