from __future__ import annotations

import os
import shutil
from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

from ..._io.atomic import replace_dir
from ..._runtime.resume import (
    CompactIndexes,
    compact_completed_index_entries,
)
from ..._runtime.sharding import validate_shard
from ..._validation import positive_int
from ...types.item import Modality, Role, View
from ..payload.integrity import validate_store_payloads
from ..payload.groups import write_payload_groups
from .writer import (
    DatasetPartWriter,
    fragment_json_path as _fragment_json_path,
    part_json_path as _part_json_path,
)
from .sample_write import (
    view_path,
)
from ..jsonio import read_json, write_json
from ..manifest.schema import (
    DatasetManifest,
    SampleManifestEntry,
    STORE_SCHEMA_VERSION,
    ViewManifestEntry,
    dataset_manifest_dict,
    normalize_provenance,
)
from ..manifest.io import (
    ManifestParquetCache,
    read_samples_manifest,
    read_view_manifest,
    sample_manifest_row_count,
    sample_manifest_writer,
    view_manifest_writer,
)
from ..paths import (
    dataset_json_path,
    dataset_ready_path,
    view_ready_path,
    view_shard_index_path,
    view_shard_path,
    view_shards_dir,
)
from ..payload.archive import (
    Payload,
    PayloadCache,
    read_payload_bytes,
    write_payload_index,
)
from ..reader import read_store_manifest, read_store_views
from .viewwriter import ViewWriter

T = TypeVar("T")
CommitProgress = Callable[[str, int], None]
_MERGE_FAN_IN = 32
_PROGRESS_BATCH_SIZE = 1024


@dataclass(frozen=True)
class _FragmentInfo:
    path: Path
    indexes: tuple[int, ...]
    provenance: Mapping[str, str]


@dataclass(frozen=True)
class _PartInfo:
    path: Path
    metadata: Mapping[str, object]


def _common_provenance(
    inputs: Iterable[tuple[Path, Mapping[str, str]]],
    *,
    kind: str,
) -> dict[str, str] | None:
    expected: dict[str, str] | None = None
    expected_path: Path | None = None
    for path, provenance in inputs:
        current = dict(provenance)
        if expected is None:
            expected = current
            expected_path = path
        elif current != expected:
            raise ValueError(
                f"{kind} {path} provenance does not match {expected_path}."
            )
    return expected


def _commit_provenance(
    input_provenance: Mapping[str, str] | None,
    explicit: Mapping[str, str] | None,
) -> dict[str, str]:
    if explicit is None:
        return {} if input_provenance is None else dict(input_provenance)
    normalized = normalize_provenance(explicit)
    if input_provenance is not None and normalized != input_provenance:
        raise ValueError(
            "Explicit commit provenance does not match input provenance."
        )
    return normalized


class _IndexRuns:
    def __init__(self) -> None:
        self._runs: list[list[int]] = []

    def add(self, index: int) -> None:
        if not self._runs:
            self._runs.append([index, 1, 1])
            return
        start, step, count = self._runs[-1]
        if count == 1:
            self._runs[-1][1] = index - start
            self._runs[-1][2] = 2
            return
        if index == start + step * count:
            self._runs[-1][2] += 1
            return
        self._runs.append([index, 1, 1])

    def __iter__(self) -> Iterator[int]:
        for start, step, count in self._runs:
            yield from range(start, start + step * count, step)


def commit_store_parts(
    output_dir: str | Path,
    parts_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None = None,
    provenance: Mapping[str, str] | None = None,
    progress: CommitProgress | None = None,
) -> Path:
    part_infos = _part_infos(parts_dir)
    _put_progress(progress, "scan", len(part_infos))
    if not part_infos:
        raise ValueError(f"No materialized parts found: {parts_dir}")
    input_provenance = _validate_parts(
        part_infos,
        dataset_id,
        split,
        progress=progress,
    )
    commit_provenance = _commit_provenance(input_provenance, provenance)
    parts = tuple(info.path for info in part_infos)
    views = _store_views(parts)
    validate_store_payloads(parts)
    with _bounded_store_roots(
        output_dir,
        parts,
        dataset_id=dataset_id,
        split=split,
        provenance=commit_provenance,
        progress=progress,
    ) as roots:
        return replace_dir(
            output_dir,
            lambda tmp: _commit_roots_to_tmp(
                tmp,
                roots,
                dataset_id=dataset_id,
                split=split,
                views=views,
                provenance=commit_provenance,
                progress=progress,
            ),
        )


def commit_store_fragments(
    output_dir: str | Path,
    fragments_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None = None,
    expected_sample_count: int | None = None,
    max_shard_samples: int | None = None,
    provenance: Mapping[str, str] | None = None,
    progress: CommitProgress | None = None,
) -> Path:
    if expected_sample_count is not None and expected_sample_count < 0:
        raise ValueError("expected_sample_count must be non-negative.")
    if max_shard_samples is not None:
        max_shard_samples = positive_int(
            "max_shard_samples",
            max_shard_samples,
        )
    fragments_root = Path(fragments_dir).expanduser()
    fragment_infos = (
        _fragment_infos(
            fragments_root,
            dataset_id=dataset_id,
            split=split,
            progress=progress,
        )
        if fragments_root.is_dir()
        else ()
    )
    fragments = tuple(info.path for info in fragment_infos)
    if not fragments:
        raise ValueError(f"No materialized fragments found: {fragments_dir}")
    input_provenance = _common_provenance(
        ((info.path, info.provenance) for info in fragment_infos),
        kind="Fragment",
    )
    commit_provenance = _commit_provenance(input_provenance, provenance)
    views = _store_views(fragments)
    validate_store_payloads(fragments)
    with _bounded_store_roots(
        output_dir,
        fragments,
        dataset_id=dataset_id,
        split=split,
        provenance=commit_provenance,
        progress=progress,
    ) as roots:
        return replace_dir(
            output_dir,
            lambda tmp: _commit_roots_to_tmp(
                tmp,
                roots,
                dataset_id=dataset_id,
                split=split,
                expected_sample_count=expected_sample_count,
                views=views,
                repack_max_shard_samples=max_shard_samples,
                shard_prefix="part-00000-",
                provenance=commit_provenance,
                progress=progress,
            ),
        )


def commit_fragment_part(
    output_dir: str | Path,
    fragments: Sequence[str | Path],
    *,
    dataset_id: str,
    shard_id: int,
    num_shards: int,
    split: str | None = None,
    max_shard_samples: int | None = None,
    provenance: Mapping[str, str] | None = None,
    progress: CommitProgress | None = None,
) -> Path:
    validate_shard(num_shards, shard_id)
    if max_shard_samples is not None:
        max_shard_samples = positive_int(
            "max_shard_samples",
            max_shard_samples,
        )
    fragment_infos, roots = _fragment_info_roots(
        tuple(Path(path) for path in fragments),
        dataset_id=dataset_id,
        split=split,
        progress=progress,
    )
    input_provenance = _common_provenance(
        ((info.path, info.provenance) for info in fragment_infos),
        kind="Fragment",
    )
    commit_provenance = _commit_provenance(input_provenance, provenance)
    if not roots:
        return DatasetPartWriter(
            output_dir,
            dataset_id=dataset_id,
            shard_id=shard_id,
            num_shards=num_shards,
            split=split,
            provenance=commit_provenance,
        ).write(())
    views = _store_views(roots)
    sample_count = sum(len(fragment.indexes) for fragment in fragment_infos)

    with _bounded_store_roots(
        output_dir,
        roots,
        dataset_id=dataset_id,
        split=split,
        provenance=commit_provenance,
        progress=progress,
    ) as merged:
        def write(root: Path) -> Path:
            _commit_roots_to_tmp(
                root,
                merged,
                dataset_id=dataset_id,
                split=split,
                views=views,
                dense=False,
                repack_max_shard_samples=max_shard_samples,
                shard_prefix=f"part-{shard_id:05d}-",
                provenance=commit_provenance,
                progress=progress,
            )
            write_json(
                _part_json_path(root),
                {
                    "dataset_id": dataset_id,
                    "split": split,
                    "num_shards": num_shards,
                    "shard_id": shard_id,
                    "sample_count": sample_count,
                },
            )
            return root

        return replace_dir(output_dir, write)


def compact_completed_fragment_indexes(
    fragments_dir: str | Path,
    *,
    dataset_id: str,
    expected: int,
    split: str | None = None,
) -> CompactIndexes:
    root = Path(fragments_dir)
    if not root.is_dir():
        return CompactIndexes(expected, array("q"))
    fragment_paths = _fragment_paths(root)
    entries = tuple(
        (
            path.name,
            _read_fragment_info(path, dataset_id=dataset_id, split=split).indexes,
        )
        for path in fragment_paths
    )
    return compact_completed_index_entries(entries, expected=expected)


def store_fragments(
    fragments_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
) -> tuple[Path, ...]:
    if (shard_id is None) != (num_shards is None):
        raise ValueError("shard_id and num_shards must be provided together.")
    if shard_id is not None and num_shards is not None:
        validate_shard(num_shards, shard_id)
        root = Path(fragments_dir).expanduser()
        if not root.is_dir():
            return ()
        paths = _fragment_paths(root)
        keyed: list[tuple[tuple[int, str], Path]] = []
        for path in paths:
            key = _materializer_fragment_sort_key(path)
            if key is None:
                return _fragment_roots(
                    root,
                    dataset_id=dataset_id,
                    split=split,
                )[shard_id::num_shards]
            keyed.append((key, path))
        ordered = tuple(
            path
            for _key, path in sorted(keyed, key=lambda item: item[0])
        )
        paths = ordered[shard_id::num_shards]
        return tuple(
            info.path
            for info in _fragment_infos_from_paths(
                paths,
                dataset_id=dataset_id,
                split=split,
            )
        )
    return _fragment_roots(
        fragments_dir,
        dataset_id=dataset_id,
        split=split,
    )


@contextmanager
def _bounded_store_roots(
    output_dir: str | Path,
    stores: tuple[Path, ...],
    *,
    dataset_id: str,
    split: str | None,
    provenance: Mapping[str, str],
    progress: CommitProgress | None,
) -> Iterator[tuple[Path, ...]]:
    if len(stores) <= _MERGE_FAN_IN:
        yield stores
        return

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output.name}-merge-",
        dir=str(output.parent),
    ) as tmpdir:
        current = stores
        level = 0
        while len(current) > _MERGE_FAN_IN:
            merged: list[Path] = []
            for run, start in enumerate(range(0, len(current), _MERGE_FAN_IN)):
                batch = current[start : start + _MERGE_FAN_IN]
                path = Path(tmpdir) / f"level-{level:03d}-run-{run:06d}"

                def write(root: Path, batch: tuple[Path, ...] = batch) -> Path:
                    return _commit_roots_to_tmp(
                        root,
                        batch,
                        dataset_id=dataset_id,
                        split=split,
                        views=_store_views(batch),
                        dense=False,
                        provenance=provenance,
                        progress=progress,
                    )

                merged.append(replace_dir(path, write))
                _put_progress(progress, "merge-runs", len(batch))
            current = tuple(merged)
            level += 1
        yield current


def _commit_roots_to_tmp(
    root: Path,
    stores: tuple[Path, ...],
    *,
    dataset_id: str,
    split: str | None,
    provenance: Mapping[str, str],
    expected_sample_count: int | None = None,
    views: tuple[tuple[Role, Modality, View], ...] | None = None,
    dense: bool = True,
    repack_max_shard_samples: int | None = None,
    shard_prefix: str = "",
    progress: CommitProgress | None = None,
) -> Path:
    selected_views = views if views is not None else _store_views(stores)
    sample_count = _write_ordered_samples_manifest(
        root,
        stores,
        expected_sample_count=expected_sample_count,
        dense=dense,
        progress=progress,
    )
    _write_committed_view_manifests(
        root,
        stores,
        views=selected_views,
        repack_max_shard_samples=repack_max_shard_samples,
        shard_prefix=shard_prefix,
        progress=progress,
    )
    if dense:
        write_payload_groups(root, selected_views, sample_count)
    _write_committed_dataset_manifest(
        root,
        dataset_id=dataset_id,
        split=split,
        sample_count=sample_count,
        provenance=provenance,
    )
    dataset_ready_path(root).touch()
    return root


def _write_committed_view_manifests(
    root: Path,
    stores: tuple[Path, ...],
    *,
    views: tuple[tuple[Role, Modality, View], ...] | None,
    repack_max_shard_samples: int | None,
    shard_prefix: str,
    progress: CommitProgress | None,
) -> None:
    selected_views = views if views is not None else _store_views(stores)
    if not selected_views:
        return
    sample_indexes_by_ref = _sample_indexes_by_ref(root)
    for view in selected_views:
        if repack_max_shard_samples is not None and len(stores) > 1:
            repack = True
            view_count, expected_view_count = _write_repacked_view_manifest(
                root,
                stores,
                view,
                sample_indexes_by_ref.get(view[:2], ()),
                max_shard_samples=repack_max_shard_samples,
                shard_prefix=shard_prefix,
                progress=progress,
            )
            shards: frozenset[str] = frozenset()
        else:
            repack = False
            view_count, expected_view_count, shards = _write_ordered_view_manifest(
                root,
                stores,
                view,
                sample_indexes_by_ref.get(view[:2], ()),
                progress=progress,
            )
        if view_count != expected_view_count:
            raise ValueError(
                f"View {view_path(view)} sample count {view_count} "
                f"does not match item count {expected_view_count}."
            )
        if not repack:
            for store in stores:
                _copy_view_shards(
                    store,
                    root,
                    view,
                    shards=shards,
                    progress=progress,
                )
            _validate_copied_view_shards(root, view, shards)
        view_ready_path(root, view).touch()


def _write_committed_dataset_manifest(
    root: Path,
    *,
    dataset_id: str,
    split: str | None,
    sample_count: int,
    provenance: Mapping[str, str],
) -> None:
    write_json(
        dataset_json_path(root),
        dataset_manifest_dict(
            DatasetManifest(
                dataset_id=dataset_id,
                schema_version=STORE_SCHEMA_VERSION,
                split=split,
                sample_count=sample_count,
                provenance=provenance,
            )
        ),
    )


def _write_ordered_samples_manifest(
    root: Path,
    stores: tuple[Path, ...],
    *,
    expected_sample_count: int | None,
    dense: bool = True,
    progress: CommitProgress | None = None,
) -> int:
    writer = sample_manifest_writer(root)
    previous_index: int | None = None
    count = 0
    pending = 0
    try:
        for count, entry in enumerate(_merged_sample_entries(stores), start=1):
            if previous_index is not None:
                if entry.sample_index == previous_index:
                    raise ValueError(f"Duplicate sample_index {entry.sample_index}.")
                if entry.sample_index < previous_index:
                    raise ValueError(
                        "Sample manifests must be ordered by sample_index."
                    )
            if expected_sample_count is not None and count > expected_sample_count:
                raise ValueError(
                    "Materialized fragments coverage mismatch: "
                    f"unexpected sample_index {entry.sample_index}"
                )
            expected_index = count - 1
            if dense and entry.sample_index != expected_index:
                if expected_sample_count is not None:
                    raise ValueError(
                        "Materialized fragments coverage mismatch: "
                        f"missing sample_index {expected_index}"
                    )
                raise ValueError(
                    "Sample manifests must be dense by sample_index: "
                    f"missing sample_index {expected_index}."
                )
            previous_index = entry.sample_index
            writer.write(
                SampleManifestEntry(
                    sample_id=entry.sample_id,
                    sample_index=entry.sample_index,
                    items=entry.items,
                )
            )
            pending += 1
            if pending >= _PROGRESS_BATCH_SIZE:
                _put_progress(progress, "merge-samples", pending)
                pending = 0
        if expected_sample_count is not None and count != expected_sample_count:
            raise ValueError(
                "Materialized fragments coverage mismatch: "
                f"missing sample_index {count}"
            )
        writer.close()
        _put_progress(progress, "merge-samples", pending)
    except Exception:
        writer.abort()
        raise
    return count


def _write_ordered_view_manifest(
    root: Path,
    stores: tuple[Path, ...],
    view: tuple[Role, Modality, View],
    sample_indexes: Iterable[int],
    *,
    progress: CommitProgress | None,
) -> tuple[int, int, frozenset[str]]:
    writer = view_manifest_writer(root, view)
    entries = iter(_unique_view_entries(_merged_view_entries(stores, view)))
    current = _next_entry(entries)
    count = 0
    expected_count = 0
    pending = 0
    shards: set[str] = set()
    try:
        for sample_index in sample_indexes:
            expected_count += 1
            if current is None:
                raise ValueError(
                    f"View {view_path(view)} is missing sample_index "
                    f"{sample_index}."
                )
            if current.sample_index < sample_index:
                raise ValueError(
                    f"View {view_path(view)} has unexpected sample_index "
                    f"{current.sample_index}."
                )
            if current.sample_index != sample_index:
                raise ValueError(
                    f"View {view_path(view)} is missing sample_index "
                    f"{sample_index}."
                )
            writer.write(current)
            shards.add(current.shard)
            count += 1
            pending += 1
            if pending >= _PROGRESS_BATCH_SIZE:
                _put_progress(progress, "merge-views", pending)
                pending = 0
            current = _next_entry(entries)
        if current is not None:
            raise ValueError(
                f"View {view_path(view)} has unexpected sample_index "
                f"{current.sample_index}."
            )
        writer.close()
        _put_progress(progress, "merge-views", pending)
    except Exception:
        writer.abort()
        raise
    return count, expected_count, frozenset(shards)


def _write_repacked_view_manifest(
    root: Path,
    stores: tuple[Path, ...],
    view: tuple[Role, Modality, View],
    sample_indexes: Iterable[int],
    *,
    max_shard_samples: int,
    shard_prefix: str,
    progress: CommitProgress | None,
) -> tuple[int, int]:
    indexes = iter(sample_indexes)
    try:
        first_index = next(indexes)
    except StopIteration:
        first_index = None

    entries = iter(
        _unique_sourced_view_entries(_merged_sourced_view_entries(stores, view))
    )
    current = _next_sourced_entry(entries)
    if first_index is None:
        if current is not None:
            raise ValueError(
                f"View {view_path(view)} has unexpected sample_index "
                f"{current[1].sample_index}."
            )
        writer = view_manifest_writer(root, view)
        writer.close()
        return 0, 0

    writer = ViewWriter(
        root,
        view,
        max_shard_samples=max_shard_samples,
        shard_prefix=shard_prefix,
    )
    payloads = PayloadCache()
    count = 0
    expected_count = 0
    pending = 0
    try:
        for sample_index in chain((first_index,), indexes):
            expected_count += 1
            if current is None:
                raise ValueError(
                    f"View {view_path(view)} is missing sample_index "
                    f"{sample_index}."
                )
            source, entry = current
            if entry.sample_index < sample_index:
                raise ValueError(
                    f"View {view_path(view)} has unexpected sample_index "
                    f"{entry.sample_index}."
                )
            if entry.sample_index != sample_index:
                raise ValueError(
                    f"View {view_path(view)} is missing sample_index "
                    f"{sample_index}."
                )
            writer.write_payload(
                sample_index,
                Payload(
                    entry.key,
                    read_payload_bytes(source, view, entry, cache=payloads),
                ),
            )
            count += 1
            pending += 1
            if pending >= _PROGRESS_BATCH_SIZE:
                _put_progress(progress, "merge-views", pending)
                pending = 0
            current = _next_sourced_entry(entries)
        if current is not None:
            raise ValueError(
                f"View {view_path(view)} has unexpected sample_index "
                f"{current[1].sample_index}."
            )
        writer.close()
        _put_progress(progress, "merge-views", pending)
    except Exception:
        writer.abort()
        raise
    finally:
        payloads.close()
    return count, expected_count


def _sample_indexes_by_ref(
    root: Path,
) -> dict[tuple[Role, Modality], _IndexRuns]:
    indexes: dict[tuple[Role, Modality], _IndexRuns] = {}
    for entry in read_samples_manifest(root):
        for item_ref in {item_ref for item_ref, _meta in entry.items}:
            runs = indexes.get(item_ref)
            if runs is None:
                runs = _IndexRuns()
                indexes[item_ref] = runs
            runs.add(entry.sample_index)
    return indexes


def _merged_sample_entries(stores: tuple[Path, ...]) -> Iterator[SampleManifestEntry]:
    cache = ManifestParquetCache(max_open_files=max(1, len(stores)))
    try:
        yield from _merged_iterators(
            (read_samples_manifest(store, cache=cache) for store in stores),
            _sample_entry_key,
        )
    finally:
        cache.close()


def _merged_view_entries(
    stores: tuple[Path, ...],
    view: tuple[Role, Modality, View],
) -> Iterator[ViewManifestEntry]:
    selected = tuple(
        store for store in stores if view_ready_path(store, view).exists()
    )
    cache = ManifestParquetCache(max_open_files=max(1, len(selected)))
    try:
        entries = (
            _validated_view_entries(
                read_view_manifest(store, view, cache=cache),
                view,
            )
            for store in selected
        )
        yield from _merged_iterators(entries, _view_entry_key)
    finally:
        cache.close()


def _merged_sourced_view_entries(
    stores: tuple[Path, ...],
    view: tuple[Role, Modality, View],
) -> Iterator[tuple[Path, ViewManifestEntry]]:
    selected = tuple(
        store for store in stores if view_ready_path(store, view).exists()
    )
    cache = ManifestParquetCache(max_open_files=max(1, len(selected)))
    try:
        entries = (
            _sourced_view_entries(
                store,
                _validated_view_entries(
                    read_view_manifest(store, view, cache=cache),
                    view,
                ),
            )
            for store in selected
        )
        yield from _merged_iterators(
            entries,
            lambda item: _view_entry_key(item[1]),
        )
    finally:
        cache.close()


def _sourced_view_entries(
    source: Path,
    entries: Iterable[ViewManifestEntry],
) -> Iterator[tuple[Path, ViewManifestEntry]]:
    for entry in entries:
        yield source, entry


def _unique_sourced_view_entries(
    entries: Iterator[tuple[Path, ViewManifestEntry]],
) -> Iterator[tuple[Path, ViewManifestEntry]]:
    previous_index: int | None = None
    for source, entry in entries:
        if entry.sample_index == previous_index:
            raise ValueError(
                f"Duplicate view entry for sample_index {entry.sample_index}."
            )
        if previous_index is not None and entry.sample_index < previous_index:
            raise ValueError("View manifests must be ordered by sample_index.")
        previous_index = entry.sample_index
        yield source, entry


def _unique_view_entries(
    entries: Iterator[ViewManifestEntry],
) -> Iterator[ViewManifestEntry]:
    previous_index: int | None = None
    for entry in entries:
        if entry.sample_index == previous_index:
            raise ValueError(
                f"Duplicate view entry for sample_index {entry.sample_index}."
            )
        if previous_index is not None and entry.sample_index < previous_index:
            raise ValueError("View manifests must be ordered by sample_index.")
        previous_index = entry.sample_index
        yield entry


def _merged_iterators(
    entries: Iterable[Iterable[T]],
    key: Callable[[T], int],
) -> Iterator[T]:
    loaded = [iter(items) for items in entries]
    heap: list[tuple[int, int, T]] = []
    for store_index, iterator in enumerate(loaded):
        try:
            entry = next(iterator)
        except StopIteration:
            continue
        heappush(heap, (key(entry), store_index, entry))
    while heap:
        _entry_key, store_index, entry = heappop(heap)
        yield entry
        try:
            next_entry = next(loaded[store_index])
        except StopIteration:
            continue
        heappush(heap, (key(next_entry), store_index, next_entry))


def _validated_view_entries(
    entries: Iterable[ViewManifestEntry],
    view: tuple[Role, Modality, View],
) -> Iterator[ViewManifestEntry]:
    for entry in entries:
        _validate_view_entry(entry, view)
        yield entry


def _sample_entry_key(entry: SampleManifestEntry) -> int:
    return entry.sample_index


def _view_entry_key(entry: ViewManifestEntry) -> int:
    return entry.sample_index


def _next_entry(entries: Iterator[ViewManifestEntry]) -> ViewManifestEntry | None:
    try:
        return next(entries)
    except StopIteration:
        return None


def _next_sourced_entry(
    entries: Iterator[tuple[Path, ViewManifestEntry]],
) -> tuple[Path, ViewManifestEntry] | None:
    try:
        return next(entries)
    except StopIteration:
        return None


def _validate_view_entry(
    entry: ViewManifestEntry,
    view: tuple[Role, Modality, View],
) -> None:
    if (entry.role, entry.modality, entry.view) != view:
        raise ValueError("View manifest entry ref must match its path.")


def _store_views(stores: tuple[Path, ...]) -> tuple[tuple[Role, Modality, View], ...]:
    views: set[tuple[Role, Modality, View]] = set()
    for store in stores:
        views.update(read_store_views(store))
    return tuple(sorted(views, key=view_path))


def _part_infos(parts_dir: str | Path) -> tuple[_PartInfo, ...]:
    root = Path(parts_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return tuple(
        sorted(
            (
                _PartInfo(
                    path=path,
                    metadata=_read_metadata(_part_json_path(path), "Part"),
                )
                for path in root.iterdir()
                if _part_json_path(path).is_file()
            ),
            key=lambda info: (
                _metadata_integer(info.metadata, "shard_id", kind="Part"),
                info.path.name,
            ),
        )
    )


def _fragment_roots(
    fragments_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None,
    progress: CommitProgress | None = None,
) -> tuple[Path, ...]:
    root = Path(fragments_dir).expanduser()
    if not root.is_dir():
        return ()
    return tuple(
        info.path
        for info in _fragment_infos(
            root,
            dataset_id=dataset_id,
            split=split,
            progress=progress,
        )
    )


def _fragment_infos(
    root: Path,
    *,
    dataset_id: str,
    split: str | None,
    progress: CommitProgress | None = None,
) -> tuple[_FragmentInfo, ...]:
    fragments = _fragment_paths(root)
    _put_progress(progress, "scan", len(fragments))
    return _fragment_infos_from_paths(
        fragments,
        dataset_id=dataset_id,
        split=split,
        progress=progress,
    )


def _fragment_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir()
                if not path.name.startswith(".")
                if _fragment_json_path(path).is_file()
            ),
            key=lambda path: path.name,
        )
    )


def _materializer_fragment_sort_key(path: Path) -> tuple[int, str] | None:
    parts = path.name.split("-", 3)
    if (
        len(parts) != 4
        or parts[0] != "batch"
        or len(parts[1]) != 12
        or len(parts[2]) != 12
        or not parts[1].isdigit()
        or not parts[2].isdigit()
        or not parts[3]
    ):
        return None
    return int(parts[1]), path.name


def _fragment_infos_from_paths(
    fragments: tuple[Path, ...],
    *,
    dataset_id: str,
    split: str | None,
    progress: CommitProgress | None = None,
) -> tuple[_FragmentInfo, ...]:
    infos: list[_FragmentInfo] = []
    for fragment in fragments:
        infos.append(
            _read_fragment_info(
                fragment,
                dataset_id=dataset_id,
                split=split,
            )
        )
        _put_progress(progress, "validate", 1)
    return tuple(sorted(infos, key=lambda info: (min(info.indexes), info.path.name)))


def _fragment_info_roots(
    fragments: tuple[Path, ...],
    *,
    dataset_id: str,
    split: str | None,
    progress: CommitProgress | None = None,
) -> tuple[tuple[_FragmentInfo, ...], tuple[Path, ...]]:
    infos = _fragment_infos_from_paths(
        fragments,
        dataset_id=dataset_id,
        split=split,
        progress=progress,
    )
    return infos, tuple(info.path for info in infos)


def _read_metadata(path: Path, kind: str) -> Mapping[str, object]:
    data = read_json(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"{kind} metadata must be a JSON object.")
    return data


def _metadata_integer(
    data: Mapping[str, object],
    name: str,
    *,
    kind: str,
) -> int:
    value = data.get(name)
    if type(value) is not int:
        raise ValueError(f"{kind} {name} must be an integer.")
    return value


def _validate_parts(
    parts: tuple[_PartInfo, ...],
    dataset_id: str,
    split: str | None,
    *,
    progress: CommitProgress | None = None,
) -> dict[str, str]:
    num_shards: int | None = None
    shard_ids: set[int] = set()
    provenances: list[tuple[Path, Mapping[str, str]]] = []
    for info in parts:
        part = info.path
        data = info.metadata
        manifest = read_store_manifest(part, legacy_policy="reject")
        provenances.append((part, manifest.provenance))
        if data.get("dataset_id") != dataset_id:
            raise ValueError(f"Part {part} dataset_id does not match {dataset_id!r}.")
        if data.get("split") != split:
            raise ValueError(f"Part {part} split does not match {split!r}.")
        if manifest.dataset_id != data.get("dataset_id"):
            raise ValueError(
                f"Part {part} store manifest dataset_id does not match metadata."
            )
        if manifest.split != data.get("split"):
            raise ValueError(
                f"Part {part} store manifest split does not match metadata."
            )
        sample_count = _metadata_integer(data, "sample_count", kind="Part")
        if manifest.sample_count != sample_count:
            raise ValueError(
                f"Part {part} store manifest sample_count does not match metadata."
            )
        _validate_manifest_sample_count(part, manifest.sample_count, kind="Part")
        part_num_shards = _metadata_integer(data, "num_shards", kind="Part")
        shard_id = _metadata_integer(data, "shard_id", kind="Part")
        validate_shard(part_num_shards, shard_id)
        if num_shards is None:
            num_shards = part_num_shards
        elif num_shards != part_num_shards:
            raise ValueError("Materialized parts disagree on num_shards.")
        if shard_id in shard_ids:
            raise ValueError(f"Duplicate materialized part for shard_id {shard_id}.")
        shard_ids.add(shard_id)
        _put_progress(progress, "validate", 1)
    if num_shards is not None and shard_ids != set(range(num_shards)):
        missing = sorted(set(range(num_shards)) - shard_ids)
        raise ValueError(f"Missing materialized part for shard_id {missing[0]}.")
    provenance = _common_provenance(provenances, kind="Part")
    if provenance is None:
        raise RuntimeError("Part validation requires at least one input.")
    return provenance


def _read_fragment_info(
    path: Path,
    *,
    dataset_id: str,
    split: str | None,
) -> _FragmentInfo:
    data = _read_metadata(_fragment_json_path(path), "Fragment")
    if data.get("dataset_id") != dataset_id:
        raise ValueError(f"Fragment {path} dataset_id does not match {dataset_id!r}.")
    if data.get("split") != split:
        raise ValueError(f"Fragment {path} split does not match {split!r}.")
    if data.get("fragment_id") != path.name:
        raise ValueError(f"Fragment {path} id does not match its directory name.")
    indexes = _fragment_sample_indexes(data, path=path)
    manifest = read_store_manifest(path, legacy_policy="reject")
    if manifest.dataset_id != data.get("dataset_id"):
        raise ValueError(
            f"Fragment {path} store manifest dataset_id does not match metadata."
        )
    if manifest.split != data.get("split"):
        raise ValueError(
            f"Fragment {path} store manifest split does not match metadata."
        )
    if manifest.sample_count != len(indexes):
        raise ValueError(f"Fragment {path} sample indexes do not match its metadata.")
    _validate_manifest_sample_count(path, manifest.sample_count, kind="Fragment")
    _validate_fragment_sample_manifest(path, indexes)
    return _FragmentInfo(
        path=path,
        indexes=indexes,
        provenance=manifest.provenance,
    )


def _validate_manifest_sample_count(path: Path, expected: int, *, kind: str) -> None:
    actual = sample_manifest_row_count(path)
    if actual != expected:
        raise ValueError(
            f"{kind} {path} sample manifest row count {actual} "
            f"does not match declared sample_count {expected}."
        )


def _validate_fragment_sample_manifest(path: Path, indexes: tuple[int, ...]) -> None:
    samples = iter(read_samples_manifest(path))
    for expected in indexes:
        try:
            sample = next(samples)
        except StopIteration as exc:
            raise ValueError(
                f"Fragment {path} sample indexes do not match its metadata."
            ) from exc
        if sample.sample_index != expected:
            raise ValueError(
                f"Fragment {path} sample indexes do not match its metadata."
            )
    try:
        next(samples)
    except StopIteration:
        return
    raise ValueError(f"Fragment {path} sample indexes do not match its metadata.")


def _fragment_sample_indexes(
    data: Mapping[str, object],
    *,
    path: Path,
) -> tuple[int, ...]:
    raw = data.get("sample_indexes")
    if not isinstance(raw, list):
        raise ValueError("Fragment sample_indexes must be a list.")
    if not raw:
        raise ValueError(f"Fragment {path} sample_indexes must not be empty.")
    indexes: list[int] = []
    for value in raw:
        if type(value) is not int:
            raise ValueError("Fragment sample_indexes entries must be integers.")
        indexes.append(value)
    if _metadata_integer(data, "sample_count", kind="Fragment") != len(indexes):
        raise ValueError("Fragment sample_count does not match sample_indexes.")
    return tuple(indexes)


def _validate_copied_view_shards(
    root: Path,
    view: tuple[Role, Modality, View],
    shards: Iterable[str],
) -> None:
    for shard in shards:
        path = view_shard_path(root, view, shard)
        if not path.is_file():
            raise FileNotFoundError(
                f"View {view_path(view)} is missing copied shard {path}."
            )


def _copy_view_shards(
    source_root: Path,
    target_root: Path,
    view: tuple[Role, Modality, View],
    *,
    shards: Iterable[str],
    progress: CommitProgress | None,
) -> None:
    source_dir = view_shards_dir(source_root, view)
    if not source_dir.is_dir():
        return
    for shard in sorted(shards):
        source = view_shard_path(source_root, view, shard)
        if not source.is_file():
            continue
        target = view_shard_path(target_root, view, shard)
        if target.exists():
            raise ValueError(
                f"Duplicate view shard {shard!r} for {view_path(view)}."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(source, target)
        if not target.is_file():
            raise FileNotFoundError(target)
        source_index = view_shard_index_path(source_root, view, shard)
        target_index = view_shard_index_path(target_root, view, shard)
        if source_index.is_file():
            _link_or_copy(source_index, target_index)
        else:
            write_payload_index(target_root, view, shard)
        _put_progress(progress, "link-shards", 1)


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _put_progress(
    progress: CommitProgress | None,
    stage: str,
    count: int,
) -> None:
    if progress is None or count <= 0:
        return
    progress(stage, count)
