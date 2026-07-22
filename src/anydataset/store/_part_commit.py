from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from heapq import heappop, heappush
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

from .._io.atomic import replace_dir
from .._resume import cached_completed_indexes, write_completed_index_cache
from .._sharding import validate_shard
from ..types.item import Modality, Role, View
from ._integrity import validate_store_payloads
from ._part_writer import (
    DatasetPartWriter,
    fragment_json_path as _fragment_json_path,
    part_json_path as _part_json_path,
)
from ._sample_write import (
    view_path,
)
from .jsonio import read_json, write_json
from .manifest import (
    DatasetManifest,
    SampleManifestEntry,
    STORE_SCHEMA_VERSION,
    ViewManifestEntry,
    dataset_manifest_dict,
)
from .manifestio import (
    read_samples_manifest,
    read_view_manifest,
    sample_manifest_row_count,
    sample_manifest_writer,
    view_manifest_writer,
)
from .paths import (
    dataset_json_path,
    dataset_ready_path,
    view_ready_path,
    view_shard_path,
    view_shards_dir,
)
from .reader import read_store_manifest, read_store_views

T = TypeVar("T")
_MERGE_FAN_IN = 32


def commit_store_parts(
    output_dir: str | Path,
    parts_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None = None,
) -> Path:
    parts = _part_roots(parts_dir)
    if not parts:
        raise ValueError(f"No materialized parts found: {parts_dir}")
    _validate_parts(parts, dataset_id, split)
    views = _store_views(parts)
    validate_store_payloads(parts)
    with _bounded_store_roots(
        output_dir,
        parts,
        dataset_id=dataset_id,
        split=split,
    ) as roots:
        return replace_dir(
            output_dir,
            lambda tmp: _commit_roots_to_tmp(
                tmp,
                roots,
                dataset_id=dataset_id,
                split=split,
                views=views,
            ),
        )


def commit_store_fragments(
    output_dir: str | Path,
    fragments_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None = None,
    expected_sample_count: int | None = None,
) -> Path:
    if expected_sample_count is not None and expected_sample_count < 0:
        raise ValueError("expected_sample_count must be non-negative.")
    fragments = _fragment_roots(
        fragments_dir,
        dataset_id=dataset_id,
        split=split,
    )
    if not fragments:
        raise ValueError(f"No materialized fragments found: {fragments_dir}")
    views = _store_views(fragments)
    validate_store_payloads(fragments)
    with _bounded_store_roots(
        output_dir,
        fragments,
        dataset_id=dataset_id,
        split=split,
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
) -> Path:
    validate_shard(num_shards, shard_id)
    roots = _validate_fragment_roots(
        tuple(Path(path) for path in fragments),
        dataset_id=dataset_id,
        split=split,
    )
    if not roots:
        return DatasetPartWriter(
            output_dir,
            dataset_id=dataset_id,
            shard_id=shard_id,
            num_shards=num_shards,
            split=split,
        ).write(())
    views = _store_views(roots)
    validate_store_payloads(roots)
    sample_count = sum(read_store_manifest(fragment).sample_count for fragment in roots)

    with _bounded_store_roots(
        output_dir,
        roots,
        dataset_id=dataset_id,
        split=split,
    ) as merged:
        def write(root: Path) -> Path:
            _commit_roots_to_tmp(
                root,
                merged,
                dataset_id=dataset_id,
                split=split,
                views=views,
                dense=False,
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


def completed_fragment_indexes(
    fragments_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None = None,
) -> frozenset[int]:
    root = Path(fragments_dir)
    if not root.is_dir():
        return frozenset()
    fragment_dirs = _fragment_dirs(root)
    cached = cached_completed_indexes(root, (path.name for path in fragment_dirs))
    if cached is not None:
        return cached
    indexes: set[int] = set()
    cache_entries: list[tuple[str, tuple[int, ...]]] = []
    for fragment in _validate_fragment_roots(
        fragment_dirs,
        dataset_id=dataset_id,
        split=split,
    ):
        data = read_json(_fragment_json_path(fragment))
        fragment_indexes = _fragment_sample_indexes(data)
        cache_entries.append((fragment.name, fragment_indexes))
        for index in fragment_indexes:
            if index in indexes:
                raise ValueError(f"Duplicate materialized fragment index {index}.")
            indexes.add(index)
    write_completed_index_cache(root, cache_entries)
    return frozenset(indexes)


def store_fragments(
    fragments_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None = None,
) -> tuple[Path, ...]:
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
                    )

                merged.append(replace_dir(path, write))
            current = tuple(merged)
            level += 1
        yield current


def _commit_roots_to_tmp(
    root: Path,
    stores: tuple[Path, ...],
    *,
    dataset_id: str,
    split: str | None,
    expected_sample_count: int | None = None,
    views: tuple[tuple[Role, Modality, View], ...] | None = None,
    dense: bool = True,
) -> Path:
    sample_count = _write_ordered_samples_manifest(
        root,
        stores,
        expected_sample_count=expected_sample_count,
        dense=dense,
    )
    _write_committed_view_manifests(
        root,
        stores,
        views=views,
    )
    _write_committed_dataset_manifest(
        root,
        dataset_id=dataset_id,
        split=split,
        sample_count=sample_count,
    )
    dataset_ready_path(root).touch()
    return root


def _write_committed_view_manifests(
    root: Path,
    stores: tuple[Path, ...],
    *,
    views: tuple[tuple[Role, Modality, View], ...] | None,
) -> None:
    for view in (views if views is not None else _store_views(stores)):
        view_count, expected_view_count, shards = _write_ordered_view_manifest(
            root,
            stores,
            view,
            _sample_indexes_for_ref(root, view[:2]),
        )
        if view_count != expected_view_count:
            raise ValueError(
                f"View {view_path(view)} sample count {view_count} "
                f"does not match item count {expected_view_count}."
            )
        for store in stores:
            _copy_view_shards(store, root, view)
        _validate_copied_view_shards(root, view, shards)
        view_ready_path(root, view).touch()


def _write_committed_dataset_manifest(
    root: Path,
    *,
    dataset_id: str,
    split: str | None,
    sample_count: int,
) -> None:
    write_json(
        dataset_json_path(root),
        dataset_manifest_dict(
            DatasetManifest(
                dataset_id=dataset_id,
                schema_version=STORE_SCHEMA_VERSION,
                split=split,
                sample_count=sample_count,
            )
        ),
    )


def _write_ordered_samples_manifest(
    root: Path,
    stores: tuple[Path, ...],
    *,
    expected_sample_count: int | None,
    dense: bool = True,
) -> int:
    writer = sample_manifest_writer(root)
    previous_index: int | None = None
    count = 0
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
        if expected_sample_count is not None and count != expected_sample_count:
            raise ValueError(
                "Materialized fragments coverage mismatch: "
                f"missing sample_index {count}"
            )
        writer.close()
    except Exception:
        writer.abort()
        raise
    return count


def _write_ordered_view_manifest(
    root: Path,
    stores: tuple[Path, ...],
    view: tuple[Role, Modality, View],
    sample_indexes: Iterable[int],
) -> tuple[int, int, frozenset[str]]:
    writer = view_manifest_writer(root, view)
    entries = iter(_unique_view_entries(_merged_view_entries(stores, view)))
    current = _next_entry(entries)
    count = 0
    expected_count = 0
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
            current = _next_entry(entries)
        if current is not None:
            raise ValueError(
                f"View {view_path(view)} has unexpected sample_index "
                f"{current.sample_index}."
            )
        writer.close()
    except Exception:
        writer.abort()
        raise
    return count, expected_count, frozenset(shards)


def _sample_indexes_for_ref(
    root: Path,
    ref: tuple[Role, Modality],
) -> Iterator[int]:
    for entry in read_samples_manifest(root):
        if any(item_ref == ref for item_ref, _meta in entry.items):
            yield entry.sample_index


def _merged_sample_entries(stores: tuple[Path, ...]) -> Iterator[SampleManifestEntry]:
    yield from _merged_iterators(
        (read_samples_manifest(store) for store in stores),
        _sample_entry_key,
    )


def _merged_view_entries(
    stores: tuple[Path, ...],
    view: tuple[Role, Modality, View],
) -> Iterator[ViewManifestEntry]:
    entries = (
        _validated_view_entries(read_view_manifest(store, view), view)
        for store in stores
        if view_ready_path(store, view).exists()
    )
    yield from _merged_iterators(entries, _view_entry_key)


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


def _part_roots(parts_dir: str | Path) -> tuple[Path, ...]:
    root = Path(parts_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return tuple(
        sorted(
            (path for path in root.iterdir() if _part_json_path(path).is_file()),
            key=lambda path: _part_sort_key(path),
        )
    )


def _fragment_roots(
    fragments_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None,
) -> tuple[Path, ...]:
    root = Path(fragments_dir).expanduser()
    if not root.is_dir():
        return ()
    return _validate_fragment_roots(
        _fragment_dirs(root),
        dataset_id=dataset_id,
        split=split,
    )


def _fragment_dirs(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir()
                if not path.name.startswith(".")
                if _fragment_json_path(path).is_file()
            ),
            key=_fragment_sort_key,
        )
    )


def _validate_fragment_roots(
    fragments: tuple[Path, ...],
    *,
    dataset_id: str,
    split: str | None,
) -> tuple[Path, ...]:
    for fragment in fragments:
        _validate_fragment(fragment, dataset_id, split)
    return fragments


def _part_sort_key(path: Path) -> tuple[int, str]:
    data = read_json(_part_json_path(path))
    return int(data["shard_id"]), path.name


def _validate_parts(
    parts: tuple[Path, ...],
    dataset_id: str,
    split: str | None,
) -> None:
    num_shards: int | None = None
    shard_ids: set[int] = set()
    for part in parts:
        data = read_json(_part_json_path(part))
        manifest = read_store_manifest(part)
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
        if manifest.sample_count != data.get("sample_count"):
            raise ValueError(
                f"Part {part} store manifest sample_count does not match metadata."
            )
        _validate_manifest_sample_count(part, manifest.sample_count, kind="Part")
        part_num_shards = int(data["num_shards"])
        shard_id = int(data["shard_id"])
        validate_shard(part_num_shards, shard_id)
        if num_shards is None:
            num_shards = part_num_shards
        elif num_shards != part_num_shards:
            raise ValueError("Materialized parts disagree on num_shards.")
        if shard_id in shard_ids:
            raise ValueError(f"Duplicate materialized part for shard_id {shard_id}.")
        shard_ids.add(shard_id)
    if num_shards is not None and shard_ids != set(range(num_shards)):
        missing = sorted(set(range(num_shards)) - shard_ids)
        raise ValueError(f"Missing materialized part for shard_id {missing[0]}.")


def _validate_fragment(path: Path, dataset_id: str, split: str | None) -> None:
    data = read_json(_fragment_json_path(path))
    if data.get("dataset_id") != dataset_id:
        raise ValueError(f"Fragment {path} dataset_id does not match {dataset_id!r}.")
    if data.get("split") != split:
        raise ValueError(f"Fragment {path} split does not match {split!r}.")
    if data.get("fragment_id") != path.name:
        raise ValueError(f"Fragment {path} id does not match its directory name.")
    indexes = _fragment_sample_indexes(data)
    manifest = read_store_manifest(path)
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


def _fragment_sample_indexes(data: Mapping[str, object]) -> tuple[int, ...]:
    raw = data.get("sample_indexes")
    if not isinstance(raw, list):
        raise ValueError("Fragment sample_indexes must be a list.")
    indexes: list[int] = []
    for value in raw:
        if not isinstance(value, int):
            raise ValueError("Fragment sample_indexes entries must be integers.")
        indexes.append(value)
    if data.get("sample_count") != len(indexes):
        raise ValueError("Fragment sample_count does not match sample_indexes.")
    return tuple(indexes)


def _fragment_sort_key(path: Path) -> tuple[int, str]:
    data = read_json(_fragment_json_path(path))
    indexes = _fragment_sample_indexes(data)
    return min(indexes), path.name

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
) -> None:
    source_dir = view_shards_dir(source_root, view)
    if not source_dir.is_dir():
        return
    for source in sorted(source_dir.iterdir()):
        if not source.is_file():
            continue
        target = view_shard_path(target_root, view, source.name)
        if target.exists():
            raise ValueError(
                f"Duplicate view shard {source.name!r} for {view_path(view)}."
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(source, target)
        if not target.is_file():
            raise FileNotFoundError(target)


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
