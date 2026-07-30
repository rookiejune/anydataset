"""Persist and load payload-local sample index plans for store readers."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..._io.files import StatFingerprint, stat_fingerprint
from ...types.item import Modality, Role, View
from ..manifest.index import SampleManifestSequence, StoreViews, view_path
from ..jsonio import read_json, write_json
from ..manifest.schema import ViewManifestEntry
from ..manifest.io import (
    ManifestParquetCache,
    read_samples_manifest,
    read_view_manifest,
)
from ..paths import payload_groups_path, samples_parquet_path, view_manifest_parquet_path

PAYLOAD_GROUPS_VERSION = 2
ViewRef = tuple[Role, Modality, View]
PayloadKey = tuple[tuple[ViewRef, str], ...]
PayloadLayoutFingerprint = tuple[StatFingerprint, ...]


@dataclass(frozen=True)
class PayloadGroup:
    start: int
    step: int
    count: int

    def indexes(self) -> range:
        return range(self.start, self.start + self.step * self.count, self.step)


PayloadGroups = tuple[tuple[PayloadGroup, ...], ...]


class PayloadGroupCache:
    def __init__(self) -> None:
        self.fingerprint: PayloadLayoutFingerprint | None = None
        self.groups: PayloadGroups | None = None

    def get(
        self,
        fingerprint: PayloadLayoutFingerprint,
        load: Callable[[], PayloadGroups],
    ) -> PayloadGroups:
        if self.fingerprint != fingerprint or self.groups is None:
            self.groups = load()
            self.fingerprint = fingerprint
        return self.groups

    def __getstate__(self) -> dict[str, object]:
        return {}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__init__()


def write_payload_groups(
    root: str | Path,
    views: tuple[ViewRef, ...],
    sample_count: int,
) -> Path:
    root = Path(root)
    fingerprints = {
        _relative_path(root, samples_parquet_path(root)):
        list(stat_fingerprint(samples_parquet_path(root).stat()))
    }
    for view in views:
        path = view_manifest_parquet_path(root, view)
        fingerprints[_relative_path(root, path)] = list(stat_fingerprint(path.stat()))

    cache = ManifestParquetCache(max_open_files=max(16, len(views) + 1))
    try:
        groups = _manifest_groups(root, views, sample_count, cache=cache)
    finally:
        cache.close()
    encoded = [
        [[group.start, group.step, group.count] for group in bucket]
        for bucket in groups
    ]
    path = payload_groups_path(root)
    write_json(
        path,
        {
            "version": PAYLOAD_GROUPS_VERSION,
            "sample_count": sample_count,
            "fingerprints": fingerprints,
            "groups": encoded,
            "groups_sha256": _groups_checksum(encoded),
        },
    )
    return path


def read_payload_groups(root: str | Path) -> dict[str, Any] | None:
    try:
        data = read_json(payload_groups_path(root))
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("version")
    if type(version) is not int or version != PAYLOAD_GROUPS_VERSION:
        return None
    if type(data.get("sample_count")) is not int:
        return None
    if not isinstance(data.get("fingerprints"), dict):
        return None
    if not isinstance(data.get("groups"), list):
        return None
    if not isinstance(data.get("groups_sha256"), str):
        return None
    return data


def payload_groups(
    root: Path,
    samples: SampleManifestSequence,
    views: StoreViews,
    cache: PayloadGroupCache,
) -> PayloadGroups:
    fingerprint = layout_fingerprint(root, views)
    return cache.get(
        fingerprint,
        lambda: load_payload_groups(root, samples, views),
    )


def load_payload_groups(
    root: Path,
    samples: SampleManifestSequence,
    views: StoreViews,
) -> PayloadGroups:
    sidecar = read_payload_groups(root)
    if sidecar is not None:
        groups = groups_from_sidecar(root, len(samples), views, sidecar)
        if groups is not None:
            return groups
    return scan_payload_groups(samples, views)


def groups_from_sidecar(
    root: Path,
    sample_count: int,
    views: StoreViews,
    sidecar: Mapping[str, Any],
) -> PayloadGroups | None:
    if sample_count == 0:
        return ()
    if not views:
        return None
    if sidecar.get("sample_count") != sample_count:
        return None
    fingerprints = sidecar.get("fingerprints")
    raw_groups = sidecar.get("groups")
    checksum = sidecar.get("groups_sha256")
    if (
        not isinstance(fingerprints, Mapping)
        or not isinstance(raw_groups, list)
        or not isinstance(checksum, str)
        or _groups_checksum(raw_groups) != checksum
    ):
        return None

    paths = [samples_parquet_path(root)]
    paths.extend(view_manifest_parquet_path(root, view) for view in views)
    for path in paths:
        key = _relative_path(root, path)
        expected = fingerprints.get(key)
        if (
            not isinstance(expected, list)
            or len(expected) != 5
            or any(type(value) is not int for value in expected)
            or tuple(expected) != stat_fingerprint(path.stat())
        ):
            return None

    groups: list[tuple[PayloadGroup, ...]] = []
    covered = 0
    for raw_bucket in raw_groups:
        if not isinstance(raw_bucket, list) or not raw_bucket:
            return None
        bucket: list[PayloadGroup] = []
        previous = -1
        for raw in raw_bucket:
            if (
                not isinstance(raw, list)
                or len(raw) != 3
                or any(type(value) is not int for value in raw)
            ):
                return None
            start, step, count = raw
            if start < 0 or step <= 0 or count <= 0 or start <= previous:
                return None
            last = start + step * (count - 1)
            if last >= sample_count:
                return None
            bucket.append(PayloadGroup(start, step, count))
            previous = last
            covered += count
        groups.append(tuple(bucket))
    if covered != sample_count:
        return None
    return tuple(groups)


def layout_fingerprint(
    root: Path,
    views: StoreViews,
) -> PayloadLayoutFingerprint:
    paths = [samples_parquet_path(root)]
    paths.extend(view_manifest_parquet_path(root, view) for view in views)
    return tuple(stat_fingerprint(path.stat()) for path in paths)


def scan_payload_groups(
    samples: SampleManifestSequence,
    views: StoreViews,
) -> PayloadGroups:
    if len(samples) == 0:
        return ()
    selected = tuple(views)
    if not selected:
        raise ValueError("StoreDataset shuffle requires at least one store view.")

    runs: dict[PayloadKey, list[list[int]]] = {}
    for index in range(len(samples)):
        key = payload_key(samples, views, selected, index)
        _append_index(runs.setdefault(key, []), index)
    return _group_runs(runs)


def ordered_payload_groups(groups: PayloadGroups) -> Iterator[range]:
    heap: list[tuple[int, int, Iterator[int]]] = []
    for bucket_index, bucket in enumerate(groups):
        iterator = _bucket_indexes(bucket)
        first = next(iterator, None)
        if first is not None:
            heapq.heappush(heap, (first, bucket_index, iterator))

    active_bucket: int | None = None
    start = 0
    previous = -1
    while heap:
        index, bucket_index, iterator = heapq.heappop(heap)
        if active_bucket != bucket_index or index != previous + 1:
            if active_bucket is not None:
                yield range(start, previous + 1)
            active_bucket = bucket_index
            start = index
        previous = index
        following = next(iterator, None)
        if following is not None:
            heapq.heappush(heap, (following, bucket_index, iterator))
    if active_bucket is not None:
        yield range(start, previous + 1)


def payload_key(
    samples: SampleManifestSequence,
    store_views: StoreViews,
    views: tuple[ViewRef, ...],
    index: int,
) -> PayloadKey:
    sample = samples[index]
    sample_refs = frozenset(ref for ref, _meta in sample.items)
    key: list[tuple[ViewRef, str]] = []
    for view in views:
        if view[:2] not in sample_refs:
            continue
        entry = store_views[view].entries_by_index[index]
        if entry is None:
            raise ValueError(
                f"Store view {view_path(view)} is missing sample_index {index}."
            )
        key.append((view, entry.shard))
    if not key:
        raise ValueError(
            f"Store sample_index {index} has no payload shard for selected views."
        )
    return tuple(key)


def _manifest_groups(
    root: Path,
    views: tuple[ViewRef, ...],
    sample_count: int,
    *,
    cache: ManifestParquetCache,
) -> PayloadGroups:
    if sample_count == 0:
        return ()
    if not views:
        raise ValueError("Store samples require at least one payload view.")

    iterators: dict[ViewRef, Iterator[ViewManifestEntry]] = {
        view: iter(read_view_manifest(root, view, cache=cache)) for view in views
    }
    current = {view: next(iterator, None) for view, iterator in iterators.items()}
    runs: dict[PayloadKey, list[list[int]]] = {}
    actual_count = 0
    for sample in read_samples_manifest(root, cache=cache):
        if sample.sample_index != actual_count:
            raise ValueError("Sample manifest indexes must be dense and increasing.")
        refs = frozenset(ref for ref, _meta in sample.items)
        key: list[tuple[ViewRef, str]] = []
        for view in views:
            entry = current[view]
            if view[:2] not in refs:
                if entry is not None and entry.sample_index <= sample.sample_index:
                    raise ValueError(
                        f"Store view {view_path(view)} has unexpected sample_index "
                        f"{entry.sample_index}."
                    )
                continue
            if entry is None or entry.sample_index != sample.sample_index:
                raise ValueError(
                    f"Store view {view_path(view)} is missing sample_index "
                    f"{sample.sample_index}."
                )
            if (entry.role, entry.modality, entry.view) != view:
                raise ValueError("View manifest entry ref must match its path.")
            key.append((view, entry.shard))
            current[view] = next(iterators[view], None)
        if not key:
            raise ValueError(
                f"Store sample_index {sample.sample_index} has no payload shard."
            )
        _append_index(runs.setdefault(tuple(key), []), sample.sample_index)
        actual_count += 1

    if actual_count != sample_count:
        raise ValueError("Sample manifest row count does not match sample_count.")
    if any(entry is not None for entry in current.values()):
        raise ValueError("View manifest contains sample indexes outside the dataset.")
    return _group_runs(runs)


def _append_index(runs: list[list[int]], index: int) -> None:
    if not runs:
        runs.append([index, 1, 1])
        return
    start, step, count = runs[-1]
    if count == 1:
        runs[-1][1] = index - start
        runs[-1][2] = 2
        return
    if index == start + step * count:
        runs[-1][2] += 1
        return
    runs.append([index, 1, 1])


def _group_runs(runs: Mapping[PayloadKey, list[list[int]]]) -> PayloadGroups:
    return tuple(
        tuple(PayloadGroup(start, step, count) for start, step, count in key_runs)
        for key_runs in runs.values()
    )


def _bucket_indexes(bucket: tuple[PayloadGroup, ...]) -> Iterator[int]:
    for group in bucket:
        yield from group.indexes()


def _groups_checksum(groups: list[Any]) -> str:
    payload = json.dumps(
        groups,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()
