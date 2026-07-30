"""Lazy sample/view manifest indexes backed by shared Parquet handles."""

from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from ..._io.files import stat_fingerprint
from ...types import item
from .schema import SampleManifestEntry, ViewManifestEntry
from .io import (
    ManifestParquetCache,
    manifest_parquet_cache,
    read_samples_manifest_row_group,
    read_view_manifest_indexes,
    read_view_manifest_row_group,
    view_manifest_layout,
)
from ..paths import view_manifest_parquet_path, view_ready_path


class SampleManifestSequence(Sequence[SampleManifestEntry]):
    def __init__(
        self,
        root: Path,
        *,
        count: int,
        row_groups: Sequence[int],
        max_cached_groups: int = 2,
        manifest_cache: ManifestParquetCache | None = None,
    ) -> None:
        self.root = root
        self._count = count
        self._row_groups = tuple(row_groups)
        self._offsets = offsets(self._row_groups)
        self._max_cached_groups = max_cached_groups
        self._manifest_cache = (
            manifest_parquet_cache() if manifest_cache is None else manifest_cache
        )
        self._cache: OrderedDict[int, tuple[SampleManifestEntry, ...]] = OrderedDict()

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> SampleManifestEntry: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[SampleManifestEntry, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> SampleManifestEntry | tuple[SampleManifestEntry, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return tuple(self[position] for position in range(start, stop, step))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("sample manifest index out of range.")
        row_group = bisect_right(self._offsets, index) - 1
        rows = self._row_group(row_group)
        return rows[index - self._offsets[row_group]]

    def __iter__(self) -> Iterator[SampleManifestEntry]:
        for row_group in range(len(self._row_groups)):
            yield from self._row_group(row_group)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "count": self._count,
            "row_groups": self._row_groups,
            "max_cached_groups": self._max_cached_groups,
            "manifest_cache": self._manifest_cache,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(
            state["root"],
            count=state["count"],
            row_groups=state["row_groups"],
            max_cached_groups=state["max_cached_groups"],
            manifest_cache=state.get("manifest_cache"),
        )

    def _row_group(self, row_group: int) -> tuple[SampleManifestEntry, ...]:
        cached = self._cache.get(row_group)
        if cached is not None:
            self._cache.move_to_end(row_group)
            return cached
        with self._manifest_cache.activate():
            rows = read_samples_manifest_row_group(self.root, row_group)
        start = self._offsets[row_group]
        for offset, sample in enumerate(rows):
            validate_sample_entry(sample, start + offset)
        self._cache[row_group] = rows
        while len(self._cache) > self._max_cached_groups:
            self._cache.popitem(last=False)
        return rows


@dataclass(frozen=True)
class StoreView:
    view: tuple[item.Role, item.Modality, item.View]
    entries_by_index: ViewEntryIndex


class ViewEntryIndex:
    def __init__(
        self,
        root: Path,
        view: tuple[item.Role, item.Modality, item.View],
        *,
        sample_count: int,
        row_groups: Sequence[int],
        sample_indexes: array[int],
        max_cached_groups: int = 2,
        manifest_cache: ManifestParquetCache | None = None,
    ) -> None:
        self.root = root
        self.view = view
        self._sample_count = sample_count
        self._row_groups = tuple(row_groups)
        self._offsets = offsets(self._row_groups)
        self._sample_indexes = sample_indexes
        self._max_cached_groups = max_cached_groups
        self._manifest_cache = (
            manifest_parquet_cache() if manifest_cache is None else manifest_cache
        )
        self._cache: OrderedDict[int, tuple[ViewManifestEntry, ...]] = OrderedDict()

    @classmethod
    def load(
        cls,
        root: Path,
        view: tuple[item.Role, item.Modality, item.View],
        *,
        sample_count: int,
        manifest_cache: ManifestParquetCache | None = None,
    ) -> ViewEntryIndex:
        if manifest_cache is None:
            manifest_cache = manifest_parquet_cache()
        path = view_manifest_parquet_path(root, view)
        fingerprint = stat_fingerprint(path.stat())
        with manifest_cache.activate():
            row_count, row_groups = view_manifest_layout(root, view)
            sample_indexes = array("q", read_view_manifest_indexes(root, view))
        if len(sample_indexes) != row_count:
            raise ValueError("View manifest row count changed while loading index.")
        if stat_fingerprint(path.stat()) != fingerprint:
            raise ValueError("View manifest changed while loading index.")
        validate_view_indexes(view, sample_indexes, sample_count)
        return cls(
            root,
            view,
            sample_count=sample_count,
            row_groups=row_groups,
            sample_indexes=sample_indexes,
            manifest_cache=manifest_cache,
        )

    def __len__(self) -> int:
        return self._sample_count

    def __getitem__(self, sample_index: int) -> ViewManifestEntry | None:
        if sample_index < 0:
            sample_index += self._sample_count
        if sample_index < 0 or sample_index >= self._sample_count:
            raise IndexError("view entry index out of range.")
        position = bisect_left(self._sample_indexes, sample_index)
        if position >= len(self._sample_indexes):
            return None
        if self._sample_indexes[position] != sample_index:
            return None
        row_group = bisect_right(self._offsets, position) - 1
        rows = self._row_group(row_group)
        entry = rows[position - self._offsets[row_group]]
        if entry.sample_index != sample_index:
            raise ValueError(
                f"View {view_path(self.view)} index changed while reading."
            )
        return entry

    def validate_coverage(self, expected_indexes: Iterable[int]) -> None:
        actual_position = 0
        actual_count = len(self._sample_indexes)
        for expected in expected_indexes:
            if actual_position >= actual_count:
                raise_view_coverage_error(self.view, missing=expected, extra=None)
            actual = int(self._sample_indexes[actual_position])
            if actual < expected:
                raise_view_coverage_error(self.view, missing=None, extra=actual)
            if actual > expected:
                raise_view_coverage_error(self.view, missing=expected, extra=None)
            actual_position += 1
        if actual_position < actual_count:
            raise_view_coverage_error(
                self.view,
                missing=None,
                extra=int(self._sample_indexes[actual_position]),
            )

    def validate_entries(self) -> None:
        for row_group in range(len(self._row_groups)):
            self._row_group(row_group)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "view": self.view,
            "sample_count": self._sample_count,
            "row_groups": self._row_groups,
            "sample_indexes": self._sample_indexes,
            "max_cached_groups": self._max_cached_groups,
            "manifest_cache": self._manifest_cache,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(
            state["root"],
            state["view"],
            sample_count=state["sample_count"],
            row_groups=state["row_groups"],
            sample_indexes=state["sample_indexes"],
            max_cached_groups=state["max_cached_groups"],
            manifest_cache=state.get("manifest_cache"),
        )

    def _row_group(self, row_group: int) -> tuple[ViewManifestEntry, ...]:
        cached = self._cache.get(row_group)
        if cached is not None:
            self._cache.move_to_end(row_group)
            return cached
        with self._manifest_cache.activate():
            rows = read_view_manifest_row_group(self.root, self.view, row_group)
        start = self._offsets[row_group]
        for offset, entry in enumerate(rows):
            if entry_view(entry) != self.view:
                raise ValueError("View manifest entry ref must match its path.")
            if entry.sample_index != self._sample_indexes[start + offset]:
                raise ValueError(
                    f"View {view_path(self.view)} index changed while reading."
                )
        self._cache[row_group] = rows
        while len(self._cache) > self._max_cached_groups:
            self._cache.popitem(last=False)
        return rows


class StoreViews(Mapping[tuple[item.Role, item.Modality, item.View], StoreView]):
    def __init__(
        self,
        root: Path,
        samples: SampleManifestSequence,
        views: Iterable[tuple[item.Role, item.Modality, item.View]],
        *,
        manifest_cache: ManifestParquetCache | None = None,
    ) -> None:
        self.root = root
        self.samples = samples
        self._manifest_cache = (
            samples._manifest_cache if manifest_cache is None else manifest_cache
        )
        self._views = tuple(views)
        self._view_set = frozenset(self._views)
        views_by_ref: dict[
            tuple[item.Role, item.Modality],
            list[tuple[item.Role, item.Modality, item.View]],
        ] = {}
        for view in self._views:
            views_by_ref.setdefault(view[:2], []).append(view)
        self._views_by_ref = {
            ref: tuple(ref_views) for ref, ref_views in views_by_ref.items()
        }
        self._cache: dict[tuple[item.Role, item.Modality, item.View], StoreView] = {}
        self._validated: set[tuple[item.Role, item.Modality, item.View]] = set()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        cache = state.get("_manifest_cache", self.samples._manifest_cache)
        self.bind_manifest_cache(cache)

    def __getitem__(
        self,
        view: tuple[item.Role, item.Modality, item.View],
    ) -> StoreView:
        if view not in self._view_set:
            raise KeyError(view)
        return self._view(view, validate_coverage=False)

    def __iter__(self) -> Iterator[tuple[item.Role, item.Modality, item.View]]:
        yield from self._views

    def __len__(self) -> int:
        return len(self._views)

    def preload(self) -> None:
        for view in self._views:
            self._view(view, validate_coverage=True)

    def for_ref(
        self,
        ref: tuple[item.Role, item.Modality],
    ) -> Iterator[tuple[tuple[item.Role, item.Modality, item.View], StoreView]]:
        for view in self._views_by_ref.get(ref, ()):
            yield view, self[view]

    def bind_manifest_cache(self, cache: ManifestParquetCache) -> None:
        self._manifest_cache = cache
        self.samples._manifest_cache = cache
        for store_view in self._cache.values():
            store_view.entries_by_index._manifest_cache = cache

    def _view(
        self,
        view: tuple[item.Role, item.Modality, item.View],
        *,
        validate_coverage: bool,
    ) -> StoreView:
        cached = self._cache.get(view)
        if cached is None:
            cached = load_view(
                self.root,
                view,
                len(self.samples),
                manifest_cache=self._manifest_cache,
            )
            self._cache[view] = cached
        if validate_coverage and view not in self._validated:
            cached.entries_by_index.validate_entries()
            cached.entries_by_index.validate_coverage(
                sample_indexes_for_ref(self.samples, view[:2])
            )
            self._validated.add(view)
        return cached


def offsets(counts: Sequence[int]) -> tuple[int, ...]:
    values = [0]
    for count in counts:
        values.append(values[-1] + count)
    return tuple(values)


def load_view(
    root: Path,
    view: tuple[item.Role, item.Modality, item.View],
    sample_count: int,
    *,
    manifest_cache: ManifestParquetCache | None = None,
) -> StoreView:
    if not view_ready_path(root, view).exists():
        raise ValueError(f"Store dataset view is not ready: {view_path(view)}.")
    return StoreView(
        view=view,
        entries_by_index=ViewEntryIndex.load(
            root,
            view,
            sample_count=sample_count,
            manifest_cache=manifest_cache,
        ),
    )


def validate_sample_entry(sample: SampleManifestEntry, index: int) -> None:
    if sample.sample_index != index:
        raise ValueError(
            f"Sample manifest row {index} has sample_index {sample.sample_index}."
        )
    refs: set[tuple[item.Role, item.Modality]] = set()
    for ref, _ in sample.items:
        if ref in refs:
            raise ValueError(f"Duplicate sample item ref {ref!r}.")
        refs.add(ref)


def validate_view_indexes(
    view: tuple[item.Role, item.Modality, item.View],
    indexes: Sequence[int],
    sample_count: int,
) -> None:
    previous: int | None = None
    for index in indexes:
        if index < 0 or index >= sample_count:
            raise ValueError(
                f"View {view_path(view)} has sample_index outside dataset: {index}."
            )
        if previous is not None:
            if index == previous:
                raise ValueError(f"Duplicate view entry for sample_index {index}.")
            if index < previous:
                raise ValueError("View manifest entries must be ordered by sample_index.")
        previous = index


def sample_indexes_for_ref(
    samples: Iterable[SampleManifestEntry],
    ref: tuple[item.Role, item.Modality],
) -> Iterator[int]:
    for sample in samples:
        if any(item_ref == ref for item_ref, _meta in sample.items):
            yield sample.sample_index


def raise_view_coverage_error(
    view: tuple[item.Role, item.Modality, item.View],
    *,
    missing: int | None,
    extra: int | None,
) -> None:
    details = []
    if missing is not None:
        details.append(f"missing sample_index {missing}")
    if extra is not None:
        details.append(f"unexpected sample_index {extra}")
    detail = ", ".join(details)
    raise ValueError(f"View {view_path(view)} sample coverage mismatch: {detail}.")


def entry_view(
    entry: ViewManifestEntry,
) -> tuple[item.Role, item.Modality, item.View]:
    return entry.role, entry.modality, entry.view


def view_path(
    view: tuple[item.Role, item.Modality, item.View],
) -> tuple[str, str, str]:
    role, modality, key = view
    return role.value, modality.value, key.value
