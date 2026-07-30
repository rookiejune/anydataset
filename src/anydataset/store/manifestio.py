from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .._io.parquet import (
    ParquetRowWriter,
    parquet_schema,
    pyarrow,
)
from .._io.files import StatFingerprint as _StatFingerprint
from .._io.files import stat_fingerprint as _stat_fingerprint
from ..types.item import Modality, Role, View
from .manifest import (
    SampleManifestEntry,
    STORE_SCHEMA_VERSION,
    ViewManifestEntry,
    view_from_dict,
)
from .paths import samples_parquet_path, view_manifest_parquet_path


_DEFAULT_MAX_OPEN_MANIFESTS = 16
_ACTIVE_CACHE: ContextVar[ManifestParquetCache | None] = ContextVar(
    "anydataset_manifest_cache",
    default=None,
)


class ManifestParquetCache:
    """Reuse validated manifest ParquetFile handles within one process.

    Handles are keyed by the resolved path and its stat fingerprint.  A store
    can therefore be rebuilt atomically at the same path without reusing an
    old handle.  The cache is intentionally pickleable: only its capacity is
    serialized, while file descriptors are reopened in the new process.
    """

    def __init__(self, max_open_files: int = _DEFAULT_MAX_OPEN_MANIFESTS) -> None:
        if not isinstance(max_open_files, int) or isinstance(max_open_files, bool):
            raise TypeError("max_open_files must be an integer.")
        if max_open_files <= 0:
            raise ValueError("max_open_files must be positive.")
        self.max_open_files = max_open_files
        self._pid = os.getpid()
        self._files: OrderedDict[
            tuple[int, Path, _StatFingerprint], Any
        ] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, path: str | Path) -> Any:
        self._reset_after_fork()
        resolved = Path(path).expanduser().resolve()
        fingerprint = _stat_fingerprint(resolved.stat())
        pid = os.getpid()
        key = (pid, resolved, fingerprint)
        with self._lock:
            cached = self._files.get(key)
            if cached is not None:
                self._files.move_to_end(key)
                return cached

            self._drop_path(resolved)
            _, pq = pyarrow()
            parquet = pq.ParquetFile(resolved)
            try:
                if _stat_fingerprint(resolved.stat()) != fingerprint:
                    raise ValueError(f"Manifest changed while opening: {resolved}")
            except Exception:
                _close_parquet(parquet)
                raise
            self._files[key] = parquet
            self._evict()
            return parquet

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _ACTIVE_CACHE.set(self)
        try:
            yield
        finally:
            _ACTIVE_CACHE.reset(token)

    def close(self) -> None:
        self._reset_after_fork()
        with self._lock:
            files = tuple(self._files.values())
            self._files.clear()
        for parquet in files:
            _close_parquet(parquet)

    def __getstate__(self) -> dict[str, int]:
        return {"max_open_files": self.max_open_files}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(state["max_open_files"])

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _drop_path(self, path: Path) -> None:
        stale = [key for key in self._files if key[1] == path]
        for key in stale:
            parquet = self._files.pop(key)
            _close_parquet(parquet)

    def _evict(self) -> None:
        while len(self._files) > self.max_open_files:
            _key, parquet = self._files.popitem(last=False)
            _close_parquet(parquet)

    def _reset_after_fork(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        inherited = tuple(self._files.values())
        self._files = OrderedDict()
        self._lock = threading.RLock()
        self._pid = pid
        for parquet in inherited:
            _close_parquet(parquet)


_DEFAULT_CACHE = ManifestParquetCache()


def close_manifest_parquet_cache() -> None:
    """Close handles retained by standalone manifest helper calls."""

    _DEFAULT_CACHE.close()


def manifest_parquet_cache() -> ManifestParquetCache:
    """Return the process-global cache used by standalone manifest helpers."""

    return _DEFAULT_CACHE


def _active_cache(cache: ManifestParquetCache | None) -> ManifestParquetCache:
    return cache or _ACTIVE_CACHE.get() or _DEFAULT_CACHE


def samples_manifest_exists(root: str | Path) -> bool:
    return samples_parquet_path(root).is_file()


def read_samples_manifest(
    root: str | Path,
    *,
    cache: ManifestParquetCache | None = None,
) -> Iterator[SampleManifestEntry]:
    for row in _read_manifest_rows(
        samples_parquet_path(root),
        _SAMPLE_SCHEMA,
        kind="sample",
        cache=cache,
    ):
        yield _sample_entry(row)


def sample_manifest_row_count(
    root: str | Path,
    *,
    cache: ManifestParquetCache | None = None,
) -> int:
    return sample_manifest_layout(root, cache=cache)[0]


def sample_manifest_layout(
    root: str | Path,
    *,
    cache: ManifestParquetCache | None = None,
) -> tuple[int, tuple[int, ...]]:
    return _manifest_layout(
        samples_parquet_path(root),
        _SAMPLE_SCHEMA,
        kind="sample",
        cache=cache,
    )


def read_samples_manifest_row_group(
    root: str | Path,
    row_group: int,
    *,
    cache: ManifestParquetCache | None = None,
) -> tuple[SampleManifestEntry, ...]:
    return tuple(
        _sample_entry(row)
        for row in _read_manifest_rows(
            samples_parquet_path(root),
            _SAMPLE_SCHEMA,
            kind="sample",
            row_group=row_group,
            cache=cache,
        )
    )


def read_sample_manifest_index(
    root: str | Path,
    *,
    cache: ManifestParquetCache | None = None,
) -> Iterator[tuple[int, str]]:
    parquet = _validated_parquet(
        samples_parquet_path(root),
        _SAMPLE_SCHEMA,
        kind="sample",
        cache=cache,
    )
    for batch in parquet.iter_batches(
        batch_size=4096,
        columns=["sample_index", "sample_id"],
    ):
        indexes = batch.column(0)
        sample_ids = batch.column(1)
        for position in range(len(indexes)):
            sample_index = indexes[position].as_py()
            sample_id = sample_ids[position].as_py()
            if sample_index is None or sample_id is None:
                raise ValueError("Sample manifest index columns cannot contain nulls.")
            yield int(sample_index), str(sample_id)


def write_samples_manifest(
    root: str | Path,
    entries: Iterable[SampleManifestEntry],
) -> None:
    writer = sample_manifest_writer(root)
    try:
        for entry in entries:
            writer.write(entry)
        writer.close()
    except Exception:
        writer.abort()
        raise


def read_view_manifest(
    root: str | Path,
    view: tuple[Role, Modality, View],
    *,
    cache: ManifestParquetCache | None = None,
) -> Iterator[ViewManifestEntry]:
    for row in _read_view_manifest_rows(root, view, cache=cache):
        yield _view_entry(row)


def view_manifest_layout(
    root: str | Path,
    view: tuple[Role, Modality, View],
    *,
    cache: ManifestParquetCache | None = None,
) -> tuple[int, tuple[int, ...]]:
    return _manifest_layout(
        view_manifest_parquet_path(root, view),
        _VIEW_SCHEMA,
        kind="view",
        cache=cache,
    )


def read_view_manifest_row_group(
    root: str | Path,
    view: tuple[Role, Modality, View],
    row_group: int,
    *,
    cache: ManifestParquetCache | None = None,
) -> tuple[ViewManifestEntry, ...]:
    return tuple(
        _view_entry(row)
        for row in _read_view_manifest_rows(
            root,
            view,
            row_group=row_group,
            cache=cache,
        )
    )


def read_view_manifest_indexes(
    root: str | Path,
    view: tuple[Role, Modality, View],
    *,
    cache: ManifestParquetCache | None = None,
) -> Iterator[int]:
    yield from _read_view_manifest_indexes(root, view, cache=cache)


def write_view_manifest(
    root: str | Path,
    view: tuple[Role, Modality, View],
    entries: Iterable[ViewManifestEntry],
) -> None:
    writer = view_manifest_writer(root, view)
    try:
        for entry in entries:
            writer.write(entry)
        writer.close()
    except Exception:
        writer.abort()
        raise


def sample_manifest_writer(root: str | Path) -> ParquetRowWriter:
    return ParquetRowWriter(samples_parquet_path(root), _SAMPLE_SCHEMA, _sample_row)


def view_manifest_writer(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> ParquetRowWriter:
    return ParquetRowWriter(
        view_manifest_parquet_path(root, view),
        _VIEW_SCHEMA,
        _view_row,
    )


def _sample_row(entry: SampleManifestEntry) -> dict[str, Any]:
    return {
        "sample_id": entry.sample_id,
        "sample_index": entry.sample_index,
        "items": _json_text(
            tuple(
                (
                    (role.value, modality.value),
                    dict(meta),
                )
                for (role, modality), meta in entry.items
            )
        ),
    }


def _view_row(entry: ViewManifestEntry) -> dict[str, Any]:
    return {
        "role": entry.role.value,
        "modality": entry.modality.value,
        "view": entry.view.value,
        "sample_index": entry.sample_index,
        "shard": entry.shard,
        "key": entry.key,
    }


def _sample_entry(row: dict[str, Any]) -> SampleManifestEntry:
    return SampleManifestEntry(
        **{
            **row,
            "items": tuple(
                (
                    (Role(item[0][0]), Modality(item[0][1])),
                    item[1],
                )
                for item in row["items"]
            ),
        }
    )


def _view_entry(row: dict[str, Any]) -> ViewManifestEntry:
    role, modality, view = view_from_dict(row)
    return ViewManifestEntry(
        **{
            **row,
            "role": role,
            "modality": modality,
            "view": view,
        }
    )


def _decode_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("items",):
        value = row.get(key)
        if isinstance(value, str):
            row[key] = json.loads(value)
    return row


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _read_view_manifest_rows(
    root: str | Path,
    view: tuple[Role, Modality, View],
    *,
    row_group: int | None = None,
    cache: ManifestParquetCache | None = None,
) -> Iterator[dict[str, Any]]:
    path = view_manifest_parquet_path(root, view)
    yield from _read_manifest_rows(
        path,
        _VIEW_SCHEMA,
        kind="view",
        row_group=row_group,
        cache=cache,
    )


def _read_view_manifest_indexes(
    root: str | Path,
    view: tuple[Role, Modality, View],
    *,
    cache: ManifestParquetCache | None = None,
) -> Iterator[int]:
    path = view_manifest_parquet_path(root, view)
    parquet = _validated_parquet(path, _VIEW_SCHEMA, kind="view", cache=cache)
    for batch in parquet.iter_batches(batch_size=4096, columns=["sample_index"]):
        indexes = batch.column(0)
        for position in range(len(indexes)):
            sample_index = indexes[position].as_py()
            if sample_index is None:
                raise ValueError("View manifest sample_index cannot contain nulls.")
            yield int(sample_index)


def _manifest_layout(
    path: str | Path,
    fields: tuple[tuple[str, str], ...],
    *,
    kind: str,
    cache: ManifestParquetCache | None = None,
) -> tuple[int, tuple[int, ...]]:
    parquet = _validated_parquet(path, fields, kind=kind, cache=cache)
    metadata = parquet.metadata
    return int(metadata.num_rows), tuple(
        int(metadata.row_group(index).num_rows)
        for index in range(metadata.num_row_groups)
    )


def _validated_parquet(
    path: str | Path,
    fields: tuple[tuple[str, str], ...],
    *,
    kind: str,
    cache: ManifestParquetCache | None = None,
):
    pa, _ = pyarrow()
    parquet = _active_cache(cache).get(path)
    actual = parquet.schema_arrow
    expected = parquet_schema(pa, fields)
    if not actual.equals(expected, check_metadata=False):
        raise ValueError(
            f"Store schema {STORE_SCHEMA_VERSION} {kind} manifest schema "
            "does not match expected fields."
        )
    return parquet


def _read_manifest_rows(
    path: str | Path,
    fields: tuple[tuple[str, str], ...],
    *,
    kind: str,
    row_group: int | None = None,
    cache: ManifestParquetCache | None = None,
) -> Iterator[dict[str, Any]]:
    parquet = _validated_parquet(path, fields, kind=kind, cache=cache)
    if row_group is None:
        rows = (
            row
            for batch in parquet.iter_batches(batch_size=4096)
            for row in batch.to_pylist()
        )
    else:
        rows = iter(parquet.read_row_group(row_group).to_pylist())
    for row in rows:
        yield _decode_row(row)


_SAMPLE_SCHEMA = (
    ("sample_id", "string"),
    ("sample_index", "int64"),
    ("items", "string"),
)
_VIEW_SCHEMA = (
    ("modality", "string"),
    ("role", "string"),
    ("view", "string"),
    ("sample_index", "int64"),
    ("shard", "string"),
    ("key", "string"),
)


def _close_parquet(parquet: Any) -> None:
    close = getattr(parquet, "close", None)
    if callable(close):
        close()
