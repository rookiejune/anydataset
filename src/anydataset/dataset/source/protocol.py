"""Physical dataset source contracts.

Sources prepare raw rows. Sources that can select rows without scanning the
whole stream may additionally expose original global sample indexes through
``IndexedShardingSource``.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..._sharding import validate_shard

if TYPE_CHECKING:
    from ...types import Spec


class DatasetSource(Protocol):
    def prepare(self, spec: Spec, cache_path: Path) -> Iterable[Any]:
        raise NotImplementedError


@runtime_checkable
class IndexedShardingSource(DatasetSource, Protocol):
    """Source that selects the dense global modulo shard of prepared rows."""

    def iter_indexed_shard(
        self,
        dataset: object,
        *,
        num_shards: int,
        shard_id: int,
    ) -> Iterable[tuple[int, Any]]:
        raise NotImplementedError


def native_indexed_shard(
    source: DatasetSource,
    dataset: object,
    *,
    num_shards: int,
    shard_id: int,
) -> Iterator[tuple[int, Any]] | None:
    """Return a validated native shard, or ``None`` for the scan fallback."""

    validate_shard(num_shards, shard_id)
    if not isinstance(source, IndexedShardingSource):
        return None

    rows = source.iter_indexed_shard(
        dataset,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise TypeError("Source indexed shard must return an iterable.") from exc
    return _validated_indexed_rows(
        iterator,
        num_shards=num_shards,
        shard_id=shard_id,
    )


def _validated_indexed_rows(
    rows: Iterator[Any],
    *,
    num_shards: int,
    shard_id: int,
) -> Iterator[tuple[int, Any]]:
    expected = shard_id
    for entry in rows:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(
                "Source indexed shard must yield (sample_index, row) tuples."
            )
        sample_index, row = entry
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError("Source sample_index values must be integers.")
        if sample_index != expected:
            raise ValueError(
                "Source indexed shard must yield dense global sample indexes: "
                f"expected {expected}, got {sample_index}."
            )
        yield sample_index, row
        expected += num_shards


def validate_load_options(
    spec: Spec,
    allowed: Collection[str],
    *,
    source: str,
) -> None:
    extra = set(spec.load_options) - set(allowed)
    if extra:
        name = min(extra)
        raise TypeError(f"Unexpected {source} load option: {name}.")
