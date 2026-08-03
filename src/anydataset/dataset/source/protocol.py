"""Physical dataset source contracts.

Sources prepare raw rows. Sources that can select rows without scanning the
whole stream may additionally expose original global sample indexes through
``ShardingSource``.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..._runtime.sharding import validated_shard_rows

if TYPE_CHECKING:
    from ...types import Spec


@runtime_checkable
class DatasetSource(Protocol):
    def prepare(self, spec: Spec, cache_path: Path) -> Any:
        raise NotImplementedError


@runtime_checkable
class ShardingSource(DatasetSource, Protocol):
    """Source that selects the dense global modulo shard of prepared rows.

    Sharding is opted in on the source, not by calling a prepared object's
    ``.shard()`` (for example Hugging Face ``Dataset.shard``). Prepared rows
    may expose such a method for other APIs; iterable datasets ignore it and
    only use ``iter_shard`` here or a scan-time index modulo fallback. The
    yielded index is always the dense global row index.
    """

    def iter_shard(
        self,
        dataset: object,
        *,
        num_shards: int,
        shard_id: int,
    ) -> Iterable[tuple[int, Any]]:
        raise NotImplementedError


def _native_shard(
    source: DatasetSource,
    dataset: object,
    *,
    num_shards: int,
    shard_id: int,
) -> Iterator[tuple[int, Any]] | None:
    """Return a validated native shard, or ``None`` for the scan fallback.

    Does not call ``dataset.shard(...)`` even when that method exists; only an
    ``ShardingSource`` may supply indexed rows.
    """

    if not isinstance(source, ShardingSource):
        return None

    rows = source.iter_shard(
        dataset,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    return validated_shard_rows(
        rows,
        num_shards=num_shards,
        shard_id=shard_id,
        label="Source shard",
    )


def _validate_load_options(
    spec: Spec,
    allowed: Collection[str],
    *,
    source: str,
) -> None:
    extra = set(spec.load_options) - set(allowed)
    if extra:
        name = min(extra)
        raise TypeError(f"Unexpected {source} load option: {name}.")
