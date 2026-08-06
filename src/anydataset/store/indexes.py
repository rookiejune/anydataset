from __future__ import annotations

from pathlib import Path

from .manifest.io import read_view_manifest
from .payload.archive import write_payload_index
from .reader import read_store_dataset


def rebuild_store_payload_indexes(root: str | Path) -> tuple[Path, ...]:
    """Rebuild seekable tar sidecars for every manifest-referenced store shard."""

    resolved = Path(root).expanduser()
    with read_store_dataset(resolved) as dataset:
        views = tuple(dataset.views)

    outputs: list[Path] = []
    for view in views:
        shards = dict.fromkeys(entry.shard for entry in read_view_manifest(resolved, view))
        outputs.extend(write_payload_index(resolved, view, shard) for shard in shards)
    return tuple(outputs)


__all__ = ["rebuild_store_payload_indexes"]
