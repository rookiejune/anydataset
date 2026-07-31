from __future__ import annotations

from pathlib import Path


def migrate_store(source: str | Path, output: str | Path) -> Path:
    """Migrate a schema-v1 store to an independent schema-v3 store.

    This is the public store-level migration entry point. It intentionally
    writes a standalone output store with copied payload shards, suitable for
    publishing or moving independently from the source store.
    """

    from .manifest.migration import migrate_store as _migrate_store

    return _migrate_store(source, output)


__all__ = ["migrate_store"]
