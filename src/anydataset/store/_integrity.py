from __future__ import annotations

import tarfile
from pathlib import Path

from ..types.item import Modality, Role, View
from .manifestio import read_view_manifest
from .paths import view_shard_path
from .reader import read_store_views


def validate_store_payloads(stores: tuple[Path, ...]) -> None:
    for store in stores:
        for view in read_store_views(store):
            validate_store_view_payloads(store, view)


def validate_store_view_payloads(
    root: Path,
    view: tuple[Role, Modality, View],
) -> None:
    keys_by_shard: dict[str, set[str]] = {}
    for entry in read_view_manifest(root, view):
        if (entry.role, entry.modality, entry.view) != view:
            raise ValueError("View manifest entry ref must match its path.")
        if not isinstance(entry.shard, str) or Path(entry.shard).name != entry.shard:
            raise ValueError(
                f"View {_view_path(view)} has invalid shard name {entry.shard!r}."
            )
        if not isinstance(entry.key, str) or Path(entry.key).name != entry.key:
            raise ValueError(
                f"View {_view_path(view)} has invalid payload key {entry.key!r}."
            )
        keys = keys_by_shard.setdefault(entry.shard, set())
        if entry.key in keys:
            raise ValueError(
                f"View {_view_path(view)} shard {entry.shard!r} "
                f"has duplicate payload key {entry.key!r}."
            )
        keys.add(entry.key)

    while keys_by_shard:
        shard, expected = keys_by_shard.popitem()
        path = view_shard_path(root, view, shard)
        if not path.is_file():
            raise FileNotFoundError(
                f"View {_view_path(view)} is missing referenced shard {path}."
            )
        try:
            with tarfile.open(path, "r") as archive:
                missing = set(expected)
                for member in archive:
                    if member.isfile():
                        missing.discard(member.name)
        except tarfile.TarError as exc:
            raise ValueError(f"View shard is not a valid tar archive: {path}") from exc
        if missing:
            key = min(missing)
            raise ValueError(
                f"View {_view_path(view)} shard {shard!r} is missing payload {key!r}."
            )


def _view_path(view: tuple[Role, Modality, View]) -> tuple[str, str, str]:
    role, modality, key = view
    return role.value, modality.value, key.value
