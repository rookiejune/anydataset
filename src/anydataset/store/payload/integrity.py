from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Literal

from ..._validation import validate_path_segment
from ...types.item import Modality, Role, View
from ..manifest.io import read_view_manifest
from ..paths import view_shard_path
from ..reader import read_store_views

IntegrityLevel = Literal["fast", "normal", "full"]


def validate_store_payloads(
    stores: tuple[Path, ...],
    *,
    level: IntegrityLevel = "full",
) -> None:
    _validate_level(level)
    for store in stores:
        for view in read_store_views(store):
            validate_store_view_payloads(store, view, level=level)


def validate_store_view_payloads(
    root: Path,
    view: tuple[Role, Modality, View],
    *,
    level: IntegrityLevel = "full",
) -> None:
    _validate_level(level)
    keys_by_shard: dict[str, set[str]] = {}
    for entry in read_view_manifest(root, view):
        if (entry.role, entry.modality, entry.view) != view:
            raise ValueError("View manifest entry ref must match its path.")
        if not _path_segment("shard", entry.shard):
            raise ValueError(
                f"View {_view_path(view)} has invalid shard name {entry.shard!r}."
            )
        if not _path_segment("payload key", entry.key):
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
        if level == "fast":
            continue
        try:
            with tarfile.open(path, "r") as archive:
                missing = set(expected)
                members: set[str] = set()
                for member in archive:
                    if not member.isfile():
                        continue
                    if member.name in members:
                        raise ValueError(
                            f"View {_view_path(view)} shard {shard!r} "
                            f"has duplicate payload key {member.name!r}."
                        )
                    members.add(member.name)
                    if not _path_segment("payload key", member.name):
                        raise ValueError(
                            f"View {_view_path(view)} shard {shard!r} "
                            f"has invalid payload key {member.name!r}."
                        )
                    missing.discard(member.name)
        except tarfile.TarError as exc:
            raise ValueError(f"View shard is not a valid tar archive: {path}") from exc
        if level == "full" and missing:
            key = min(missing)
            raise ValueError(
                f"View {_view_path(view)} shard {shard!r} is missing payload {key!r}."
            )


def _validate_level(level: str) -> None:
    if level not in {"fast", "normal", "full"}:
        raise ValueError(
            f"integrity level must be 'fast', 'normal', or 'full', got {level!r}."
        )


def _path_segment(name: str, value: str) -> bool:
    try:
        validate_path_segment(name, value)
    except (TypeError, ValueError):
        return False
    return True


def _view_path(view: tuple[Role, Modality, View]) -> tuple[str, str, str]:
    role, modality, key = view
    return role.value, modality.value, key.value
