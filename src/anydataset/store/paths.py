from __future__ import annotations

from pathlib import Path

from ..modalities import ViewRef


def dataset_json_path(root: str | Path) -> Path:
    return Path(root) / "dataset.json"


def samples_jsonl_path(root: str | Path) -> Path:
    return Path(root) / "samples.jsonl"


def dataset_ready_path(root: str | Path) -> Path:
    return Path(root) / ".ready"


def view_dir(root: str | Path, ref: ViewRef, revision: str) -> Path:
    _validate_revision(revision)
    return Path(root).joinpath(*ref.path_parts(), str(revision))


def view_json_path(root: str | Path, ref: ViewRef, revision: str) -> Path:
    return view_dir(root, ref, revision) / "view.json"


def view_manifest_path(root: str | Path, ref: ViewRef, revision: str) -> Path:
    return view_dir(root, ref, revision) / "manifest.jsonl"


def view_ready_path(root: str | Path, ref: ViewRef, revision: str) -> Path:
    return view_dir(root, ref, revision) / ".ready"


def view_shards_dir(root: str | Path, ref: ViewRef, revision: str) -> Path:
    return view_dir(root, ref, revision) / "shards"


def view_shard_path(root: str | Path, ref: ViewRef, revision: str, shard: str) -> Path:
    _validate_segment("shard", shard)
    return view_shards_dir(root, ref, revision) / str(shard)


def _validate_revision(revision: str) -> None:
    _validate_segment("revision", revision)


def _validate_segment(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value in {"", ".", ".."}:
        raise ValueError(f"{name} must be a non-empty path segment.")
    if "/" in value:
        raise ValueError(f"{name} cannot contain '/'.")
