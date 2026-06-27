from __future__ import annotations

from pathlib import Path

from ..types.item import Modality, Role, View


def dataset_json_path(root: str | Path) -> Path:
    return Path(root) / "dataset.json"


def samples_parquet_path(root: str | Path) -> Path:
    return Path(root) / "samples.parquet"


def dataset_ready_path(root: str | Path) -> Path:
    return Path(root) / ".ready"


def view_dir(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> Path:
    role, modality, key = view
    return Path(root) / role.value / modality.value / key.value


def view_manifest_parquet_path(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> Path:
    return view_dir(root, view) / "manifest.parquet"


def view_ready_path(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> Path:
    return view_dir(root, view) / ".ready"


def view_shards_dir(
    root: str | Path,
    view: tuple[Role, Modality, View],
) -> Path:
    return view_dir(root, view) / "shards"


def view_shard_path(
    root: str | Path,
    view: tuple[Role, Modality, View],
    shard: str,
) -> Path:
    _validate_segment("shard", shard)
    return view_shards_dir(root, view) / str(shard)


def _validate_segment(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value in {"", ".", ".."}:
        raise ValueError(f"{name} must be a non-empty path segment.")
    if "/" in value:
        raise ValueError(f"{name} cannot contain '/'.")
