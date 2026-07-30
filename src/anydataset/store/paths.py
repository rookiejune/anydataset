from __future__ import annotations

from pathlib import Path

from .._validation import validate_path_segment
from ..types.item import Modality, Role, View


def dataset_json_path(root: str | Path) -> Path:
    return Path(root) / "dataset.json"


def samples_parquet_path(root: str | Path) -> Path:
    return Path(root) / "samples.parquet"


def payload_groups_path(root: str | Path) -> Path:
    return Path(root) / "payload-groups.json"


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
    validate_path_segment("shard", shard)
    return view_shards_dir(root, view) / str(shard)


def view_shard_index_path(
    root: str | Path,
    view: tuple[Role, Modality, View],
    shard: str,
) -> Path:
    """Return the optional payload offset index path for a shard."""

    shard_path = view_shard_path(root, view, shard)
    return shard_path.with_name(f"{shard_path.name}.index.json")
