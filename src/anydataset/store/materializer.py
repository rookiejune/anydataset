from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .atomic import cleanup_dir, replace_dir, tmp_dir
from .jsonio import write_json
from .manifest import (
    DatasetManifest,
    SampleManifestEntry,
    ViewManifestEntry,
    ViewRef,
    ViewSelection,
    view_ref_to_dict,
)
from .manifestio import (
    write_samples_manifest,
)
from .payload import (
    payload_value,
    read_payload_bytes,
)
from .paths import (
    dataset_json_path,
    dataset_ready_path,
    view_dir,
)
from .reader import StoreDataset, StoreView, read_store_dataset
from .viewwriter import ViewWriter


@dataclass(frozen=True)
class ViewInput:
    sample: SampleManifestEntry
    ref: ViewRef
    revision: str
    value: Any


type ViewTransform = Callable[[ViewInput], Any]
type MaterializerMode = Literal["view_only", "self_contained"]


@dataclass(init=False)
class ViewMaterializer:
    input_dir: Path
    output_dir: Path
    input_ref: ViewRef
    output_ref: ViewRef
    transform: ViewTransform
    provider_name: str
    provider_version: str
    config: Mapping[str, Any]
    mode: MaterializerMode

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        input_ref: ViewRef,
        output_ref: ViewRef,
        transform: ViewTransform,
        provider_name: str,
        provider_version: str,
        config: Mapping[str, Any] | None = None,
        mode: MaterializerMode = "view_only",
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input_ref = input_ref
        self.output_ref = output_ref
        self.transform = transform
        self.provider_name = provider_name
        self.provider_version = provider_version
        self.config = {} if config is None else config
        self.mode = mode
        if not isinstance(self.input_ref, ViewRef):
            raise TypeError("input_ref must be a ViewRef.")
        if not isinstance(self.output_ref, ViewRef):
            raise TypeError("output_ref must be a ViewRef.")
        if not callable(self.transform):
            raise TypeError("transform must be callable.")
        _validate_segment("provider_name", self.provider_name)
        _validate_segment("provider_version", self.provider_version)
        _validate_json_mapping("config", self.config)
        _validate_mode(self.mode)

    def write(self) -> Path:
        source, input_view = _load_source(self.input_dir, self.input_ref)
        output_revision = _revision_for(
            provider_name=self.provider_name,
            provider_version=self.provider_version,
            config=self.config,
            input_ref=self.input_ref,
            input_revision=input_view.revision,
            output_ref=self.output_ref,
        )
        provider = _provider_metadata(
            self.provider_name,
            self.provider_version,
            self.config,
            self.input_ref,
            input_view.revision,
        )
        if self.mode == "self_contained":
            return _write_self_contained_dataset(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                source=source,
                input_view=input_view,
                output_ref=self.output_ref,
                output_revision=output_revision,
                transform=self.transform,
                provider=provider,
            )
        if _same_path(self.input_dir, self.output_dir):
            return _write_view_in_place(
                output_dir=self.output_dir,
                source=source,
                input_view=input_view,
                output_ref=self.output_ref,
                output_revision=output_revision,
                transform=self.transform,
                provider=provider,
            )
        return _write_view_dataset(
            output_dir=self.output_dir,
            source=source,
            input_view=input_view,
            output_ref=self.output_ref,
            output_revision=output_revision,
            transform=self.transform,
            provider=provider,
        )


def _write_self_contained_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    source: StoreDataset,
    input_view: StoreView,
    output_ref: ViewRef,
    output_revision: str,
    transform: ViewTransform,
    provider: Mapping[str, Any],
) -> Path:
    def write(tmp_dir: Path) -> None:
        _copy_dataset_skeleton(
            input_dir,
            tmp_dir,
            source,
            output_ref,
            output_revision,
        )
        _write_materialized_view(
            root=tmp_dir,
            source=source,
            input_view=input_view,
            output_ref=output_ref,
            output_revision=output_revision,
            transform=transform,
            provider=provider,
        )
        dataset_ready_path(tmp_dir).touch()

    return replace_dir(output_dir, write)


def _write_view_dataset(
    *,
    output_dir: Path,
    source: StoreDataset,
    input_view: StoreView,
    output_ref: ViewRef,
    output_revision: str,
    transform: ViewTransform,
    provider: Mapping[str, Any],
) -> Path:
    def write(tmp_dir: Path) -> None:
        _write_dataset_manifest(
            tmp_dir,
            source.manifest,
            (ViewSelection(output_ref, output_revision),),
        )
        write_samples_manifest(tmp_dir, source.samples)
        _write_materialized_view(
            root=tmp_dir,
            source=source,
            input_view=input_view,
            output_ref=output_ref,
            output_revision=output_revision,
            transform=transform,
            provider=provider,
        )
        dataset_ready_path(tmp_dir).touch()

    return replace_dir(output_dir, write)


def _write_view_in_place(
    *,
    output_dir: Path,
    source: StoreDataset,
    input_view: StoreView,
    output_ref: ViewRef,
    output_revision: str,
    transform: ViewTransform,
    provider: Mapping[str, Any],
) -> Path:
    target_view = view_dir(output_dir, output_ref, output_revision)
    if target_view.exists():
        raise ValueError(f"Output view revision already exists: {target_view}")
    tmp_root = tmp_dir(output_dir)
    tmp_root.mkdir(parents=True)
    try:
        _write_materialized_view(
            root=tmp_root,
            source=source,
            input_view=input_view,
            output_ref=output_ref,
            output_revision=output_revision,
            transform=transform,
            provider=provider,
        )
        target_view.parent.mkdir(parents=True, exist_ok=True)
        os.replace(view_dir(tmp_root, output_ref, output_revision), target_view)
        selections = {selection.ref: selection for selection in source.manifest.views}
        selections[output_ref] = ViewSelection(output_ref, output_revision)
        _write_dataset_manifest(
            output_dir,
            source.manifest,
            tuple(sorted(selections.values(), key=lambda item: item.ref.path_parts())),
        )
        return output_dir
    finally:
        cleanup_dir(tmp_root)


def _load_source(root: Path, input_ref: ViewRef) -> tuple[StoreDataset, StoreView]:
    source = read_store_dataset(root, cache_path=root / ".cache", views=(input_ref,))
    return source, _view_for(source, input_ref)


def _copy_dataset_skeleton(
    input_root: Path,
    output_root: Path,
    source: StoreDataset,
    output_ref: ViewRef,
    output_revision: str,
) -> None:
    selections = {selection.ref: selection for selection in source.manifest.views}
    for selection in selections.values():
        source_view = view_dir(input_root, selection.ref, selection.revision)
        target_view = view_dir(output_root, selection.ref, selection.revision)
        shutil.copytree(source_view, target_view)

    selections[output_ref] = ViewSelection(output_ref, output_revision)
    _write_dataset_manifest(
        output_root,
        source.manifest,
        tuple(sorted(selections.values(), key=lambda item: item.ref.path_parts())),
    )
    write_samples_manifest(output_root, source.samples)


def _write_dataset_manifest(
    root: Path,
    source: DatasetManifest,
    views: tuple[ViewSelection, ...],
) -> None:
    manifest = DatasetManifest(
        dataset_id=source.dataset_id,
        split=source.split,
        sample_count=source.sample_count,
        views=views,
        config=source.config,
        provenance=source.provenance,
    )
    write_json(dataset_json_path(root), manifest.to_dict())


def _write_materialized_view(
    root: Path,
    source: StoreDataset,
    input_view: StoreView,
    output_ref: ViewRef,
    output_revision: str,
    transform: ViewTransform,
    provider: Mapping[str, Any],
) -> None:
    writer = ViewWriter(
        root=root,
        ref=output_ref,
        revision=output_revision,
        provider=provider,
        manifest_format="parquet",
    )
    try:
        for sample in source.samples:
            input_entry = input_view.entries.get(sample.sample_id)
            if input_entry is None:
                raise KeyError(f"Sample {sample.sample_id!r} is missing input view.")
            value = _load_value(source.root, input_view, input_entry)
            output = transform(
                ViewInput(
                    sample=sample,
                    ref=input_view.ref,
                    revision=input_view.revision,
                    value=value,
                )
            )
            writer.write(sample.sample_id, output)
        writer.close()
    except Exception:
        writer.abort()
        raise


def _load_value(root: Path, view: StoreView, entry: ViewManifestEntry) -> Any:
    data = read_payload_bytes(root, view.ref, view.revision, entry)
    return payload_value(view.ref, data)


def _revision_for(
    provider_name: str,
    provider_version: str,
    config: Mapping[str, Any],
    input_ref: ViewRef,
    input_revision: str,
    output_ref: ViewRef,
) -> str:
    payload = {
        "provider_name": provider_name,
        "provider_version": provider_version,
        "config": dict(config),
        "input": {**view_ref_to_dict(input_ref), "revision": input_revision},
        "output": view_ref_to_dict(output_ref),
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{_slug(provider_name)}-{digest}"


def _provider_metadata(
    provider_name: str,
    provider_version: str,
    config: Mapping[str, Any],
    input_ref: ViewRef,
    input_revision: str,
) -> dict[str, Any]:
    return {
        "name": provider_name,
        "version": provider_version,
        "config": dict(config),
        "input": {**view_ref_to_dict(input_ref), "revision": input_revision},
    }


def _view_for(source: StoreDataset, ref: ViewRef) -> StoreView:
    view = source.views.get(ref)
    if view is None:
        raise KeyError(f"Unified dataset is missing requested view: {ref.path_parts()}.")
    return view


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def _validate_mode(mode: MaterializerMode) -> None:
    if mode not in {"view_only", "self_contained"}:
        raise ValueError("mode must be 'view_only' or 'self_contained'.")


def _validate_segment(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value in {"", ".", ".."}:
        raise ValueError(f"{name} must be a non-empty path segment.")
    if "/" in value:
        raise ValueError(f"{name} cannot contain '/'.")


def _validate_json_mapping(name: str, value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    try:
        json.dumps(dict(value), ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON serializable.") from exc


def _slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-")
    return text or "view"
