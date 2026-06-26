from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..types.item import (
    AudioItem,
    AudioKey,
    ImageItem,
    Item,
    Modality,
    Role,
    Sample,
    TextItem,
)
from .jsonio import write_json
from .manifest import (
    DatasetManifest,
    SampleItemEntry,
    SampleManifestEntry,
    ViewRef,
    ViewSelection,
)
from .manifestio import (
    ManifestFormat,
    SampleManifestWriter,
    preflight_manifest_format,
)
from .paths import (
    dataset_json_path,
    dataset_ready_path,
)
from .atomic import replace_dir
from .viewwriter import ViewWriter

WRITER_PROVIDER = {"name": "DatasetWriter", "version": "canonical-v2"}


@dataclass
class DatasetWriter:
    output_dir: str | Path
    dataset_id: str
    split: str | None = None
    revision: str = "raw"
    views: tuple[ViewRef, ...] | None = None
    max_shard_samples: int | None = None
    max_shard_bytes: int | None = None
    manifest_format: ManifestFormat = "parquet"
    config: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.max_shard_samples = _optional_positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )
        self.max_shard_bytes = _optional_positive_int(
            "max_shard_bytes",
            self.max_shard_bytes,
        )
        preflight_manifest_format(self.manifest_format)
        if self.views is not None:
            self.views = tuple(self.views)
            for ref in self.views:
                if not isinstance(ref, ViewRef):
                    raise TypeError("views entries must be ViewRef instances.")

    def write(self, samples: Iterable[Sample]) -> Path:
        return replace_dir(self.output_dir, lambda tmp: self._write_to_tmp(tmp, samples))

    def _write_to_tmp(self, root: Path, samples: Iterable[Sample]) -> Path:
        sinks: dict[ViewRef, ViewWriter] = {}
        sample_manifest = SampleManifestWriter(root, self.manifest_format)
        sample_count = 0

        try:
            for index, sample in enumerate(samples):
                if not isinstance(sample, Mapping):
                    raise TypeError("DatasetWriter.write expects Sample mappings.")
                sample_id = _sample_id(self.dataset_id, index)
                _validate_sample(sample)
                refs = (
                    self.views if self.views is not None else _sample_view_refs(sample)
                )
                if not refs:
                    raise ValueError(f"Sample {sample_id} has no views.")
                sample_manifest.write(
                    _sample_manifest_entry(sample, sample_id, index, self.dataset_id)
                )
                sample_count += 1
                for ref in refs:
                    value = _sample_view_value(sample, ref)
                    if value is None:
                        if self.views is not None:
                            raise KeyError(
                                f"Sample {sample_id} is missing view {ref.path_parts()}."
                            )
                        continue
                    sink = sinks.get(ref)
                    if sink is None:
                        sink = ViewWriter(
                            root=root,
                            ref=ref,
                            revision=self.revision,
                            provider=WRITER_PROVIDER,
                            manifest_format=self.manifest_format,
                            max_shard_samples=self.max_shard_samples,
                            max_shard_bytes=self.max_shard_bytes,
                        )
                        sinks[ref] = sink
                    sink.write(sample_id, value)

            selections = tuple(
                ViewSelection(ref, self.revision)
                for ref in sorted(sinks, key=lambda item: item.path_parts())
            )
            manifest = DatasetManifest(
                dataset_id=self.dataset_id,
                split=self.split,
                sample_count=sample_count,
                views=selections,
                config=self.config,
                provenance=self.provenance,
            )
            write_json(dataset_json_path(root), manifest.to_dict())
            sample_manifest.close()
            for sink in sinks.values():
                sink.close()
            dataset_ready_path(root).touch()
            return root
        except Exception:
            sample_manifest.abort()
            for sink in sinks.values():
                sink.abort()
            raise


def _sample_manifest_entry(
    sample: Sample,
    sample_id: str,
    sample_index: int,
    dataset_id: str,
) -> SampleManifestEntry:
    return SampleManifestEntry(
        sample_id=sample_id,
        dataset_name=dataset_id,
        sample_index=sample_index,
        source={
            "dataset_name": dataset_id,
            "sample_index": sample_index,
        },
        items=tuple(_item_entry(ref, item) for ref, item in sample.items()),
    )


def _item_entry(ref: tuple[Role, Modality], item: Item) -> SampleItemEntry:
    return SampleItemEntry(
        ref=ref,
        required=_string_key_dict(item.required),
        optional=_string_key_dict(item.optional),
    )


def _sample_view_refs(sample: Sample) -> tuple[ViewRef, ...]:
    refs: list[ViewRef] = []
    for (role, modality), item in sample.items():
        for view in item.views:
            refs.append(ViewRef(modality=modality, view_key=view, role=role))
    return tuple(refs)


def _sample_view_value(sample: Sample, ref: ViewRef) -> Any:
    item = sample.get(ref.sample_ref)
    if item is None:
        return None
    return item.views.get(ref.view_key)


def _validate_item(modality: Modality, item: Item) -> None:
    match modality:
        case Modality.AUDIO:
            if not isinstance(item, AudioItem):
                raise TypeError("audio sample items must be AudioItem instances.")
            sample_rate = item.required.get(AudioKey.SAMPLE_RATE)
            if sample_rate is None:
                raise ValueError("audio sample items require sample_rate.")
            if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
                raise TypeError("audio sample_rate must be an integer.")
        case Modality.IMAGE:
            if not isinstance(item, ImageItem):
                raise TypeError("image sample items must be ImageItem instances.")
        case Modality.TEXT:
            if not isinstance(item, TextItem):
                raise TypeError("text sample items must be TextItem instances.")


def _validate_sample(sample: Sample) -> None:
    for ref, item in sample.items():
        if not isinstance(ref, tuple) or len(ref) != 2:
            raise TypeError("sample keys must be (Role, Modality) tuples.")
        role, modality = ref
        if not isinstance(role, Role):
            raise TypeError("sample role keys must be Role instances.")
        if not isinstance(modality, Modality):
            raise TypeError("sample modality keys must be Modality instances.")
        _validate_item(modality, item)


def _string_key_dict(values: Mapping[Any, Any]) -> dict[str, Any]:
    return {
        key.value if isinstance(key, StrEnum) else str(key): value
        for key, value in values.items()
    }


def _optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _sample_id(dataset_id: str, index: int) -> str:
    dataset = _slug(dataset_id)
    return f"{index:012d}-{dataset}"


def _slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-")
    return text or "sample"
