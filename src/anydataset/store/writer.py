from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._validation import positive_int
from ..types.item import (
    AudioItem,
    ImageItem,
    Item,
    Modality,
    Role,
    Sample,
    TextItem,
    View,
)
from .._io.atomic import replace_dir
from .jsonio import write_json
from .manifest import (
    STORE_SCHEMA_VERSION,
    DatasetManifest,
    SampleItem,
    SampleManifestEntry,
    dataset_manifest_dict,
    string_key_dict,
)
from .manifestio import sample_manifest_writer
from .paths import dataset_json_path, dataset_ready_path
from .viewwriter import ViewWriter

DEFAULT_MAX_SHARD_SAMPLES = 100_000


@dataclass
class DatasetWriter:
    output_dir: str | Path
    dataset_id: str
    split: str | None = None
    views: tuple[tuple[Role, Modality, View], ...] | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.max_shard_samples = positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )

    def write(self, samples: Iterable[Sample]) -> Path:
        return replace_dir(
            self.output_dir, lambda tmp: self._write_to_tmp(tmp, samples)
        )

    def _write_to_tmp(self, root: Path, samples: Iterable[Sample]) -> Path:
        sinks: dict[tuple[Role, Modality, View], ViewWriter] = {}
        sample_views: dict[tuple[Role, Modality], frozenset[View]] = {}
        sample_manifest = sample_manifest_writer(root)
        sample_count = 0
        sample_id_prefix = _sample_id_prefix(self.dataset_id)

        try:
            for index, sample in enumerate(samples):
                if not isinstance(sample, Mapping):
                    raise TypeError("DatasetWriter.write expects Sample mappings.")
                sample_id = _sample_id(sample_id_prefix, index)
                _validate_sample(sample)
                views = (
                    self.views if self.views is not None else _sample_view_refs(sample)
                )
                if not views:
                    raise ValueError(f"Sample {sample_id} has no views.")
                if self.views is None:
                    _validate_view_sets(sample, sample_views, sample_id)
                sample_manifest.write(
                    _sample_manifest_entry(sample, sample_id, index)
                )
                sample_count += 1
                for view in views:
                    value = _sample_view_value(sample, view)
                    if value is None:
                        if self.views is not None:
                            raise KeyError(
                                f"Sample {sample_id} is missing view {_view_path(view)}."
                            )
                        continue
                    sink = sinks.get(view)
                    if sink is None:
                        sink = ViewWriter(
                            root=root,
                            view=view,
                            max_shard_samples=self.max_shard_samples,
                        )
                        sinks[view] = sink
                    sink.write(index, value)

            manifest = DatasetManifest(
                dataset_id=self.dataset_id,
                schema_version=STORE_SCHEMA_VERSION,
                split=self.split,
                sample_count=sample_count,
            )
            write_json(dataset_json_path(root), dataset_manifest_dict(manifest))
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
) -> SampleManifestEntry:
    return SampleManifestEntry(
        sample_id=sample_id,
        sample_index=sample_index,
        items=tuple(_item_entry(ref, item) for ref, item in sample.items()),
    )


def _item_entry(ref: tuple[Role, Modality], item: Item) -> SampleItem:
    return ref, string_key_dict(item.meta)


def _sample_view_refs(sample: Sample) -> tuple[tuple[Role, Modality, View], ...]:
    views: list[tuple[Role, Modality, View]] = []
    for (role, modality), item in sample.items():
        for view in item.views:
            views.append((role, modality, view))
    return tuple(views)


def _validate_view_sets(
    sample: Sample,
    expected: dict[tuple[Role, Modality], frozenset[View]],
    sample_id: str,
) -> None:
    for ref, item in sample.items():
        views = frozenset(item.views)
        previous = expected.setdefault(ref, views)
        if views != previous:
            raise ValueError(
                f"Sample {sample_id} view set for {_sample_ref_path(ref)} "
                f"does not match earlier samples."
            )


def _sample_view_value(sample: Sample, view: tuple[Role, Modality, View]) -> Any:
    role, modality, key = view
    item = sample.get((role, modality))
    if item is None:
        return None
    return item.views.get(key)


def _validate_item(modality: Modality, item: Item) -> None:
    if modality is Modality.AUDIO:
        if not isinstance(item, AudioItem):
            raise TypeError("audio sample items must be AudioItem instances.")
    elif modality is Modality.IMAGE:
        if not isinstance(item, ImageItem):
            raise TypeError("image sample items must be ImageItem instances.")
    elif modality is Modality.TEXT:
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


def _sample_id_prefix(dataset_id: str) -> str:
    return _slug(dataset_id)


def _sample_id(dataset: str, index: int) -> str:
    return f"{index:012d}-{dataset}"


def _slug(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-")
    return text or "sample"


def _view_path(view: tuple[Role, Modality, View]) -> tuple[str, str, str]:
    role, modality, key = view
    return role.value, modality.value, key.value


def _sample_ref_path(ref: tuple[Role, Modality]) -> tuple[str, str]:
    role, modality = ref
    return role.value, modality.value
