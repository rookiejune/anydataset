from __future__ import annotations

import re
import tarfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..dataset.abc import Sample
from ..types.item import (
    AudioItem,
    AudioKey,
    AudioView,
    ImageItem,
    ImageView,
    Item,
    Modality,
    Role,
    TextItem,
    TextView,
    View,
)
from .jsonio import write_json
from .manifest import (
    STORE_SCHEMA_VERSION,
    DatasetManifest,
    SampleItemEntry,
    SampleManifestEntry,
    ViewManifestEntry,
    ViewRef,
    ViewSelection,
    view_ref_to_dict,
)
from .manifestio import (
    ManifestFormat,
    preflight_manifest_format,
    write_samples_manifest,
    write_view_manifest,
)
from .paths import (
    dataset_json_path,
    dataset_ready_path,
    view_json_path,
    view_ready_path,
    view_shard_path,
)
from .atomic import replace_dir
from .payload import add_payload, checksum, payload_for_view

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
                _validate_view_ref(ref)

    def write(self, samples: Iterable[Sample]) -> Path:
        return replace_dir(self.output_dir, lambda tmp: self._write_to_tmp(tmp, samples))

    def _write_to_tmp(self, root: Path, samples: Iterable[Sample]) -> Path:
        sinks: dict[ViewRef, _ViewSink] = {}
        sample_entries: list[SampleManifestEntry] = []

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
                sample_entries.append(
                    _sample_manifest_entry(
                        sample=sample,
                        sample_id=sample_id,
                        sample_index=index,
                        dataset_id=self.dataset_id,
                    )
                )
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
                        sink = _ViewSink(
                            root=root,
                            ref=ref,
                            revision=self.revision,
                            max_shard_samples=self.max_shard_samples,
                            max_shard_bytes=self.max_shard_bytes,
                            manifest_format=self.manifest_format,
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
                sample_count=len(sample_entries),
                views=selections,
                config=self.config,
                provenance=self.provenance,
            )
            write_json(dataset_json_path(root), manifest.to_dict())
            write_samples_manifest(root, sample_entries, self.manifest_format)
            for sink in sinks.values():
                sink.close()
            dataset_ready_path(root).touch()
            return root
        except Exception:
            for sink in sinks.values():
                sink.close_payload()
            raise


class _ViewSink:
    def __init__(
        self,
        root: Path,
        ref: ViewRef,
        revision: str,
        max_shard_samples: int | None,
        max_shard_bytes: int | None,
        manifest_format: ManifestFormat,
    ) -> None:
        self.root = root
        self.ref = ref
        self.revision = revision
        self.max_shard_samples = max_shard_samples
        self.max_shard_bytes = max_shard_bytes
        self.manifest_format: ManifestFormat = manifest_format
        self.shard_index = 0
        self.shard = _shard_name(self.shard_index)
        self.shard_samples = 0
        self.shard_bytes = 0
        self.entries: list[ViewManifestEntry] = []
        self.tar = self._open_shard(self.shard)
        self.closed = False

    def write(self, sample_id: str, value: Any) -> None:
        payload = payload_for_view(self.ref, sample_id, value, WRITER_PROVIDER)
        if self._should_roll(len(payload.data)):
            self._roll_shard()
        add_payload(self.tar, payload)
        self.shard_samples += 1
        self.shard_bytes += len(payload.data)
        self.entries.append(
            ViewManifestEntry(
                ref=self.ref,
                revision=self.revision,
                sample_id=sample_id,
                shard=self.shard,
                key=payload.key,
                shape=payload.shape,
                dtype=payload.dtype,
                checksum=checksum(payload.data),
                provenance=payload.provenance,
            )
        )

    def close(self) -> None:
        self.close_payload()
        view_json = {
            "schema_version": STORE_SCHEMA_VERSION,
            **view_ref_to_dict(self.ref),
            "revision": self.revision,
            "provider": WRITER_PROVIDER,
        }
        write_json(view_json_path(self.root, self.ref, self.revision), view_json)
        write_view_manifest(
            self.root,
            self.ref,
            self.revision,
            self.entries,
            self.manifest_format,
        )
        view_ready_path(self.root, self.ref, self.revision).touch()

    def close_payload(self) -> None:
        if not self.closed:
            self.tar.close()
            self.closed = True

    def _should_roll(self, payload_bytes: int) -> bool:
        if self.shard_samples == 0:
            return False
        if (
            self.max_shard_samples is not None
            and self.shard_samples >= self.max_shard_samples
        ):
            return True
        return (
            self.max_shard_bytes is not None
            and self.shard_bytes + payload_bytes > self.max_shard_bytes
        )

    def _roll_shard(self) -> None:
        self.tar.close()
        self.shard_index += 1
        self.shard = _shard_name(self.shard_index)
        self.shard_samples = 0
        self.shard_bytes = 0
        self.tar = self._open_shard(self.shard)

    def _open_shard(self, shard: str) -> tarfile.TarFile:
        path = view_shard_path(self.root, self.ref, self.revision, shard)
        path.parent.mkdir(parents=True, exist_ok=True)
        return tarfile.open(path, "w")


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
    _validate_view_ref(ref)
    item = sample.get(ref.sample_ref)
    if item is None:
        return None
    match ref.modality:
        case Modality.AUDIO:
            if not isinstance(item, AudioItem):
                raise TypeError("audio sample items must be AudioItem instances.")
            return item.views.get(_audio_view_key(ref))
        case Modality.IMAGE:
            if not isinstance(item, ImageItem):
                raise TypeError("image sample items must be ImageItem instances.")
            return item.views.get(_image_view_key(ref))
        case Modality.TEXT:
            if not isinstance(item, TextItem):
                raise TypeError("text sample items must be TextItem instances.")
            return item.views.get(_text_view_key(ref))


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


def _validate_view_ref(ref: ViewRef) -> None:
    if not isinstance(ref, ViewRef):
        raise TypeError("views entries must be ViewRef instances.")
    _view_key(ref)


def _view_key(ref: ViewRef) -> View:
    return ref.view_key


def _audio_view_key(ref: ViewRef) -> AudioView:
    if not isinstance(ref.view_key, AudioView):
        raise TypeError("audio view refs must use AudioView keys.")
    return ref.view_key


def _image_view_key(ref: ViewRef) -> ImageView:
    if not isinstance(ref.view_key, ImageView):
        raise TypeError("image view refs must use ImageView keys.")
    return ref.view_key


def _text_view_key(ref: ViewRef) -> TextView:
    if not isinstance(ref.view_key, TextView):
        raise TypeError("text view refs must use TextView keys.")
    return ref.view_key


def _string_key_dict(values: Mapping[Any, Any]) -> dict[str, Any]:
    return {
        str(key.value if hasattr(key, "value") else key): value
        for key, value in values.items()
    }


def _shard_name(index: int) -> str:
    return f"{index:06d}.tar"


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
