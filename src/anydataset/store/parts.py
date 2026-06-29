from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .._sharding import validate_shard
from .._validation import positive_int
from ..types.item import Modality, Role, Sample, View
from .atomic import replace_dir
from .jsonio import read_json, write_json
from .manifest import (
    DatasetManifest,
    SampleItem,
    SampleManifestEntry,
    ViewManifestEntry,
    string_key_dict,
)
from .manifestio import (
    read_view_manifest,
    sample_manifest_writer,
    write_samples_manifest,
    write_view_manifest,
)
from .paths import (
    dataset_json_path,
    dataset_ready_path,
    view_ready_path,
    view_shard_path,
    view_shards_dir,
)
from .reader import read_store_dataset
from .viewwriter import ViewWriter
from .writer import (
    DEFAULT_MAX_SHARD_SAMPLES,
    _sample_id,
    _sample_view_refs,
    _sample_view_value,
    _validate_sample,
    _validate_view_sets,
    _view_path,
)

type IndexedSample = tuple[int, Sample]


@dataclass
class DatasetPartWriter:
    output_dir: str | Path
    dataset_id: str
    shard_id: int
    num_shards: int
    split: str | None = None
    views: tuple[tuple[Role, Modality, View], ...] | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES
    shard_prefix: str | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        validate_shard(self.num_shards, self.shard_id)
        self.max_shard_samples = positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )

    def write(self, samples: Iterable[IndexedSample]) -> Path:
        return replace_dir(
            self.output_dir, lambda tmp: self._write_to_tmp(tmp, samples)
        )

    def _write_to_tmp(self, root: Path, samples: Iterable[IndexedSample]) -> Path:
        sinks: dict[tuple[Role, Modality, View], ViewWriter] = {}
        sample_views: dict[tuple[Role, Modality], frozenset[View]] = {}
        sample_manifest = sample_manifest_writer(root)
        sample_count = 0
        seen_indexes: set[int] = set()

        try:
            for sample_index, sample in samples:
                if sample_index in seen_indexes:
                    raise ValueError(
                        f"Duplicate materialized sample index {sample_index}."
                    )
                seen_indexes.add(sample_index)
                if not isinstance(sample, Mapping):
                    raise TypeError("DatasetPartWriter.write expects Sample mappings.")
                sample_id = _sample_id(self.dataset_id, sample_index)
                _validate_sample(sample)
                views = (
                    self.views if self.views is not None else _sample_view_refs(sample)
                )
                if not views:
                    raise ValueError(f"Sample {sample_id} has no views.")
                if self.views is None:
                    _validate_view_sets(sample, sample_views, sample_id)
                sample_manifest.write(
                    _sample_manifest_entry(sample, sample_id, sample_index)
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
                            shard_prefix=self._shard_prefix(),
                        )
                        sinks[view] = sink
                    sink.write(sample_id, value)

            manifest = DatasetManifest(
                dataset_id=self.dataset_id,
                split=self.split,
                sample_count=sample_count,
            )
            write_json(dataset_json_path(root), asdict(manifest))
            write_json(
                _part_json_path(root),
                {
                    "dataset_id": self.dataset_id,
                    "split": self.split,
                    "num_shards": self.num_shards,
                    "shard_id": self.shard_id,
                    "sample_count": sample_count,
                },
            )
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

    def _shard_prefix(self) -> str:
        if self.shard_prefix is not None:
            return self.shard_prefix
        return f"part-{self.shard_id:05d}-"


@dataclass
class DatasetFragmentWriter:
    output_dir: str | Path
    dataset_id: str
    fragment_id: str
    split: str | None = None
    max_shard_samples: int = DEFAULT_MAX_SHARD_SAMPLES

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        _validate_fragment_id(self.fragment_id)
        self.max_shard_samples = positive_int(
            "max_shard_samples",
            self.max_shard_samples,
        )

    def write(self, samples: Sequence[IndexedSample]) -> Path:
        if not samples:
            raise ValueError("DatasetFragmentWriter.write requires samples.")
        indexes = tuple(index for index, _ in samples)
        if len(set(indexes)) != len(indexes):
            raise ValueError("Dataset fragment sample indexes must be unique.")
        return replace_dir(
            self.output_dir,
            lambda tmp: self._write_to_tmp(tmp, tuple(samples), indexes),
        )

    def _write_to_tmp(
        self,
        root: Path,
        samples: tuple[IndexedSample, ...],
        indexes: tuple[int, ...],
    ) -> Path:
        DatasetPartWriter(
            root,
            dataset_id=self.dataset_id,
            split=self.split,
            shard_id=0,
            num_shards=1,
            max_shard_samples=self.max_shard_samples,
            shard_prefix=f"fragment-{self.fragment_id}-",
        )._write_to_tmp(root, samples)
        write_json(
            _fragment_json_path(root),
            {
                "dataset_id": self.dataset_id,
                "split": self.split,
                "fragment_id": self.fragment_id,
                "sample_count": len(indexes),
                "sample_indexes": list(indexes),
            },
        )
        return root


def commit_store_parts(
    output_dir: str | Path,
    parts_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None = None,
) -> Path:
    parts = _part_roots(parts_dir)
    if not parts:
        raise ValueError(f"No materialized parts found: {parts_dir}")
    return replace_dir(
        output_dir,
        lambda tmp: _commit_to_tmp(tmp, parts, dataset_id=dataset_id, split=split),
    )


def commit_store_fragments(
    output_dir: str | Path,
    fragments_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None = None,
    expected_sample_count: int | None = None,
) -> Path:
    if expected_sample_count is not None and expected_sample_count < 0:
        raise ValueError("expected_sample_count must be non-negative.")
    fragments = _fragment_roots(
        fragments_dir,
        dataset_id=dataset_id,
        split=split,
    )
    if not fragments:
        raise ValueError(f"No materialized fragments found: {fragments_dir}")
    return replace_dir(
        output_dir,
        lambda tmp: _commit_fragments_to_tmp(
            tmp,
            fragments,
            dataset_id=dataset_id,
            split=split,
            expected_sample_count=expected_sample_count,
        ),
    )


def completed_fragment_indexes(
    fragments_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None = None,
) -> frozenset[int]:
    if not Path(fragments_dir).is_dir():
        return frozenset()
    indexes: set[int] = set()
    for fragment in _fragment_roots(
        fragments_dir,
        dataset_id=dataset_id,
        split=split,
    ):
        data = read_json(_fragment_json_path(fragment))
        for index in _fragment_sample_indexes(data):
            if index in indexes:
                raise ValueError(f"Duplicate materialized fragment index {index}.")
            indexes.add(index)
    return frozenset(indexes)


def _commit_to_tmp(
    root: Path,
    parts: tuple[Path, ...],
    *,
    dataset_id: str,
    split: str | None,
) -> Path:
    _validate_parts(parts, dataset_id, split)
    return _commit_roots_to_tmp(root, parts, dataset_id=dataset_id, split=split)


def _commit_fragments_to_tmp(
    root: Path,
    fragments: tuple[Path, ...],
    *,
    dataset_id: str,
    split: str | None,
    expected_sample_count: int | None,
) -> Path:
    return _commit_roots_to_tmp(
        root,
        fragments,
        dataset_id=dataset_id,
        split=split,
        expected_sample_count=expected_sample_count,
    )


def _commit_roots_to_tmp(
    root: Path,
    stores: tuple[Path, ...],
    *,
    dataset_id: str,
    split: str | None,
    expected_sample_count: int | None = None,
) -> Path:
    sample_entries: list[SampleManifestEntry] = []
    view_entries: dict[tuple[Role, Modality, View], list[ViewManifestEntry]] = {}

    for store in stores:
        dataset = read_store_dataset(store)
        sample_entries.extend(dataset.samples)
        for view in dataset.views:
            entries = view_entries.setdefault(view, [])
            entries.extend(read_view_manifest(store, view))

    ordered = _ordered_samples(sample_entries)
    if expected_sample_count is not None:
        _validate_expected_indexes(ordered, expected_sample_count)
    write_samples_manifest(root, _renumber_samples(ordered))
    for view, entries in sorted(
        view_entries.items(), key=lambda item: _view_path(item[0])
    ):
        write_view_manifest(root, view, _ordered_view_entries(entries, ordered))
        for store in stores:
            _copy_view_shards(store, root, view)
        view_ready_path(root, view).touch()

    write_json(
        dataset_json_path(root),
        asdict(
            DatasetManifest(
                dataset_id=dataset_id,
                split=split,
                sample_count=len(ordered),
            )
        ),
    )
    dataset_ready_path(root).touch()
    read_store_dataset(root)
    return root


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


def _item_entry(ref, item) -> SampleItem:
    return ref, string_key_dict(item.meta)


def _part_roots(parts_dir: str | Path) -> tuple[Path, ...]:
    root = Path(parts_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return tuple(
        sorted(
            (path for path in root.iterdir() if _part_json_path(path).is_file()),
            key=lambda path: _part_sort_key(path),
        )
    )


def _fragment_roots(
    fragments_dir: str | Path,
    *,
    dataset_id: str,
    split: str | None,
) -> tuple[Path, ...]:
    root = Path(fragments_dir).expanduser()
    if not root.is_dir():
        return ()
    fragments = tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir()
                if not path.name.startswith(".")
                if _fragment_json_path(path).is_file()
            ),
            key=_fragment_sort_key,
        )
    )
    for fragment in fragments:
        _validate_fragment(fragment, dataset_id, split)
    return fragments


def _part_sort_key(path: Path) -> tuple[int, str]:
    data = read_json(_part_json_path(path))
    return int(data["shard_id"]), path.name


def _validate_parts(
    parts: tuple[Path, ...],
    dataset_id: str,
    split: str | None,
) -> None:
    num_shards: int | None = None
    shard_ids: set[int] = set()
    for part in parts:
        data = read_json(_part_json_path(part))
        if data.get("dataset_id") != dataset_id:
            raise ValueError(f"Part {part} dataset_id does not match {dataset_id!r}.")
        if data.get("split") != split:
            raise ValueError(f"Part {part} split does not match {split!r}.")
        part_num_shards = int(data["num_shards"])
        shard_id = int(data["shard_id"])
        validate_shard(part_num_shards, shard_id)
        if num_shards is None:
            num_shards = part_num_shards
        elif num_shards != part_num_shards:
            raise ValueError("Materialized parts disagree on num_shards.")
        if shard_id in shard_ids:
            raise ValueError(f"Duplicate materialized part for shard_id {shard_id}.")
        shard_ids.add(shard_id)
    if num_shards is not None and shard_ids != set(range(num_shards)):
        missing = sorted(set(range(num_shards)) - shard_ids)
        raise ValueError(f"Missing materialized part for shard_id {missing[0]}.")


def _validate_fragment(path: Path, dataset_id: str, split: str | None) -> None:
    data = read_json(_fragment_json_path(path))
    if data.get("dataset_id") != dataset_id:
        raise ValueError(f"Fragment {path} dataset_id does not match {dataset_id!r}.")
    if data.get("split") != split:
        raise ValueError(f"Fragment {path} split does not match {split!r}.")
    if data.get("fragment_id") != path.name:
        raise ValueError(f"Fragment {path} id does not match its directory name.")
    dataset = read_store_dataset(path)
    expected = set(_fragment_sample_indexes(data))
    actual = {sample.sample_index for sample in dataset.samples}
    if actual != expected:
        raise ValueError(f"Fragment {path} sample indexes do not match its metadata.")


def _validate_expected_indexes(
    entries: tuple[SampleManifestEntry, ...],
    expected_sample_count: int,
) -> None:
    if expected_sample_count < 0:
        raise ValueError("expected_sample_count must be non-negative.")
    actual = {entry.sample_index for entry in entries}
    expected = set(range(expected_sample_count))
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing sample_index {missing[0]}")
    if extra:
        details.append(f"unexpected sample_index {extra[0]}")
    raise ValueError("Materialized fragments coverage mismatch: " + ", ".join(details))


def _fragment_sample_indexes(data: Mapping[str, object]) -> tuple[int, ...]:
    raw = data.get("sample_indexes")
    if not isinstance(raw, list):
        raise ValueError("Fragment sample_indexes must be a list.")
    indexes: list[int] = []
    for value in raw:
        if not isinstance(value, int):
            raise ValueError("Fragment sample_indexes entries must be integers.")
        indexes.append(value)
    if data.get("sample_count") != len(indexes):
        raise ValueError("Fragment sample_count does not match sample_indexes.")
    return tuple(indexes)


def _fragment_sort_key(path: Path) -> tuple[int, str]:
    data = read_json(_fragment_json_path(path))
    indexes = _fragment_sample_indexes(data)
    return min(indexes), path.name


def _validate_fragment_id(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("fragment_id must be a string.")
    if value in {"", ".", ".."}:
        raise ValueError("fragment_id must be a non-empty path segment.")
    if "/" in value:
        raise ValueError("fragment_id cannot contain '/'.")


def _copy_view_shards(
    source_root: Path,
    target_root: Path,
    view: tuple[Role, Modality, View],
) -> None:
    source_dir = view_shards_dir(source_root, view)
    if not source_dir.is_dir():
        return
    for source in sorted(source_dir.iterdir()):
        if not source.is_file():
            continue
        target = view_shard_path(target_root, view, source.name)
        if target.exists():
            raise ValueError(
                f"Duplicate view shard {source.name!r} for {_view_path(view)}."
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if not target.is_file():
            raise FileNotFoundError(target)


def _ordered_samples(
    entries: Iterable[SampleManifestEntry],
) -> tuple[SampleManifestEntry, ...]:
    sample_ids: set[str] = set()
    indexes: set[int] = set()
    ordered = sorted(entries, key=lambda entry: entry.sample_index)
    for entry in ordered:
        if entry.sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id {entry.sample_id!r}.")
        if entry.sample_index in indexes:
            raise ValueError(f"Duplicate sample_index {entry.sample_index}.")
        sample_ids.add(entry.sample_id)
        indexes.add(entry.sample_index)
    return tuple(ordered)


def _renumber_samples(
    entries: Iterable[SampleManifestEntry],
) -> Iterable[SampleManifestEntry]:
    for index, entry in enumerate(entries):
        yield SampleManifestEntry(
            sample_id=entry.sample_id,
            sample_index=index,
            items=entry.items,
        )


def _ordered_view_entries(
    entries: Iterable[ViewManifestEntry],
    samples: tuple[SampleManifestEntry, ...],
) -> Iterable[ViewManifestEntry]:
    sample_order = {sample.sample_id: index for index, sample in enumerate(samples)}
    seen: set[str] = set()
    ordered = sorted(entries, key=lambda entry: sample_order.get(entry.sample_id, -1))
    for entry in ordered:
        if entry.sample_id not in sample_order:
            raise ValueError(f"View entry has unknown sample_id {entry.sample_id!r}.")
        if entry.sample_id in seen:
            raise ValueError(f"Duplicate view entry for sample_id {entry.sample_id!r}.")
        seen.add(entry.sample_id)
        yield entry


def _part_json_path(root: str | Path) -> Path:
    return Path(root) / "part.json"


def _fragment_json_path(root: str | Path) -> Path:
    return Path(root) / "fragment.json"
