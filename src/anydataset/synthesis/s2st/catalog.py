from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..._io.files import atomic_write_text
from ...cache import FileLock
from ...dataset import AnyDataset
from ...types import Lang
from .model import (
    PairPlan,
    S2STStage,
    SourceFamily,
    SpeakerVoice,
)

SNAPSHOT_SCHEMA = "anydataset-s2st-snapshot-v1"
CATALOG_SCHEMA = "anydataset-s2st-stage-catalog-v3"
PAIR_INDEX_SCHEMA = "anydataset-s2st-pair-index-v1"
CATALOG_FILE = "catalog.json"
CATALOGS_DIR = "catalogs"
_CATALOG_LOCK_FILE = ".catalog.lock"


@dataclass(frozen=True)
class SnapshotManifest:
    lineage_id: str
    config_revision: str
    revision: int
    stage: S2STStage
    snapshot_id: str
    upstream_snapshot_id: str | None
    upstream_digest: str | None
    previous_snapshot_id: str | None
    added_sources: int
    added_pairs: int
    total_sources: int
    total_pairs: int
    coverage: tuple[tuple[str, int], ...]
    store_path: str
    store_digest: str

    def __post_init__(self) -> None:
        for name, value in (
            ("lineage_id", self.lineage_id),
            ("config_revision", self.config_revision),
            ("snapshot_id", self.snapshot_id),
            ("store_path", self.store_path),
            ("store_digest", self.store_digest),
        ):
            _non_empty(name, value)
        _integer("revision", self.revision, minimum=0)
        if not isinstance(self.stage, S2STStage):
            raise TypeError("stage must be an S2STStage.")
        for name, value in (
            ("added_sources", self.added_sources),
            ("added_pairs", self.added_pairs),
            ("total_sources", self.total_sources),
            ("total_pairs", self.total_pairs),
        ):
            _integer(name, value, minimum=0)
        if self.added_sources > self.total_sources:
            raise ValueError("added_sources cannot exceed total_sources.")
        if self.added_pairs > self.total_pairs:
            raise ValueError("added_pairs cannot exceed total_pairs.")
        _validate_coverage(self.coverage, total_pairs=self.total_pairs)
        if self.stage is S2STStage.SOURCE:
            if (
                self.upstream_snapshot_id is not None
                or self.upstream_digest is not None
            ):
                raise ValueError("source snapshots do not accept an upstream snapshot.")
        elif self.upstream_snapshot_id is None or self.upstream_digest is None:
            raise ValueError("translation and tts snapshots require an exact upstream.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "lineage_id": self.lineage_id,
            "config_revision": self.config_revision,
            "revision": self.revision,
            "stage": self.stage.value,
            "snapshot_id": self.snapshot_id,
            "upstream_snapshot_id": self.upstream_snapshot_id,
            "upstream_digest": self.upstream_digest,
            "previous_snapshot_id": self.previous_snapshot_id,
            "added_sources": self.added_sources,
            "added_pairs": self.added_pairs,
            "total_sources": self.total_sources,
            "total_pairs": self.total_pairs,
            "coverage": {name: count for name, count in self.coverage},
            "store_path": self.store_path,
            "store_digest": self.store_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> SnapshotManifest:
        data = _mapping(value, "snapshot manifest")
        if data.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported S2ST snapshot manifest schema.")
        return cls(
            lineage_id=_string(data.get("lineage_id"), "lineage_id"),
            config_revision=_string(data.get("config_revision"), "config_revision"),
            revision=_integer("revision", data.get("revision"), minimum=0),
            stage=S2STStage(_string(data.get("stage"), "stage")),
            snapshot_id=_string(data.get("snapshot_id"), "snapshot_id"),
            upstream_snapshot_id=_optional_string(
                data.get("upstream_snapshot_id"), "upstream_snapshot_id"
            ),
            upstream_digest=_optional_string(
                data.get("upstream_digest"), "upstream_digest"
            ),
            previous_snapshot_id=_optional_string(
                data.get("previous_snapshot_id"), "previous_snapshot_id"
            ),
            added_sources=_integer(
                "added_sources", data.get("added_sources"), minimum=0
            ),
            added_pairs=_integer("added_pairs", data.get("added_pairs"), minimum=0),
            total_sources=_integer(
                "total_sources", data.get("total_sources"), minimum=0
            ),
            total_pairs=_integer("total_pairs", data.get("total_pairs"), minimum=0),
            coverage=_coverage(data.get("coverage")),
            store_path=_string(data.get("store_path"), "store_path"),
            store_digest=_string(data.get("store_digest"), "store_digest"),
        )


def validate_upstream(child: SnapshotManifest, parent: SnapshotManifest) -> None:
    if child.lineage_id != parent.lineage_id or child.revision != parent.revision:
        raise ValueError("S2ST snapshot upstream must share lineage and revision.")
    expected = {
        S2STStage.TRANSLATION: S2STStage.SOURCE,
        S2STStage.TTS: S2STStage.TRANSLATION,
    }.get(child.stage)
    if expected is None:
        raise ValueError("source snapshots do not have an upstream stage.")
    if parent.stage is not expected:
        raise ValueError(
            f"{child.stage.value} snapshot requires {expected.value} upstream."
        )
    if child.upstream_snapshot_id != parent.snapshot_id:
        raise ValueError("S2ST snapshot upstream id does not match the exact parent.")
    if child.upstream_digest != parent.store_digest:
        raise ValueError(
            "S2ST snapshot upstream digest does not match the exact parent."
        )
    if child.config_revision != parent.config_revision:
        raise ValueError("S2ST snapshot upstream config revision does not match.")
    for name in (
        "added_sources",
        "added_pairs",
        "total_sources",
        "total_pairs",
        "coverage",
    ):
        if getattr(child, name) != getattr(parent, name):
            raise ValueError(f"S2ST snapshot upstream {name} does not match.")


def write_snapshot_manifest(path: str | Path, manifest: SnapshotManifest) -> Path:
    target = Path(path)
    atomic_write_text(
        target,
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n",
    )
    return target


def read_snapshot_manifest(path: str | Path) -> SnapshotManifest:
    return SnapshotManifest.from_dict(_read_json(Path(path), "snapshot manifest"))


@dataclass(frozen=True)
class PairIndexRecord:
    pair_id: str
    source_slot: str
    source_row: int
    source_sequence: int
    source_language: str
    target_language: str
    speaker_id: str | None
    first_for_source: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("pair_id", self.pair_id),
            ("source_slot", self.source_slot),
            ("source_language", self.source_language),
            ("target_language", self.target_language),
        ):
            _non_empty(name, value)
        _integer("source_row", self.source_row, minimum=0)
        _integer("source_sequence", self.source_sequence, minimum=0)
        if self.speaker_id is not None:
            _non_empty("speaker_id", self.speaker_id)
        if type(self.first_for_source) is not bool:
            raise TypeError("first_for_source must be a boolean.")
        source = Lang(self.source_language)
        target = Lang(self.target_language)
        if source is Lang.UND or target is Lang.UND or source is target:
            raise ValueError(
                "pair index languages must be distinct declared languages."
            )
        expected = f"{self.source_slot}:{self.source_row}->{self.target_language}"
        if self.pair_id != expected:
            raise ValueError(f"pair_id must match its source and target: {expected!r}.")

    @classmethod
    def from_plan(
        cls,
        family: SourceFamily,
        pair: PairPlan,
    ) -> PairIndexRecord:
        if pair.key.source != family.key or pair.source_sequence != family.sequence:
            raise ValueError("pair plan does not belong to the supplied source family.")
        speaker = (
            family.voice.speaker_id if isinstance(family.voice, SpeakerVoice) else None
        )
        return cls(
            pair_id=pair.key.id,
            source_slot=family.key.slot,
            source_row=family.key.row,
            source_sequence=family.sequence,
            source_language=family.language.value,
            target_language=pair.key.target_language.value,
            speaker_id=speaker,
            first_for_source=pair.first_for_source,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "source_slot": self.source_slot,
            "source_row": self.source_row,
            "source_sequence": self.source_sequence,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "speaker_id": self.speaker_id,
            "first_for_source": self.first_for_source,
        }

    @classmethod
    def from_dict(cls, value: object) -> PairIndexRecord:
        data = _mapping(value, "pair index record")
        first = data.get("first_for_source")
        if type(first) is not bool:
            raise TypeError("pair index first_for_source must be a boolean.")
        return cls(
            pair_id=_string(data.get("pair_id"), "pair_id"),
            source_slot=_string(data.get("source_slot"), "source_slot"),
            source_row=_integer("source_row", data.get("source_row"), minimum=0),
            source_sequence=_integer(
                "source_sequence", data.get("source_sequence"), minimum=0
            ),
            source_language=_string(data.get("source_language"), "source_language"),
            target_language=_string(data.get("target_language"), "target_language"),
            speaker_id=_optional_string(data.get("speaker_id"), "speaker_id"),
            first_for_source=first,
        )


@dataclass(frozen=True)
class PairIndexSummary:
    source_slot: str
    source_row: int
    source_sequence: int
    source_language: str
    speaker_id: str | None
    target_languages: tuple[str, ...]
    first_target_language: str | None
    first_local_index: int | None

    def __post_init__(self) -> None:
        for name, value in (
            ("source_slot", self.source_slot),
            ("source_language", self.source_language),
        ):
            _non_empty(name, value)
        _integer("source_row", self.source_row, minimum=0)
        _integer("source_sequence", self.source_sequence, minimum=0)
        if self.speaker_id is not None:
            _non_empty("speaker_id", self.speaker_id)
        if not isinstance(self.target_languages, tuple) or not self.target_languages:
            raise TypeError("target_languages must be a non-empty tuple.")
        _unique("target languages", self.target_languages)
        source = Lang(self.source_language)
        if source is Lang.UND:
            raise ValueError("source_language must be a declared language.")
        for target_language in self.target_languages:
            target = Lang(_non_empty("target language", target_language))
            if target is Lang.UND or target is source:
                raise ValueError(
                    "index summary languages must be distinct declared languages."
                )
        if self.first_target_language is not None:
            _non_empty("first_target_language", self.first_target_language)
            if self.first_target_language not in self.target_languages:
                raise ValueError(
                    "first_target_language must be one of target_languages."
                )
            if self.first_local_index is None:
                raise ValueError(
                    "first_local_index is required with first_target_language."
                )
            _integer("first_local_index", self.first_local_index, minimum=0)
        elif self.first_local_index is not None:
            raise ValueError("first_local_index requires a first_target_language.")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_slot": self.source_slot,
            "source_row": self.source_row,
            "source_sequence": self.source_sequence,
            "source_language": self.source_language,
            "speaker_id": self.speaker_id,
            "target_languages": list(self.target_languages),
            "first_target_language": self.first_target_language,
            "first_local_index": self.first_local_index,
        }

    @classmethod
    def from_dict(cls, value: object) -> PairIndexSummary:
        data = _mapping(value, "pair index summary")
        raw_targets = data.get("target_languages")
        if not isinstance(raw_targets, list):
            raise TypeError("pair index summary target_languages must be a list.")
        return cls(
            source_slot=_string(data.get("source_slot"), "source_slot"),
            source_row=_integer("source_row", data.get("source_row"), minimum=0),
            source_sequence=_integer(
                "source_sequence", data.get("source_sequence"), minimum=0
            ),
            source_language=_string(data.get("source_language"), "source_language"),
            speaker_id=_optional_string(data.get("speaker_id"), "speaker_id"),
            target_languages=tuple(
                _string(target, "target language") for target in raw_targets
            ),
            first_target_language=_optional_string(
                data.get("first_target_language"), "first_target_language"
            ),
            first_local_index=_optional_integer(
                "first_local_index", data.get("first_local_index"), minimum=0
            ),
        )


@dataclass(frozen=True)
class FinalCatalogEntry:
    revision: int
    snapshot_id: str
    config_revision: str
    added_sources: int
    total_sources: int
    coverage: tuple[tuple[str, int], ...]
    store_path: str
    store_digest: str
    store_identity: str
    index_path: str
    index_digest: str
    index_identity: str
    index_summary: tuple[PairIndexSummary, ...]
    index_summary_digest: str
    start: int
    sample_count: int

    def __post_init__(self) -> None:
        _integer("revision", self.revision, minimum=0)
        _integer("start", self.start, minimum=0)
        _integer("sample_count", self.sample_count, minimum=1)
        _integer("added_sources", self.added_sources, minimum=0)
        _integer("total_sources", self.total_sources, minimum=0)
        if self.added_sources > self.total_sources:
            raise ValueError("added_sources cannot exceed total_sources.")
        _validate_coverage(
            self.coverage,
            total_pairs=self.start + self.sample_count,
        )
        for name, value in (
            ("snapshot_id", self.snapshot_id),
            ("config_revision", self.config_revision),
            ("store_path", self.store_path),
            ("store_digest", self.store_digest),
            ("store_identity", self.store_identity),
            ("index_path", self.index_path),
            ("index_digest", self.index_digest),
            ("index_identity", self.index_identity),
            ("index_summary_digest", self.index_summary_digest),
        ):
            _non_empty(name, value)
        if not isinstance(self.index_summary, tuple) or not self.index_summary:
            raise TypeError("index_summary must be a non-empty tuple.")
        if any(
            not isinstance(summary, PairIndexSummary) for summary in self.index_summary
        ):
            raise TypeError("index_summary must contain PairIndexSummary values.")
        if (
            sum(len(summary.target_languages) for summary in self.index_summary)
            != self.sample_count
        ):
            raise ValueError("index summary count must match sample_count.")
        first_indexes = tuple(
            summary.first_local_index
            for summary in self.index_summary
            if summary.first_local_index is not None
        )
        _unique("index summary first local indexes", first_indexes)
        if first_indexes and max(first_indexes) >= self.sample_count:
            raise ValueError("index summary first_local_index is outside the entry.")
        if self.index_summary_digest != _index_summary_digest(
            self.index_digest,
            self.index_summary,
        ):
            raise ValueError("index summary digest does not match its catalog data.")

    def to_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "snapshot_id": self.snapshot_id,
            "config_revision": self.config_revision,
            "added_sources": self.added_sources,
            "total_sources": self.total_sources,
            "coverage": {name: count for name, count in self.coverage},
            "store_path": self.store_path,
            "store_digest": self.store_digest,
            "store_identity": self.store_identity,
            "index_path": self.index_path,
            "index_digest": self.index_digest,
            "index_identity": self.index_identity,
            "index_summary": [summary.to_dict() for summary in self.index_summary],
            "index_summary_digest": self.index_summary_digest,
            "start": self.start,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> FinalCatalogEntry:
        data = _mapping(value, "final catalog entry")
        return cls(
            revision=_integer("revision", data.get("revision"), minimum=0),
            snapshot_id=_string(data.get("snapshot_id"), "snapshot_id"),
            config_revision=_string(data.get("config_revision"), "config_revision"),
            added_sources=_integer(
                "added_sources", data.get("added_sources"), minimum=0
            ),
            total_sources=_integer(
                "total_sources", data.get("total_sources"), minimum=0
            ),
            coverage=_coverage(data.get("coverage")),
            store_path=_string(data.get("store_path"), "store_path"),
            store_digest=_string(data.get("store_digest"), "store_digest"),
            store_identity=_string(data.get("store_identity"), "store_identity"),
            index_path=_string(data.get("index_path"), "index_path"),
            index_digest=_string(data.get("index_digest"), "index_digest"),
            index_identity=_string(data.get("index_identity"), "index_identity"),
            index_summary=tuple(
                PairIndexSummary.from_dict(summary)
                for summary in _list(data.get("index_summary"), "index_summary")
            ),
            index_summary_digest=_string(
                data.get("index_summary_digest"), "index_summary_digest"
            ),
            start=_integer("start", data.get("start"), minimum=0),
            sample_count=_integer("sample_count", data.get("sample_count"), minimum=1),
        )


@dataclass(frozen=True)
class FinalCatalog:
    lineage_id: str
    stage: S2STStage = S2STStage.TTS
    entries: tuple[FinalCatalogEntry, ...] = ()
    sealed: bool = False

    def __post_init__(self) -> None:
        _non_empty("lineage_id", self.lineage_id)
        if not isinstance(self.stage, S2STStage):
            raise TypeError("catalog stage must be an S2STStage.")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, FinalCatalogEntry) for entry in self.entries
        ):
            raise TypeError("entries must be a tuple of FinalCatalogEntry values.")
        if type(self.sealed) is not bool:
            raise TypeError("sealed must be a boolean.")
        start = 0
        revision = -1
        snapshots: set[str] = set()
        for entry in self.entries:
            if entry.start != start:
                raise ValueError(
                    "final catalog entries must form one dense append-only history."
                )
            if entry.revision <= revision:
                raise ValueError("final catalog revisions must be strictly increasing.")
            if entry.snapshot_id in snapshots:
                raise ValueError("final catalog snapshot ids must be unique.")
            snapshots.add(entry.snapshot_id)
            start += entry.sample_count
            revision = entry.revision
        _catalog_index(self.entries)

    @property
    def sample_count(self) -> int:
        if not self.entries:
            return 0
        latest = self.entries[-1]
        return latest.start + latest.sample_count

    @property
    def latest_snapshot_id(self) -> str | None:
        return None if not self.entries else self.entries[-1].snapshot_id

    def validate_successor(self, previous: FinalCatalog) -> None:
        if self.lineage_id != previous.lineage_id:
            raise ValueError("S2ST catalog lineage changed during refresh.")
        if self.stage is not previous.stage:
            raise ValueError("S2ST catalog stage changed during refresh.")
        if self.entries[: len(previous.entries)] != previous.entries:
            raise ValueError(
                "final catalog update does not preserve all previous entries."
            )
        if previous.sealed and self != previous:
            raise ValueError("sealed final catalog cannot change.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CATALOG_SCHEMA,
            "lineage_id": self.lineage_id,
            "stage": self.stage.value,
            "sealed": self.sealed,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: object) -> FinalCatalog:
        data = _mapping(value, "final catalog")
        if data.get("schema") != CATALOG_SCHEMA:
            raise ValueError("unsupported S2ST final catalog schema.")
        sealed = data.get("sealed")
        if type(sealed) is not bool:
            raise TypeError("final catalog sealed must be a boolean.")
        raw_entries = data.get("entries")
        if not isinstance(raw_entries, list):
            raise TypeError("final catalog entries must be a list.")
        return cls(
            lineage_id=_string(data.get("lineage_id"), "lineage_id"),
            stage=S2STStage(_string(data.get("stage"), "catalog stage")),
            entries=tuple(FinalCatalogEntry.from_dict(entry) for entry in raw_entries),
            sealed=sealed,
        )


class CatalogPublisher:
    """Atomically append standalone snapshots to one stage catalog."""

    def __init__(
        self,
        root: str | Path,
        lineage_id: str,
        *,
        stage: S2STStage = S2STStage.TTS,
        lock_timeout: float = 3600.0,
    ) -> None:
        self.root = Path(root)
        self.lineage_id = _non_empty("lineage_id", lineage_id)
        if not isinstance(stage, S2STStage):
            raise TypeError("stage must be an S2STStage.")
        self.stage = stage
        if isinstance(lock_timeout, bool) or not isinstance(lock_timeout, (int, float)):
            raise TypeError("lock_timeout must be numeric.")
        if lock_timeout <= 0:
            raise ValueError("lock_timeout must be positive.")
        self.lock_timeout = float(lock_timeout)

    def load(self) -> FinalCatalog:
        return load_catalog(
            self.root,
            lineage_id=self.lineage_id,
            stage=self.stage,
            missing_ok=True,
        )

    def publish(
        self,
        manifest: SnapshotManifest,
        records: Sequence[PairIndexRecord],
    ) -> FinalCatalog:
        with FileLock(
            _catalog_lock_path(self.root, self.stage),
            wait_timeout=self.lock_timeout,
        ):
            return self._publish_locked(manifest, records)

    def _publish_locked(
        self,
        manifest: SnapshotManifest,
        records: Sequence[PairIndexRecord],
    ) -> FinalCatalog:
        if manifest.stage is not self.stage:
            raise ValueError(
                "snapshot stage does not match the target S2ST catalog."
            )
        if manifest.lineage_id != self.lineage_id:
            raise ValueError("snapshot lineage does not match final catalog publisher.")
        values = tuple(records)
        if not values:
            raise ValueError("final catalog publication requires pair index records.")
        if any(not isinstance(record, PairIndexRecord) for record in values):
            raise TypeError("records must contain PairIndexRecord values.")
        if len(values) != manifest.added_pairs:
            raise ValueError("pair index count must match snapshot added_pairs.")
        _unique("pair ids", (record.pair_id for record in values))
        encoded_index = _encode_pair_index(values)
        index_summary = _index_summary(values)

        current = self.load()
        _validate_catalog_resources(self.root, current)
        repeated = _published_entry(
            current,
            manifest,
            encoded_index=encoded_index,
            index_summary=index_summary,
        )
        if repeated is not None:
            return current
        if current.sealed:
            raise RuntimeError("cannot publish after the stage catalog is sealed.")
        if manifest.previous_snapshot_id != current.latest_snapshot_id:
            raise ValueError(
                "snapshot previous id does not match stage catalog latest."
            )
        if manifest.total_pairs != current.sample_count + len(values):
            raise ValueError(
                "snapshot total_pairs does not extend the current stage catalog."
            )
        if current.entries and manifest.revision <= current.entries[-1].revision:
            raise ValueError("snapshot revision must advance the stage catalog.")
        _validate_index_successor(current, index_summary, manifest)

        store = _root_path(self.root, manifest.store_path, "snapshot store_path")
        store_identity = _store_identity(store)
        actual_digest = store_digest(store)
        if _store_identity(store) != store_identity:
            raise ValueError("snapshot store changed while validating publication.")
        if actual_digest != manifest.store_digest:
            raise ValueError(
                "snapshot store digest does not match the manifest: "
                f"{actual_digest} != {manifest.store_digest}."
            )
        dataset = AnyDataset.from_store(store)
        try:
            if len(dataset) != len(values):
                raise ValueError(
                    "snapshot store sample count must match pair index records: "
                    f"{len(dataset)} != {len(values)}."
                )
        finally:
            _close_dataset(dataset)

        index_relpath = Path("indexes") / self.stage.value / (
            f"{manifest.revision:08d}-{manifest.snapshot_id}.jsonl"
        )
        index_path = _root_path(self.root, str(index_relpath), "pair index path")
        if index_path.exists():
            if index_path.read_bytes() != encoded_index:
                raise FileExistsError(
                    f"pair index path contains different data: {index_path}."
                )
        else:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(index_path, encoded_index.decode("utf-8"))
        index_digest = _sha256(encoded_index)
        index_identity = _file_identity(index_path, "pair index")
        entry = FinalCatalogEntry(
            revision=manifest.revision,
            snapshot_id=manifest.snapshot_id,
            config_revision=manifest.config_revision,
            added_sources=manifest.added_sources,
            total_sources=manifest.total_sources,
            coverage=manifest.coverage,
            store_path=manifest.store_path,
            store_digest=manifest.store_digest,
            store_identity=store_identity,
            index_path=str(index_relpath),
            index_digest=index_digest,
            index_identity=index_identity,
            index_summary=index_summary,
            index_summary_digest=_index_summary_digest(
                index_digest,
                index_summary,
            ),
            start=current.sample_count,
            sample_count=len(values),
        )
        updated = FinalCatalog(
            lineage_id=self.lineage_id,
            stage=self.stage,
            entries=(*current.entries, entry),
        )
        if _store_identity(store) != store_identity:
            raise ValueError("snapshot store changed before catalog publication.")
        if _file_identity(index_path, "pair index") != index_identity:
            raise ValueError("pair index changed before catalog publication.")
        write_catalog(self.root, updated)
        return updated

    def seal(self) -> FinalCatalog:
        with FileLock(
            _catalog_lock_path(self.root, self.stage),
            wait_timeout=self.lock_timeout,
        ):
            current = self.load()
            _validate_catalog_resources(self.root, current)
            if current.sealed:
                return current
            if not current.entries:
                raise ValueError("cannot seal an empty final catalog.")
            updated = FinalCatalog(
                lineage_id=current.lineage_id,
                stage=current.stage,
                entries=current.entries,
                sealed=True,
            )
            write_catalog(self.root, updated)
            return updated


def write_catalog(root: str | Path, catalog: FinalCatalog) -> Path:
    target = catalog_path(root, catalog.stage)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        target,
        json.dumps(catalog.to_dict(), sort_keys=True, indent=2) + "\n",
    )
    return target


def load_catalog(
    root: str | Path,
    *,
    lineage_id: str | None = None,
    stage: S2STStage = S2STStage.TTS,
    missing_ok: bool = False,
) -> FinalCatalog:
    if not isinstance(stage, S2STStage):
        raise TypeError("stage must be an S2STStage.")
    path = catalog_path(root, stage)
    if not path.exists():
        if not missing_ok:
            raise FileNotFoundError(path)
        if lineage_id is None:
            raise ValueError(
                "lineage_id is required when a missing catalog is allowed."
            )
        return FinalCatalog(lineage_id=lineage_id, stage=stage)
    catalog = FinalCatalog.from_dict(_read_json(path, "S2ST stage catalog"))
    if lineage_id is not None and catalog.lineage_id != lineage_id:
        raise ValueError("S2ST catalog lineage does not match the requested lineage.")
    if catalog.stage is not stage:
        raise ValueError("S2ST catalog stage does not match the requested stage.")
    return catalog


def catalog_path(root: str | Path, stage: S2STStage) -> Path:
    """Return the publication path for one stage catalog."""

    if not isinstance(stage, S2STStage):
        raise TypeError("stage must be an S2STStage.")
    path = Path(root)
    if stage is S2STStage.TTS:
        return path / CATALOG_FILE
    return path / CATALOGS_DIR / f"{stage.value}.json"


def _catalog_lock_path(root: Path, stage: S2STStage) -> Path:
    if stage is S2STStage.TTS:
        return root / _CATALOG_LOCK_FILE
    return root / f".{stage.value}-catalog.lock"


def catalog_source_locations(
    catalog: FinalCatalog,
    source_keys: Collection[tuple[str, int]],
) -> dict[tuple[str, int], tuple[FinalCatalogEntry, int]]:
    """Locate each published source waveform from catalog summaries only."""

    if not isinstance(catalog, FinalCatalog):
        raise TypeError("catalog must be a FinalCatalog.")
    if not isinstance(source_keys, Collection):
        raise TypeError("source_keys must be a collection.")
    requested: set[tuple[str, int]] = set()
    for key in source_keys:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("source keys must be (slot, row) tuples.")
        slot, row = key
        requested.add(
            (
                _non_empty("source key slot", slot),
                _integer("source key row", row, minimum=0),
            )
        )
    locations: dict[tuple[str, int], tuple[FinalCatalogEntry, int]] = {}
    for entry in catalog.entries:
        for summary in entry.index_summary:
            local_index = summary.first_local_index
            if local_index is None:
                continue
            key = (summary.source_slot, summary.source_row)
            if key in requested:
                locations[key] = (entry, local_index)
        if len(locations) == len(requested):
            break
    return locations


def read_pair_index(
    root: str | Path,
    entry: FinalCatalogEntry,
) -> tuple[PairIndexRecord, ...]:
    path = _root_path(Path(root), entry.index_path, "pair index path")
    identity = _file_identity(path, "pair index")
    if identity != entry.index_identity:
        raise ValueError(f"pair index identity changed: {path}.")
    data = path.read_bytes()
    if _file_identity(path, "pair index") != identity:
        raise ValueError(f"pair index changed while reading: {path}.")
    if _sha256(data) != entry.index_digest:
        raise ValueError(f"pair index digest mismatch: {path}.")
    records: list[PairIndexRecord] = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid pair index JSON at {path}:{line_number}."
            ) from exc
        record = _mapping(value, "pair index row")
        if record.get("schema") != PAIR_INDEX_SCHEMA:
            raise ValueError(f"unsupported pair index schema at {path}:{line_number}.")
        records.append(PairIndexRecord.from_dict(record))
    if len(records) != entry.sample_count:
        raise ValueError("pair index count does not match final catalog entry.")
    _unique("pair ids", (record.pair_id for record in records))
    result = tuple(records)
    if _index_summary(result) != entry.index_summary:
        raise ValueError("pair index does not match its catalog summary.")
    return result


def store_digest(path: str | Path) -> str:
    """Return a content digest for one immutable standalone store directory."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files: list[Path] = []
    for value in root.rglob("*"):
        if value.is_symlink():
            raise ValueError(f"S2ST stores must not contain symbolic links: {value}.")
        if value.is_file():
            files.append(value)
    if not files:
        raise ValueError(f"S2ST store directory is empty: {root}.")
    digest = hashlib.sha256()
    for value in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = value.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(value.stat().st_size.to_bytes(8, "big"))
        with value.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _published_entry(
    catalog: FinalCatalog,
    manifest: SnapshotManifest,
    *,
    encoded_index: bytes,
    index_summary: tuple[PairIndexSummary, ...],
) -> FinalCatalogEntry | None:
    for index, entry in enumerate(catalog.entries):
        if entry.snapshot_id != manifest.snapshot_id:
            continue
        previous = None if index == 0 else catalog.entries[index - 1].snapshot_id
        expected = (
            entry.revision == manifest.revision
            and entry.config_revision == manifest.config_revision
            and entry.added_sources == manifest.added_sources
            and entry.total_sources == manifest.total_sources
            and entry.coverage == manifest.coverage
            and entry.store_path == manifest.store_path
            and entry.store_digest == manifest.store_digest
            and entry.start + entry.sample_count == manifest.total_pairs
            and entry.sample_count == manifest.added_pairs
            and manifest.previous_snapshot_id == previous
            and entry.index_digest == _sha256(encoded_index)
            and entry.index_summary == index_summary
        )
        if not expected:
            raise ValueError(
                "published snapshot id was reused with different catalog data."
            )
        return entry
    return None


def _validate_index_successor(
    catalog: FinalCatalog,
    added: tuple[PairIndexSummary, ...],
    manifest: SnapshotManifest,
) -> None:
    state = _catalog_index(catalog.entries)
    _apply_index_summary(
        state,
        added,
        added_sources=manifest.added_sources,
        total_sources=manifest.total_sources,
        added_pairs=manifest.added_pairs,
    )


@dataclass
class _SourceIndexState:
    source_slot: str
    source_row: int
    source_language: str
    speaker_id: str | None
    targets: set[str]


@dataclass
class _CatalogIndexState:
    sources: list[_SourceIndexState]
    sequences_by_source: dict[tuple[str, int], int]
    pair_count: int = 0


def _catalog_index(
    entries: tuple[FinalCatalogEntry, ...],
) -> _CatalogIndexState:
    state = _CatalogIndexState([], {})
    for entry in entries:
        if entry.start != state.pair_count:
            raise ValueError("final catalog index summary is not append-only.")
        _apply_index_summary(
            state,
            entry.index_summary,
            added_sources=entry.added_sources,
            total_sources=entry.total_sources,
            added_pairs=entry.sample_count,
        )
    return state


def _apply_index_summary(
    state: _CatalogIndexState,
    summaries: tuple[PairIndexSummary, ...],
    *,
    added_sources: int,
    total_sources: int,
    added_pairs: int,
) -> None:
    seen_sequences: set[int] = set()
    new_sources = 0
    pair_count = 0
    for summary in summaries:
        sequence = summary.source_sequence
        if sequence in seen_sequences:
            raise ValueError("index summary must group each source exactly once.")
        seen_sequences.add(sequence)
        source = (summary.source_slot, summary.source_row)
        identity = (
            summary.source_language,
            summary.speaker_id,
        )
        if sequence < len(state.sources):
            existing = state.sources[sequence]
            if (existing.source_slot, existing.source_row) != source:
                raise ValueError(
                    "pair index source_sequence maps to multiple source keys."
                )
            if (existing.source_language, existing.speaker_id) != identity:
                raise ValueError("pair index source identity changed across revisions.")
            if summary.first_target_language is not None:
                raise ValueError(
                    "existing pair index source cannot add another first record."
                )
        elif sequence == len(state.sources):
            if source in state.sequences_by_source:
                raise ValueError("pair index source key maps to multiple sequences.")
            if summary.first_target_language is None:
                raise ValueError(
                    "new pair index source must contain exactly one first record."
                )
            state.sequences_by_source[source] = sequence
            state.sources.append(
                _SourceIndexState(
                    source_slot=summary.source_slot,
                    source_row=summary.source_row,
                    source_language=summary.source_language,
                    speaker_id=summary.speaker_id,
                    targets=set(),
                )
            )
            new_sources += 1
        else:
            raise ValueError(
                "pair index source sequences must be contiguous from zero."
            )
        source_state = state.sources[sequence]
        overlap = source_state.targets.intersection(summary.target_languages)
        if overlap:
            pair_id = f"{summary.source_slot}:{summary.source_row}->{min(overlap)}"
            raise ValueError(f"final catalog pair id already exists: {pair_id}.")
        source_state.targets.update(summary.target_languages)
        pair_count += len(summary.target_languages)
    if pair_count != added_pairs:
        raise ValueError("index summary pair count does not match added_pairs.")
    if new_sources != added_sources:
        raise ValueError("pair index new source count does not match added_sources.")
    if len(state.sources) != total_sources:
        raise ValueError("pair index source count does not match total_sources.")
    state.pair_count += pair_count


def _index_summary(
    records: tuple[PairIndexRecord, ...],
) -> tuple[PairIndexSummary, ...]:
    grouped: dict[int, list[tuple[int, PairIndexRecord]]] = {}
    order: list[int] = []
    for local_index, record in enumerate(records):
        if record.source_sequence not in grouped:
            grouped[record.source_sequence] = []
            order.append(record.source_sequence)
        grouped[record.source_sequence].append((local_index, record))
    summaries: list[PairIndexSummary] = []
    for sequence in order:
        values = grouped[sequence]
        first = values[0][1]
        identity = (
            first.source_slot,
            first.source_row,
            first.source_language,
            first.speaker_id,
        )
        if any(
            (
                value.source_slot,
                value.source_row,
                value.source_language,
                value.speaker_id,
            )
            != identity
            for _, value in values[1:]
        ):
            raise ValueError("pair index source identity differs within one revision.")
        first_targets = [
            (local_index, value.target_language)
            for local_index, value in values
            if value.first_for_source
        ]
        if len(first_targets) > 1:
            raise ValueError(
                "pair index source must contain at most one first record per revision."
            )
        summaries.append(
            PairIndexSummary(
                source_slot=first.source_slot,
                source_row=first.source_row,
                source_sequence=sequence,
                source_language=first.source_language,
                speaker_id=first.speaker_id,
                target_languages=tuple(value.target_language for _, value in values),
                first_target_language=(
                    None if not first_targets else first_targets[0][1]
                ),
                first_local_index=(None if not first_targets else first_targets[0][0]),
            )
        )
    return tuple(summaries)


def _encode_pair_index(records: tuple[PairIndexRecord, ...]) -> bytes:
    return "".join(
        json.dumps(
            {"schema": PAIR_INDEX_SCHEMA, **record.to_dict()},
            sort_keys=True,
        )
        + "\n"
        for record in records
    ).encode("utf-8")


def _index_summary_digest(
    index_digest: str,
    summaries: tuple[PairIndexSummary, ...],
) -> str:
    encoded = json.dumps(
        {
            "index_digest": index_digest,
            "summaries": [summary.to_dict() for summary in summaries],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(encoded)


def _validate_catalog_resources(root: Path, catalog: FinalCatalog) -> None:
    for entry in catalog.entries:
        validate_catalog_entry(root, entry)


def validate_catalog_entry(root: str | Path, entry: FinalCatalogEntry) -> None:
    """Validate one entry's immutable index and store identities."""

    if not isinstance(entry, FinalCatalogEntry):
        raise TypeError("entry must be a FinalCatalogEntry.")
    index = _root_path(Path(root), entry.index_path, "pair index path")
    if _file_identity(index, "pair index") != entry.index_identity:
        raise ValueError(f"pair index identity changed: {index}.")
    validate_catalog_store(root, entry)


def validate_catalog_store(root: str | Path, entry: FinalCatalogEntry) -> None:
    """Validate an immutable store without rereading its waveform payloads."""

    if not isinstance(entry, FinalCatalogEntry):
        raise TypeError("entry must be a FinalCatalogEntry.")
    store = _root_path(Path(root), entry.store_path, "snapshot store_path")
    actual = _store_identity(store)
    if actual != entry.store_identity:
        raise ValueError(
            f"final catalog store identity changed: {actual} != {entry.store_identity}."
        )


def _store_identity(root: Path) -> str:
    if root.is_symlink():
        raise ValueError(f"S2ST stores must not contain symbolic links: {root}.")
    if not root.is_dir():
        raise FileNotFoundError(root)
    values = [root, *root.rglob("*")]
    return _stat_identity(root, values, "S2ST store")


def _file_identity(path: Path, name: str) -> str:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link: {path}.")
    if not path.is_file():
        raise FileNotFoundError(path)
    return _stat_identity(path.parent, [path], name)


def _stat_identity(root: Path, values: list[Path], name: str) -> str:
    digest = hashlib.sha256()
    for value in sorted(values, key=lambda item: item.relative_to(root).as_posix()):
        if value.is_symlink():
            raise ValueError(f"{name} must not contain symbolic links: {value}.")
        metadata = value.lstat()
        mode = metadata.st_mode
        kind = (
            "directory"
            if stat.S_ISDIR(mode)
            else "file"
            if stat.S_ISREG(mode)
            else "other"
        )
        relative = value.relative_to(root).as_posix() or "."
        encoded = json.dumps(
            (
                relative,
                kind,
                mode,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            ),
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _root_path(root: Path, value: str, name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be relative to the S2ST root.")
    target = root / path
    if not target.exists() and name == "snapshot store_path":
        raise FileNotFoundError(target)
    return target


def _close_dataset(dataset: AnyDataset) -> None:
    prepared = dataset.dataset
    close = getattr(prepared, "close", None)
    if callable(close):
        close()


def _read_json(path: Path, name: str) -> object:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {name} JSON: {path}.") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list.")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _optional_integer(name: str, value: object, *, minimum: int) -> int | None:
    return None if value is None else _integer(name, value, minimum=minimum)


def _coverage(value: object) -> tuple[tuple[str, int], ...]:
    data = _mapping(value, "coverage")
    return tuple(
        sorted(
            (
                _non_empty("coverage name", name),
                _integer("coverage count", count, minimum=0),
            )
            for name, count in data.items()
        )
    )


def _validate_coverage(
    coverage: tuple[tuple[str, int], ...],
    *,
    total_pairs: int,
) -> None:
    if not isinstance(coverage, tuple) or any(
        not isinstance(entry, tuple) or len(entry) != 2 for entry in coverage
    ):
        raise TypeError("coverage must be a tuple of (name, count) pairs.")
    names: list[str] = []
    total = 0
    for name, count in coverage:
        names.append(_non_empty("coverage name", name))
        total += _integer("coverage count", count, minimum=0)
    _unique("coverage names", names)
    if tuple(sorted(coverage)) != coverage:
        raise ValueError("coverage entries must be sorted by name.")
    if total != total_pairs:
        raise ValueError("coverage counts must sum to total_pairs.")


def _non_empty(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return value


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _unique(name: str, values) -> None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{name} must be unique; duplicate {value!r}.")
        seen.add(value)


__all__ = [
    "CATALOG_FILE",
    "CATALOG_SCHEMA",
    "PAIR_INDEX_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "CatalogPublisher",
    "FinalCatalog",
    "FinalCatalogEntry",
    "PairIndexRecord",
    "SnapshotManifest",
    "catalog_source_locations",
    "load_catalog",
    "read_pair_index",
    "read_snapshot_manifest",
    "store_digest",
    "validate_catalog_entry",
    "validate_catalog_store",
    "validate_upstream",
    "write_snapshot_manifest",
]
