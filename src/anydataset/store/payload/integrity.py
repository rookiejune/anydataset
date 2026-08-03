from __future__ import annotations

import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Literal

from ..._validation import validate_path_segment
from ...types.item import Modality, Role, View
from .._refs import validate_entry_ref, view_path
from ..manifest.io import read_view_manifest
from ..paths import view_shard_path
from ..reader import read_store_views

IntegrityLevel = Literal["fast", "normal", "full"]
_EXPECTED_KEYS_MEMORY_LIMIT = 65_536


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
    with _ExpectedPayloads() as expected:
        for entry in read_view_manifest(root, view):
            validate_entry_ref((entry.role, entry.modality, entry.view), view)
            if not _path_segment("shard", entry.shard):
                raise ValueError(
                    f"View {view_path(view)} has invalid shard name {entry.shard!r}."
                )
            if not _path_segment("payload key", entry.key):
                raise ValueError(
                    f"View {view_path(view)} has invalid payload key {entry.key!r}."
                )
            if not expected.add(entry.shard, entry.key):
                raise ValueError(
                    f"View {view_path(view)} shard {entry.shard!r} "
                    f"has duplicate payload key {entry.key!r}."
                )

        for shard in expected.shards():
            path = view_shard_path(root, view, shard)
            if not path.is_file():
                raise FileNotFoundError(
                    f"View {view_path(view)} is missing referenced shard {path}."
                )
            if level == "fast":
                continue
            expected_keys = expected.keys(shard) if level == "full" else None
            try:
                with tarfile.open(path, "r|") as archive:
                    members: set[str] = set()
                    for member in archive:
                        if not member.isfile():
                            continue
                        if member.name in members:
                            raise ValueError(
                                f"View {view_path(view)} shard {shard!r} "
                                f"has duplicate payload key {member.name!r}."
                            )
                        members.add(member.name)
                        if not _path_segment("payload key", member.name):
                            raise ValueError(
                                f"View {view_path(view)} shard {shard!r} "
                                f"has invalid payload key {member.name!r}."
                            )
            except tarfile.TarError as exc:
                raise ValueError(
                    f"View shard is not a valid tar archive: {path}"
                ) from exc
            if level == "full":
                if expected_keys is None:
                    raise RuntimeError("Expected payload keys were not loaded.")
                missing = expected_keys - members
                if missing:
                    raise ValueError(
                        f"View {view_path(view)} shard {shard!r} "
                        f"is missing payload {min(missing)!r}."
                    )
                extra = members - expected_keys
                if extra:
                    raise ValueError(
                        f"View {view_path(view)} shard {shard!r} "
                        f"has extra payload {min(extra)!r}."
                    )
                try:
                    with tarfile.open(path, "r|") as archive:
                        for member in archive:
                            if member.isfile() and member.name in expected_keys:
                                _read_payload_member(archive, member, path)
                except tarfile.TarError as exc:
                    raise ValueError(
                        f"View shard is not a valid tar archive: {path}"
                    ) from exc


class _ExpectedPayloads:
    def __init__(self) -> None:
        self._keys: dict[str, set[str]] | None = {}
        self._shards: set[str] = set()
        self._count = 0
        self._pending = 0
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._database: sqlite3.Connection | None = None

    def __enter__(self) -> _ExpectedPayloads:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def add(self, shard: str, key: str) -> bool:
        self._shards.add(shard)
        if self._database is not None:
            try:
                self._database.execute(
                    "INSERT INTO expected(shard, key) VALUES (?, ?)",
                    (shard, key),
                )
            except sqlite3.IntegrityError:
                return False
            self._pending += 1
            if self._pending >= 8192:
                self._database.commit()
                self._pending = 0
            return True

        if self._keys is None:
            raise RuntimeError("Expected payload storage is unavailable.")
        keys = self._keys.setdefault(shard, set())
        if key in keys:
            return False
        keys.add(key)
        self._count += 1
        if self._count > _EXPECTED_KEYS_MEMORY_LIMIT:
            self._spill()
        return True

    def shards(self) -> tuple[str, ...]:
        if self._database is not None:
            self._database.commit()
            self._pending = 0
        return tuple(sorted(self._shards))

    def keys(self, shard: str) -> set[str]:
        if self._database is not None:
            return {
                str(row[0])
                for row in self._database.execute(
                    "SELECT key FROM expected WHERE shard = ?",
                    (shard,),
                )
            }
        if self._keys is None:
            raise RuntimeError("Expected payload storage is unavailable.")
        return self._keys[shard]

    def close(self) -> None:
        if self._database is not None:
            self._database.close()
            self._database = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def _spill(self) -> None:
        keys_by_shard = self._keys
        if keys_by_shard is None:
            return
        self._temporary = tempfile.TemporaryDirectory(
            prefix="anydataset-payload-integrity-"
        )
        database = sqlite3.connect(Path(self._temporary.name) / "expected.sqlite")
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute(
            "CREATE TABLE expected ("
            "shard TEXT NOT NULL, key TEXT NOT NULL, "
            "PRIMARY KEY (shard, key)"
            ") WITHOUT ROWID"
        )
        database.executemany(
            "INSERT INTO expected(shard, key) VALUES (?, ?)",
            (
                (shard, key)
                for shard, keys in keys_by_shard.items()
                for key in keys
            ),
        )
        database.commit()
        self._keys = None
        self._database = database
        self._pending = 0


def _read_payload_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    path: Path,
) -> None:
    try:
        payload = archive.extractfile(member)
        if payload is None:
            raise ValueError(f"View shard is missing payload body: {path}")
        payload.read()
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"View shard payload could not be read: {path}") from exc


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
