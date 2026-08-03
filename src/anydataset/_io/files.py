"""Atomic file replacement and stable file identity helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

StatFingerprint = tuple[int, int, int, int, int]
PortableStatFingerprint = tuple[int, int]


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    durable: bool = True,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            delete=False,
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        ) as file:
            tmp = Path(file.name)
            file.write(data)
            if durable:
                file.flush()
                os.fsync(file.fileno())
        os.replace(tmp, target)
        if durable:
            fsync_directory(target.parent)
    except Exception:
        if tmp is not None and tmp.exists():
            tmp.unlink()
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def stat_fingerprint(stat: os.stat_result) -> StatFingerprint:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def portable_stat_fingerprint(stat: os.stat_result) -> PortableStatFingerprint:
    """Identity for disposable sidecars that survives metadata-preserving copies."""

    return stat.st_size, stat.st_mtime_ns


def fsync_directory(path: str | Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, os.O_RDONLY | flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        return
    finally:
        os.close(fd)
