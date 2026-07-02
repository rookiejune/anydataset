from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

ValueT = TypeVar("ValueT")


def replace_dir(target: str | Path, write: Callable[[Path], ValueT]) -> Path:
    target = Path(target)
    validate_empty_target(target)
    tmp = tmp_dir(target)
    tmp.mkdir(parents=True)
    try:
        write(tmp)
        if target.exists():
            target.rmdir()
        os.replace(tmp, target)
        return target
    except Exception:
        cleanup_dir(tmp)
        raise


def replace_existing_dir(target: str | Path, write: Callable[[Path], ValueT]) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir(target)
    cleanup_dir(tmp)
    tmp.mkdir(parents=True)
    try:
        write(tmp)
        cleanup_dir(target)
        os.replace(tmp, target)
        return target
    except Exception:
        cleanup_dir(tmp)
        raise


def validate_empty_target(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Target path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"Target directory must be empty: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def tmp_dir(path: Path) -> Path:
    return path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"


def cleanup_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
