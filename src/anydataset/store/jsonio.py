from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol


class _TextWriter(Protocol):
    def write(self, text: str, /) -> object: ...


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    target = Path(path)
    _atomic_write_text(
        target,
        json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    def write(file: _TextWriter) -> None:
        file.write(text)

    _atomic_write(path, write)


def _atomic_write(path: Path, write: Callable[[_TextWriter], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=path.parent,
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as file:
            tmp_path = Path(file.name)
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise
