from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any, TextIO


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    target = Path(path)
    _atomic_write_text(
        target,
        json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
    )


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if text == "":
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} must be an object.")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    _atomic_write(target, lambda file: _write_jsonl_rows(file, rows))


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write(path, lambda file: file.write(text))


def _write_jsonl_rows(file: TextIO, rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        file.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")


def _atomic_write(path: Path, write: Callable[[TextIO], None]) -> None:
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
