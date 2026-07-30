from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from .._io.files import atomic_write_text


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    target = Path(path)
    atomic_write_text(
        target,
        json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
    )
