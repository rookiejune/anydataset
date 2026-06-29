"""Internal run-scoped logging paths.

The module owns process-local log run directory discovery under
`ANYDATASET_HOME`. It does not configure application-wide Python logging or
store sample-level audit metrics.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .cache import anydataset_home

_RUN_DIR: Path | None = None
_RUN_HOME: Path | None = None
_RUN_OVERRIDE: Path | None = None
_RUN_DIR_LOCK = threading.Lock()


def run_logs_dir() -> Path:
    global _RUN_DIR, _RUN_HOME
    with _RUN_DIR_LOCK:
        if _RUN_OVERRIDE is not None:
            _RUN_OVERRIDE.mkdir(parents=True, exist_ok=True)
            return _RUN_OVERRIDE
        home = anydataset_home()
        if _RUN_DIR is None or _RUN_HOME != home:
            _RUN_HOME = home
            _RUN_DIR = _new_run_logs_dir(home)
        _RUN_DIR.mkdir(parents=True, exist_ok=True)
        return _RUN_DIR


def write_info(source: str, message: str) -> None:
    _write_log(source, "INFO", message)


def write_warning(source: str, message: str) -> None:
    _write_log(source, "WARNING", message)


def _write_log(source: str, level: str, message: str) -> None:
    path = run_logs_dir() / f"{source}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{timestamp} {level} {message}\n")


def set_run_logs_dir(path: Path) -> None:
    global _RUN_OVERRIDE
    _RUN_OVERRIDE = path.expanduser()


@contextmanager
def use_run_logs_dir(path: Path) -> Iterator[None]:
    global _RUN_OVERRIDE
    previous = _RUN_OVERRIDE
    _RUN_OVERRIDE = path.expanduser()
    try:
        yield
    finally:
        _RUN_OVERRIDE = previous


def _new_run_logs_dir(home: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return home / "logs" / f"{timestamp}-{os.getpid()}"
