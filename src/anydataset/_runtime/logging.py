"""Internal run-scoped logging paths.

The module owns process-local log run directory discovery under
`ANYDATASET_HOME`. It does not configure application-wide Python logging or
store sample-level audit metrics.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .._version import __version__
from ..cache import anydataset_home

_RUN_DIR: Path | None = None
_RUN_HOME: Path | None = None
_RUN_OVERRIDE: Path | None = None
_RUN_DIR_LOCK = threading.Lock()


def run_logs_dir() -> Path:
    global _RUN_DIR, _RUN_HOME
    with _RUN_DIR_LOCK:
        if _RUN_OVERRIDE is not None:
            _RUN_OVERRIDE.mkdir(parents=True, exist_ok=True)
            _ensure_run_metadata(_RUN_OVERRIDE)
            return _RUN_OVERRIDE
        home = anydataset_home()
        if _RUN_DIR is None or _RUN_HOME != home:
            _RUN_HOME = home
            _RUN_DIR = _new_run_logs_dir(home)
        _RUN_DIR.mkdir(parents=True, exist_ok=True)
        _ensure_run_metadata(_RUN_DIR)
        return _RUN_DIR


def write_info(
    source: str,
    message: str,
    *,
    event: str | None = None,
    fields: Mapping[str, object] | None = None,
) -> None:
    _write_log(source, "INFO", message)
    _write_message_event(source, "INFO", message, event=event, fields=fields)


def write_warning(
    source: str,
    message: str,
    *,
    event: str | None = None,
    fields: Mapping[str, object] | None = None,
) -> None:
    _write_log(source, "WARNING", message)
    _write_message_event(source, "WARNING", message, event=event, fields=fields)


def write_event(
    source: str,
    event: str,
    fields: Mapping[str, object] | None = None,
    *,
    level: str = "INFO",
) -> None:
    path = run_logs_dir()
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "source": source,
        "event": event,
        "run_id": path.name,
        "pid": os.getpid(),
        "fields": _json_value(fields or {}),
    }
    _append_json_line(path / "events.jsonl", entry)


def worker_logger(source: str, logs_dir: Path, worker_id: int) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"anydataset.{source}.{os.getpid()}.{worker_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for existing in tuple(logger.handlers):
        logger.removeHandler(existing)
        existing.close()
    path = logs_dir / f"part-{worker_id:05d}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(processName)s %(message)s")
    )
    logger.addHandler(handler)
    if worker_id == 0:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        stdout = logging.StreamHandler(sys.stdout)
        stdout.addFilter(_BelowErrorFilter())
        stdout.setFormatter(formatter)
        stderr = logging.StreamHandler(sys.stderr)
        stderr.setLevel(logging.ERROR)
        stderr.setFormatter(formatter)
        logger.addHandler(stdout)
        logger.addHandler(stderr)
    logger.info("worker log: %s", path)
    return logger


class _BelowErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.ERROR


def _write_log(source: str, level: str, message: str) -> None:
    path = run_logs_dir() / f"{source}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{timestamp} {level} {message}\n")


def _write_message_event(
    source: str,
    level: str,
    message: str,
    *,
    event: str | None,
    fields: Mapping[str, object] | None,
) -> None:
    event_fields: dict[str, object] = {"message": message}
    if fields is not None:
        event_fields.update(fields)
    write_event(source, event or "message", event_fields, level=level)


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


def _ensure_run_metadata(path: Path) -> None:
    metadata_path = path / "run.json"
    if metadata_path.exists():
        return
    metadata = {
        "run_id": path.name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "anydataset_home": str(anydataset_home()),
        "version": __version__,
    }
    try:
        with metadata_path.open("x", encoding="utf-8") as file:
            json.dump(_json_value(metadata), file, sort_keys=True)
            file.write("\n")
    except FileExistsError:
        return


def _append_json_line(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_value(value), sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as file:
        file.write(line)


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    return repr(value)
