from __future__ import annotations

import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

from ..._runtime.logging import write_warning
from ..._runtime.resume import (
    cleanup_resume_dir,
    prepare_resume_dir,
    quarantine_resume_dir,
    resume_dir,
)
from ..jsonio import read_json, write_json


def prepare_materializer_resume_dir(
    output_dir: str | Path,
    metadata: Mapping[str, object],
    *,
    staging_dir: str | Path | None = None,
) -> Path:
    path = materializer_fragments_dir(output_dir, staging_dir=staging_dir)
    expected = dict(metadata)
    if staging_dir is not None:
        _validate_output_target(output_dir)
    if path.exists() and _stored_resume_metadata_or_incompatible(path) != expected:
        stale = _quarantine_materializer_dir(
            output_dir,
            staging_dir=staging_dir,
        )
        write_warning(
            "materializer",
            "Quarantined incompatible resume directory "
            f"at {stale}; remove it after confirming it is no longer needed.",
            event="materializer_resume_quarantined",
            fields={
                "output_dir": Path(output_dir).expanduser(),
                "resume_dir": path,
                "quarantine_dir": stale,
            },
        )
    if staging_dir is None:
        path = prepare_resume_dir(output_dir, "fragments")
    else:
        path.mkdir(parents=True, exist_ok=True)
    write_json(path / "resume.json", expected)
    return path


def materializer_fragments_dir(
    output_dir: str | Path,
    *,
    staging_dir: str | Path | None = None,
) -> Path:
    if staging_dir is None:
        return resume_dir(output_dir, "fragments")
    return Path(staging_dir).expanduser()


def cleanup_materializer_resume_dir(
    output_dir: str | Path,
    *,
    staging_dir: str | Path | None = None,
) -> None:
    if staging_dir is None:
        cleanup_resume_dir(output_dir)
        return
    path = Path(staging_dir).expanduser()
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError(f"Materializer staging path is not a directory: {path}")
    shutil.rmtree(path)


def stored_resume_metadata(path: Path) -> Mapping[str, object] | None:
    metadata_path = path / "resume.json"
    if not metadata_path.is_file():
        return None
    data = read_json(metadata_path)
    if not isinstance(data, Mapping):
        raise ValueError("Materializer resume metadata must be a mapping.")
    return data


def _stored_resume_metadata_or_incompatible(path: Path) -> Mapping[str, object] | None:
    try:
        return stored_resume_metadata(path)
    except (OSError, TypeError, ValueError):
        return None


def _quarantine_materializer_dir(
    output_dir: str | Path,
    *,
    staging_dir: str | Path | None,
) -> Path | None:
    if staging_dir is None:
        return quarantine_resume_dir(output_dir)
    path = Path(staging_dir).expanduser()
    if not path.exists():
        return None
    suffix = f"{time.time_ns()}-{os.getpid()}"
    stale = path.with_name(f".{path.name}.stale-{suffix}")
    path.replace(stale)
    return stale


def _validate_output_target(output_dir: str | Path) -> None:
    output = Path(output_dir).expanduser()
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        return
    if not output.is_dir():
        raise ValueError(f"Target path exists and is not a directory: {output}")
    if any(output.iterdir()):
        raise ValueError(f"Target directory must be empty: {output}")


def materializer_lock_path(output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser()
    return output.parent / f".{output.name}.materialize.lock"
