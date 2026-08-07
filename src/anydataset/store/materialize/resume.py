from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path

from ..._runtime.logging import write_warning
from ..._runtime.resume import (
    cleanup_resume_dir,
    cleanup_resume_path,
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
    allow_input_growth: bool = False,
    allow_output_contents: bool = False,
) -> Path:
    path = materializer_fragments_dir(output_dir, staging_dir=staging_dir)
    expected = dict(metadata)
    if staging_dir is not None and not allow_output_contents:
        _validate_output_target(output_dir)
    stored = _stored_resume_metadata_or_incompatible(path) if path.exists() else None
    compatible_growth = (
        allow_input_growth
        and stored is not None
        and _growing_input_metadata(stored, expected)
    )
    if path.exists() and stored != expected and not compatible_growth:
        if allow_input_growth:
            raise ValueError(
                "Materializer identity does not match the existing growing-input "
                "staging state."
            )
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
    if staging_dir is None and not allow_output_contents:
        path = prepare_resume_dir(output_dir, "fragments")
    else:
        path.mkdir(parents=True, exist_ok=True)
    write_json(path / "resume.json", expected)
    return path


def _growing_input_metadata(
    stored: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    stored_copy = dict(stored)
    expected_copy = dict(expected)
    stored_input = stored_copy.get("input")
    expected_input = expected_copy.get("input")
    if not isinstance(stored_input, Mapping) or not isinstance(expected_input, Mapping):
        return False
    stored_input_copy = dict(stored_input)
    expected_input_copy = dict(expected_input)
    stored_count = stored_input_copy.pop("sample_count", None)
    expected_count = expected_input_copy.pop("sample_count", None)
    stored_copy["input"] = stored_input_copy
    expected_copy["input"] = expected_input_copy
    return (
        type(stored_count) is int
        and type(expected_count) is int
        and stored_count <= expected_count
        and stored_copy == expected_copy
    )


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
    cleanup_resume_path(path)


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
