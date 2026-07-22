from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .._logging import write_warning
from .._resume import prepare_resume_dir, quarantine_resume_dir, resume_dir
from .jsonio import read_json, write_json


def prepare_materializer_resume_dir(
    output_dir: str | Path,
    metadata: Mapping[str, object],
) -> Path:
    path = resume_dir(output_dir, "fragments")
    expected = dict(metadata)
    if path.exists() and stored_resume_metadata(path) != expected:
        stale = quarantine_resume_dir(output_dir)
        write_warning(
            "materializer",
            "Quarantined incompatible resume directory "
            f"at {stale}; remove it after confirming it is no longer needed.",
        )
    path = prepare_resume_dir(output_dir, "fragments")
    write_json(path / "resume.json", expected)
    return path


def stored_resume_metadata(path: Path) -> Mapping[str, object] | None:
    metadata_path = path / "resume.json"
    if not metadata_path.is_file():
        return None
    data = read_json(metadata_path)
    if not isinstance(data, Mapping):
        raise ValueError("Materializer resume metadata must be a mapping.")
    return data


def materializer_lock_path(output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser()
    return output.parent / f".{output.name}.materialize.lock"
