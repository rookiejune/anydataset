from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tomllib
import venv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    source_version = _source_version(root)
    if source_version != project["version"]:
        raise RuntimeError(
            f"version mismatch: anydataset={source_version!r}, "
            f"pyproject={project['version']!r}"
        )

    if not args.skip_tests:
        _run([sys.executable, "-m", "pytest", "-q"], root)

    if args.skip_build:
        return

    _require_module("build")
    _require_module("twine")

    _clean_build_outputs(root)
    _run([sys.executable, "-m", "build"], root)
    artifacts = sorted((root / "dist").glob("*"))
    if not artifacts:
        raise RuntimeError("build did not produce any dist artifacts.")
    _run([sys.executable, "-m", "twine", "check", *map(str, artifacts)], root)
    if not args.skip_smoke:
        _smoke_install(root, project["version"])


def _run(command: list[str], root: Path) -> None:
    subprocess.run(command, cwd=root, check=True)


def _require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"{name} is not installed; install the dev extra first.")


def _clean_build_outputs(root: Path) -> None:
    for path in (root / "build", root / "dist", root / "src" / "anydataset.egg-info"):
        if path.exists():
            shutil.rmtree(path)


def _smoke_install(root: Path, version: str) -> None:
    wheel = root / "dist" / f"anydataset-{version}-py3-none-any.whl"
    if not wheel.exists():
        raise RuntimeError(f"expected wheel not found: {wheel}")

    smoke_dir = root / "build" / "release-smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    venv.EnvBuilder(with_pip=True, clear=True).create(smoke_dir)
    python = _venv_python(smoke_dir)
    _run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)], root)
    _run(
        [
            str(python),
            "-c",
            (
                "from importlib.metadata import files, version; "
                f"assert version('anydataset') == {version!r}; "
                "installed = {str(path) for path in files('anydataset')}; "
                "assert 'anydataset/__init__.py' in installed; "
                "assert 'anydataset/_version.py' in installed"
            ),
        ],
        root,
    )


def _venv_python(path: Path) -> Path:
    if sys.platform == "win32":
        return path / "Scripts" / "python.exe"
    return path / "bin" / "python"


def _source_version(root: Path) -> str:
    namespace: dict[str, str] = {}
    exec((root / "src" / "anydataset" / "_version.py").read_text(), namespace)
    return namespace["__version__"]


if __name__ == "__main__":
    main()
