"""Fail when anydataset's exported type completeness regresses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TypeCounts:
    known: int
    ambiguous: int
    unknown: int

    @property
    def total(self) -> int:
        return self.known + self.ambiguous + self.unknown

    @property
    def incomplete(self) -> int:
        return self.ambiguous + self.unknown


BASELINE = TypeCounts(known=1758, ambiguous=17, unknown=106)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        version, current = _verifytypes(root)
        _validate_ratchet(current, BASELINE)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "anydataset type completeness "
        f"({version}): {current.known}/{current.total} known "
        f"({current.known / current.total:.2%}); "
        f"{current.ambiguous} ambiguous, {current.unknown} unknown"
    )


def _verifytypes(root: Path) -> tuple[str, TypeCounts]:
    environment = dict(os.environ)
    source_path = str(root / "src")
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not current_pythonpath
        else source_path + os.pathsep + current_pythonpath
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "basedpyright",
            "--verifytypes",
            "anydataset",
            "--ignoreexternal",
            "--outputjson",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        raw_report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stderr.strip() or "no JSON report"
        raise RuntimeError(f"basedpyright verifytypes failed: {detail}") from exc
    report = _mapping(raw_report, "verifytypes report")

    summary = _mapping(report.get("summary"), "verifytypes summary")
    if _integer(summary.get("errorCount"), "verifytypes errorCount") != 0:
        raise RuntimeError("basedpyright verifytypes reported analysis errors.")
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"basedpyright verifytypes exited with {completed.returncode}."
        )

    completeness = _mapping(
        report.get("typeCompleteness"),
        "verifytypes typeCompleteness",
    )
    if completeness.get("packageName") != "anydataset":
        raise RuntimeError("basedpyright verifytypes did not resolve anydataset.")
    expected_root = (root / "src" / "anydataset").resolve()
    package_root = completeness.get("packageRootDirectory")
    if not isinstance(package_root, str) or Path(package_root).resolve() != expected_root:
        raise RuntimeError(
            "basedpyright verifytypes resolved an unexpected package root."
        )
    if completeness.get("ignoreUnknownTypesFromImports") is not True:
        raise RuntimeError("basedpyright verifytypes did not ignore external types.")

    counts = _mapping(
        completeness.get("exportedSymbolCounts"),
        "verifytypes exportedSymbolCounts",
    )
    current = TypeCounts(
        known=_integer(counts.get("withKnownType"), "known type count"),
        ambiguous=_integer(
            counts.get("withAmbiguousType"),
            "ambiguous type count",
        ),
        unknown=_integer(counts.get("withUnknownType"), "unknown type count"),
    )
    if current.total <= 0:
        raise RuntimeError("basedpyright verifytypes reported no exported symbols.")
    version = report.get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError("basedpyright verifytypes omitted its version.")
    return version, current


def _validate_ratchet(current: TypeCounts, baseline: TypeCounts) -> None:
    regressions: list[str] = []
    if current.incomplete > baseline.incomplete:
        regressions.append(
            "incomplete symbols rose from "
            f"{baseline.incomplete} to {current.incomplete}"
        )
    if current.known * baseline.total < baseline.known * current.total:
        regressions.append(
            "known-type ratio fell from "
            f"{baseline.known / baseline.total:.2%} to "
            f"{current.known / current.total:.2%}"
        )
    if regressions:
        raise RuntimeError("Type completeness regression: " + "; ".join(regressions))


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RuntimeError(f"{name} must be an object.")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer.")
    return value


if __name__ == "__main__":
    main()
