from __future__ import annotations

import io
from contextlib import redirect_stderr
from unittest.mock import patch

from anydataset._progress import Progress, ProgressDashboard


def test_non_interactive_progress_reports_start_updates_and_finish() -> None:
    stderr = io.StringIO()
    with (
        patch("anydataset._progress._NON_INTERACTIVE_PROGRESS_INTERVAL", 0.0),
        redirect_stderr(stderr),
        ProgressDashboard(
            desc="materialize views",
            total=10,
            count_stage="writer",
            stages=("reader", "provider", "writer"),
        ) as progress,
    ):
        progress.put(Progress(0, 4, False, None, stage="provider"))
        progress.put(Progress(0, 4, False, None, stage="writer"))

    output = stderr.getvalue()
    assert "materialize views: 0 sample/10 (0.0%)" in output
    assert "materialize views: 4 sample/10 (40.0%)" in output
    assert "provider=4" in output
    assert "writer=4" in output


def test_non_interactive_progress_prints_only_the_primary_bar() -> None:
    stderr = io.StringIO()
    with redirect_stderr(stderr), ProgressDashboard(
        desc="scan",
        total=2,
        stages=("reader",),
    ):
        pass

    lines = stderr.getvalue().splitlines()
    assert len(lines) == 2
    assert all(line.startswith("scan:") for line in lines)
