from __future__ import annotations

import io
from contextlib import redirect_stdout
from unittest.mock import patch

from anydataset._progress import Progress, ProgressDashboard


def test_non_interactive_progress_reports_start_updates_and_finish() -> None:
    stdout = io.StringIO()
    with (
        patch("anydataset._progress._NON_INTERACTIVE_PROGRESS_INTERVAL", 0.0),
        redirect_stdout(stdout),
        ProgressDashboard(
            desc="materialize views",
            total=10,
            count_stage="writer",
            stages=("reader", "provider", "writer"),
        ) as progress,
    ):
        progress.put(Progress(0, 4, False, None, stage="provider"))
        progress.put(Progress(0, 4, False, None, stage="writer"))

    output = stdout.getvalue()
    assert "materialize views: 0 sample/10 (0.0%)" in output
    assert "materialize views: 4 sample/10 (40.0%)" in output
    assert "provider=4" in output
    assert "writer=4" in output


def test_non_interactive_progress_prints_only_the_primary_bar() -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout), ProgressDashboard(
        desc="scan",
        total=2,
        stages=("reader",),
    ):
        pass

    lines = stdout.getvalue().splitlines()
    assert len(lines) == 2
    assert all(line.startswith("scan:") for line in lines)


def test_progress_can_render_non_interactive_logs_to_stdout() -> None:
    stdout = io.StringIO()
    with redirect_stdout(stdout), ProgressDashboard(
        desc="qwen tts",
        total=2,
    ) as progress:
        progress.put(Progress(0, 1, False, None))

    output = stdout.getvalue()
    assert "qwen tts: 1 sample/2 (50.0%)" in output
