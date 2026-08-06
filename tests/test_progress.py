from __future__ import annotations

import io
from contextlib import redirect_stdout
from unittest.mock import patch

from anydataset._runtime.progress import Progress, ProgressDashboard


def test_non_interactive_progress_reports_start_updates_and_finish() -> None:
    stdout = io.StringIO()
    with (
        patch("anydataset._runtime.progress._NON_INTERACTIVE_PROGRESS_INTERVAL", 0.0),
        redirect_stdout(stdout),
        ProgressDashboard(
            desc="materialize views",
            total=10,
            count_stage="writer",
            stages=("reader", "provider", "writer"),
        ) as progress,
    ):
        progress.put(Progress(0, 4, False, None, stage="provider"))
        provider_output = stdout.getvalue()
        assert "materialize views: 0 sample/10 (0.0%)" in provider_output
        assert "provider=4" in provider_output
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


def test_resume_progress_separates_coverage_from_current_run_rate() -> None:
    stdout = io.StringIO()
    now = 0.0
    with (
        patch(
            "anydataset._runtime.progress.time.monotonic",
            side_effect=lambda: now,
        ),
        patch("anydataset._runtime.progress._NON_INTERACTIVE_PROGRESS_INTERVAL", 0.0),
        redirect_stdout(stdout),
        ProgressDashboard(
            desc="resume demo",
            total=100,
            count_stage="writer",
            initial=80,
            stages=("writer",),
        ) as progress,
    ):
        now = 2.0
        progress.put(
            Progress(
                0,
                10,
                False,
                None,
                stage="writer",
                elapsed=2.0,
            )
        )

    output = stdout.getvalue()
    assert "80 sample/100 (80.0%)" in output
    assert "resumed=80 | run=0/20" in output
    assert "90 sample/100 (90.0%) [2s, 5.0 sample/s, ETA 2s]" in output
    assert "resumed=80 | run=10/20" in output
    assert "writer=10 5.0/s last=2.00s" in output
