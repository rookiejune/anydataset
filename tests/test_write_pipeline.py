from __future__ import annotations

import unittest
from pathlib import Path
from threading import Event, Thread

from anydataset._runtime.write_pipeline import BackgroundWriteSink


class BackgroundWriteSinkTest(unittest.TestCase):
    def test_default_backend_writes_with_threads(self):
        calls = []

        with BackgroundWriteSink(
            calls.append,
            workers=1,
            start_method="spawn",
        ) as sink:
            sink.submit("a")
            sink.submit("b")

        self.assertEqual(calls, ["a", "b"])

    def test_inline_backend_runs_without_executor(self):
        calls = []

        with BackgroundWriteSink(
            calls.append,
            workers=0,
            start_method="spawn",
        ) as sink:
            sink.submit("a")

        self.assertEqual(calls, ["a"])

    def test_unknown_backend_is_rejected(self):
        sink = BackgroundWriteSink(
            Path("unused").write_text,
            workers=1,
            start_method="spawn",
            backend="bad",  # type: ignore[arg-type]
        )

        with self.assertRaises(ValueError):
            with sink:
                pass

    def test_background_failure_propagates_on_close(self):
        def write(value: str) -> None:
            if value == "bad":
                raise RuntimeError("bad write")

        with self.assertRaisesRegex(RuntimeError, "bad write"):
            with BackgroundWriteSink(
                write,
                workers=1,
                start_method="spawn",
            ) as sink:
                sink.submit("bad")

    def test_flush_waits_and_keeps_sink_reusable(self):
        calls = []

        sink = BackgroundWriteSink(
            calls.append,
            workers=1,
            start_method="spawn",
        )
        sink.__enter__()
        sink.submit("a")
        sink.flush()
        self.assertEqual(calls, ["a"])

        sink.submit("b")
        sink.flush()
        sink.close()

        self.assertEqual(calls, ["a", "b"])

    def test_background_failure_is_sticky(self):
        failure = RuntimeError("bad write")

        def write(_value: str) -> None:
            raise failure

        sink = BackgroundWriteSink(
            write,
            workers=1,
            start_method="spawn",
        )
        sink.__enter__()
        sink.submit("bad")

        with self.assertRaises(RuntimeError) as flushed:
            sink.flush()
        with self.assertRaises(RuntimeError) as submitted:
            sink.submit("later")
        with self.assertRaises(RuntimeError) as closed:
            sink.close()

        self.assertIs(flushed.exception, failure)
        self.assertIs(submitted.exception, failure)
        self.assertIs(closed.exception, failure)

    def test_completed_job_releases_capacity_before_slow_head(self):
        first_started = Event()
        second_done = Event()
        third_done = Event()
        release_first = Event()
        completed = []

        def write(value: str) -> None:
            if value == "first":
                first_started.set()
                release_first.wait()
            elif value == "second":
                second_done.set()
            else:
                third_done.set()

        with BackgroundWriteSink(
            write,
            workers=2,
            max_pending=2,
            start_method="spawn",
            on_complete=lambda job, _pending, _elapsed: completed.append(job),
        ) as sink:
            sink.submit("first")
            self.assertTrue(first_started.wait(timeout=5))
            sink.submit("second")
            self.assertTrue(second_done.wait(timeout=5))
            sink.submit("third")
            self.assertTrue(third_done.wait(timeout=5))
            release_first.set()

        self.assertEqual(completed[0], "second")
        self.assertCountEqual(completed, ["first", "second", "third"])

    def test_reports_submit_backpressure(self):
        first_started = Event()
        release_first = Event()
        second_submitted = Event()
        blocked = []

        def write(value: str) -> None:
            if value == "first":
                first_started.set()
                release_first.wait()

        with BackgroundWriteSink(
            write,
            workers=1,
            max_pending=1,
            start_method="spawn",
            on_backpressure=blocked.append,
        ) as sink:
            sink.submit("first")
            self.assertTrue(first_started.wait(timeout=5))

            def submit_second() -> None:
                sink.submit("second")
                second_submitted.set()

            submitter = Thread(target=submit_second)
            submitter.start()
            self.assertFalse(second_submitted.wait(timeout=0.05))
            release_first.set()
            self.assertTrue(second_submitted.wait(timeout=5))
            submitter.join(timeout=5)

        self.assertEqual(len(blocked), 1)
        self.assertGreater(blocked[0], 0.0)

    def test_abort_preserves_body_error(self):
        def write(value: str) -> None:
            raise RuntimeError(f"background failed: {value}")

        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with BackgroundWriteSink(
                write,
                workers=1,
                start_method="spawn",
            ) as sink:
                sink.submit("a")
                raise RuntimeError("body failed")


if __name__ == "__main__":
    unittest.main()
