from __future__ import annotations

import unittest
from pathlib import Path

from anydataset._write_pipeline import BackgroundWriteSink


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
