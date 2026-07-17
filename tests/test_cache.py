import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from anydataset import Source, Spec, anydataset_home
from anydataset._logging import run_logs_dir, worker_logger, write_warning
from anydataset.cache import CacheManager, FileLock, FileLockError


class CacheManagerTest(unittest.TestCase):
    def test_file_lock_fails_fast_when_held_in_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "work.lock"

            with FileLock(path):
                with self.assertRaisesRegex(
                    FileLockError,
                    f"File lock is already held: {path}",
                ):
                    with FileLock(path):
                        self.fail("nested lock unexpectedly succeeded")

            with FileLock(path):
                pass

    def test_file_lock_reports_external_contention(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "work.lock"

            with mock.patch(
                "anydataset.cache.fcntl.flock",
                side_effect=BlockingIOError,
            ):
                with self.assertRaisesRegex(
                    FileLockError,
                    f"File lock is already held: {path}",
                ):
                    with FileLock(path):
                        self.fail("contended lock unexpectedly succeeded")

            with FileLock(path):
                pass

    def test_prepare_creates_stable_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                manager = CacheManager()
                spec = Spec(
                    source=Source.HF,
                    path="ylecun/mnist",
                    split="train",
                )

                first = manager.prepare(spec)
                second = manager.prepare(spec)

            self.assertEqual(first.cache_path, second.cache_path)
            self.assertEqual(
                first.cache_path,
                Path(tmpdir) / "cache" / "sources" / spec.cache_relpath,
            )
            self.assertTrue(first.metadata_path.exists())
            metadata = json.loads(Path(first.metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["source"], "hf")
            self.assertEqual(metadata["path"], "ylecun/mnist")
            self.assertEqual(metadata["split"], "train")
            self.assertEqual(metadata["id"], spec.id)

    def test_cache_path_uses_physical_spec_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                manager = CacheManager()
                first = Spec(
                    source=Source.HF,
                    path="google/fleurs",
                    split="train",
                    load_options={"config_name": "en_us", "streaming": True},
                )
                second = Spec(
                    source=Source.HF,
                    path="google/fleurs",
                    split="train",
                    load_options={"config_name": "en_us", "streaming": True},
                )

                self.assertEqual(manager.prepare(first).cache_path, manager.prepare(second).cache_path)

    def test_anydataset_home_uses_environment_at_call_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(
                os.environ,
                {"ANYDATASET_HOME": tmpdir},
            ):
                self.assertEqual(anydataset_home(), Path(tmpdir))

    def test_run_logs_dir_uses_anydataset_home(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                first = run_logs_dir()
                second = run_logs_dir()

            self.assertEqual(first, second)
            self.assertEqual(first.parent, home / "logs")
            self.assertRegex(first.name, r"^\d{8}-\d{6}-\d+$")

    def test_write_warning_writes_source_log_in_run_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                write_warning("source", "careful")

            logs = list((home / "logs").glob("*/source.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("WARNING careful", logs[0].read_text(encoding="utf-8"))

    def test_worker_logger_writes_rank_zero_to_file_and_stderr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                logger = worker_logger("filter", logs_dir, 0)
                logger.info("started")
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()

            text = (logs_dir / "part-00000.log").read_text(encoding="utf-8")

        self.assertIn("worker log:", text)
        self.assertIn("started", text)
        self.assertIn("worker log:", stderr.getvalue())
        self.assertIn("started", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
