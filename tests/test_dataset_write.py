from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from anydataset.dataset.write import DatasetStoreWriter
import anydataset.dataset.write as dataset_write
from anydataset.store import DatasetWriter


def _dataset_factory() -> tuple[()]:
    return ()


class DatasetWriteTest(unittest.TestCase):
    def test_dataset_store_writer_is_a_lazy_compatibility_alias(self):
        self.assertIs(DatasetStoreWriter, DatasetWriter)
        self.assertIsInstance(DatasetStoreWriter("unused"), DatasetWriter)

    def test_dataset_writer_restores_pre_parallel_pickle_state(self):
        restored = DatasetWriter.__new__(DatasetWriter)

        restored.__setstate__(
            {
                "output_dir": "legacy-output",
                "dataset_id": "legacy",
                "split": "train",
                "views": None,
                "max_shard_samples": 100,
            }
        )

        self.assertEqual(restored.output_dir, Path("legacy-output"))
        self.assertEqual(restored.provenance, {})
        self.assertEqual(restored.num_shards, 1)
        self.assertEqual(restored.num_workers, 0)
        self.assertIsNone(restored.prefetch_factor)

    def test_run_parts_closes_parent_queue_after_success_and_worker_error(self):
        for error in (None, RuntimeError("worker monitoring failed")):
            with self.subTest(error=error):
                context = mock.Mock()
                progress = mock.Mock()
                worker = mock.Mock(exitcode=0)
                worker.is_alive.return_value = True
                context.Queue.return_value = progress
                context.Process.return_value = worker

                with (
                    mock.patch.object(
                        dataset_write,
                        "multiprocessing_context",
                        return_value=context,
                    ),
                    mock.patch.object(dataset_write, "free_port", return_value="1234"),
                    mock.patch.object(
                        dataset_write,
                        "watch_workers",
                        side_effect=error,
                    ),
                ):
                    if error is None:
                        self._run_parts()
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "worker monitoring failed",
                        ):
                            self._run_parts()

                progress.close.assert_called_once_with()
                progress.join_thread.assert_called_once_with()
                self.assertLess(
                    progress.method_calls.index(mock.call.close()),
                    progress.method_calls.index(mock.call.join_thread()),
                )

    def test_run_parts_closes_parent_queue_when_process_construction_fails(self):
        context = mock.Mock()
        progress = mock.Mock()
        context.Queue.return_value = progress
        context.Process.side_effect = RuntimeError("process construction failed")

        with (
            mock.patch.object(
                dataset_write,
                "multiprocessing_context",
                return_value=context,
            ),
            mock.patch.object(dataset_write, "free_port", return_value="1234"),
        ):
            with self.assertRaisesRegex(RuntimeError, "process construction failed"):
                self._run_parts()

        progress.close.assert_called_once_with()
        progress.join_thread.assert_called_once_with()

    def test_queue_cleanup_does_not_replace_worker_error(self):
        context = mock.Mock()
        progress = mock.Mock()
        worker = mock.Mock(exitcode=0)
        worker.is_alive.return_value = True
        context.Queue.return_value = progress
        context.Process.return_value = worker
        progress.close.side_effect = RuntimeError("queue close failed")
        progress.join_thread.side_effect = RuntimeError("queue join failed")

        with (
            mock.patch.object(
                dataset_write,
                "multiprocessing_context",
                return_value=context,
            ),
            mock.patch.object(dataset_write, "free_port", return_value="1234"),
            mock.patch.object(
                dataset_write,
                "watch_workers",
                side_effect=RuntimeError("worker monitoring failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "worker monitoring failed"):
                self._run_parts()

        progress.close.assert_called_once_with()
        progress.join_thread.assert_called_once_with()

    def test_queue_cleanup_error_is_visible_after_success(self):
        context = mock.Mock()
        progress = mock.Mock()
        worker = mock.Mock(exitcode=0)
        worker.is_alive.return_value = True
        context.Queue.return_value = progress
        context.Process.return_value = worker
        progress.close.side_effect = RuntimeError("queue close failed")

        with (
            mock.patch.object(
                dataset_write,
                "multiprocessing_context",
                return_value=context,
            ),
            mock.patch.object(dataset_write, "free_port", return_value="1234"),
            mock.patch.object(dataset_write, "watch_workers"),
        ):
            with self.assertRaisesRegex(RuntimeError, "queue close failed"):
                self._run_parts()

        progress.join_thread.assert_called_once_with()

    def _run_parts(self) -> None:
        dataset_write._run_parts(
            Path("unused"),
            dataset_id="dataset",
            split=None,
            views=None,
            max_shard_samples=1,
            num_shards=1,
            num_workers=0,
            prefetch_factor=None,
            dataset_factory=_dataset_factory,
        )


if __name__ == "__main__":
    unittest.main()
