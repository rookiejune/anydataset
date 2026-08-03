from __future__ import annotations

import pickle
import unittest
from pathlib import Path
from unittest import mock

import anydataset.store.part.dispatch as dataset_write
from anydataset.store import DatasetWriter


def _dataset_factory() -> tuple[()]:
    return ()


class DatasetWriteTest(unittest.TestCase):
    def test_dataset_writer_pickle_state_is_explicit_and_versioned(self):
        writer = DatasetWriter("output", dataset_id="dataset", split="train")

        state = writer.__getstate__()

        self.assertEqual(state["pickle_schema_version"], 1)
        self.assertEqual(
            set(state),
            {
                "pickle_schema_version",
                "output_dir",
                "dataset_id",
                "split",
                "views",
                "max_shard_samples",
                "provenance",
                "num_shards",
                "num_workers",
                "prefetch_factor",
            },
        )
        self.assertEqual(pickle.loads(pickle.dumps(writer)), writer)

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

    def test_dataset_writer_rejects_unknown_pickle_schema(self):
        state = DatasetWriter("output").__getstate__()
        restored = DatasetWriter.__new__(DatasetWriter)

        for version in (0, 2):
            with self.subTest(version=version):
                state["pickle_schema_version"] = version
                with self.assertRaisesRegex(
                    ValueError,
                    f"Unsupported DatasetWriter pickle_schema_version {version}",
                ):
                    restored.__setstate__(state)

        for version in (True, "1"):
            with self.subTest(version=version):
                state["pickle_schema_version"] = version
                with self.assertRaisesRegex(
                    TypeError,
                    "pickle_schema_version must be an integer",
                ):
                    restored.__setstate__(state)

    def test_dataset_writer_rejects_invalid_pickle_fields(self):
        state = DatasetWriter("output").__getstate__()
        state["output_dir"] = object()
        restored = DatasetWriter.__new__(DatasetWriter)

        with self.assertRaisesRegex(
            TypeError,
            "field 'output_dir' must be a string or Path",
        ):
            restored.__setstate__(state)

        state = DatasetWriter("output").__getstate__()
        state["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unsupported field 'unexpected'"):
            restored.__setstate__(state)

        state = DatasetWriter("output").__getstate__()
        state.pop("views")
        with self.assertRaisesRegex(ValueError, "missing required field 'views'"):
            restored.__setstate__(state)

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
            provenance={},
            dataset_factory=_dataset_factory,
        )


if __name__ == "__main__":
    unittest.main()
