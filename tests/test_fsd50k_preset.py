from __future__ import annotations

import tempfile
import threading
import unittest
from unittest import mock

from anydataset.presets.fsd50k import FSD50K


class FSD50KPresetTest(unittest.TestCase):
    def test_revision_is_part_of_listing_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = FSD50K(revision="refs/convert/parquet")

            with (
                mock.patch.dict("os.environ", {"ANYDATASET_HOME": tmpdir}),
                mock.patch(
                    "anydataset.presets.fsd50k._list_files",
                    return_value=["clips/dev/example.wav"],
                ) as list_files,
            ):
                state = dataset.prepare()

        self.assertEqual(dataset.spec.load_options["revision"], "refs/convert/parquet")
        self.assertEqual(state["revision"], "refs/convert/parquet")
        list_files.assert_called_once_with(
            "Fhrozen/FSD50k",
            "dev",
            "refs/convert/parquet",
        )

    def test_rejects_unknown_load_options(self):
        with self.assertRaisesRegex(TypeError, "Unexpected FSD50K load option"):
            FSD50K(streaming=True)

    def test_rejects_empty_revision(self):
        with self.assertRaisesRegex(ValueError, "revision"):
            FSD50K(revision="")

    def test_rejects_invalid_split_at_construction(self):
        with self.assertRaisesRegex(ValueError, "split"):
            FSD50K(split="train")

    def test_rejects_invalid_cached_file_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = FSD50K()
            with mock.patch.dict("os.environ", {"ANYDATASET_HOME": tmpdir}):
                cache = dataset.cache_manager.prepare(dataset.spec)
                (cache.cache_path / "dev_files.json").write_text(
                    "[]\n",
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, "non-empty list"):
                    dataset.prepare()

    def test_rejects_hub_file_entry_without_string_path(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'[{"type": "file", "path": 1}]'
        response.headers = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = FSD50K()
            with (
                mock.patch.dict("os.environ", {"ANYDATASET_HOME": tmpdir}),
                mock.patch(
                    "anydataset.presets.fsd50k.urlopen",
                    return_value=response,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "must contain a path"):
                    dataset.prepare()

    def test_concurrent_prepare_lists_files_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = FSD50K()
            second = FSD50K()
            entered = threading.Event()
            release = threading.Event()
            repeated = threading.Event()
            calls = 0
            calls_lock = threading.Lock()
            errors = []

            def list_files(_repo_id, _split, _revision):
                nonlocal calls
                with calls_lock:
                    calls += 1
                    current = calls
                if current == 1:
                    entered.set()
                    release.wait(timeout=2)
                else:
                    repeated.set()
                return ["clips/dev/example.wav"]

            def prepare(dataset):
                try:
                    dataset.prepare()
                except Exception as exc:
                    errors.append(exc)

            with (
                mock.patch.dict("os.environ", {"ANYDATASET_HOME": tmpdir}),
                mock.patch(
                    "anydataset.presets.fsd50k._list_files",
                    side_effect=list_files,
                ),
            ):
                first_thread = threading.Thread(target=prepare, args=(first,))
                second_thread = threading.Thread(target=prepare, args=(second,))
                first_thread.start()
                self.assertTrue(entered.wait(timeout=1))
                second_thread.start()
                self.assertFalse(repeated.wait(timeout=0.1))
                release.set()
                first_thread.join(timeout=2)
                second_thread.join(timeout=2)

            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
