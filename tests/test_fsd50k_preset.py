from __future__ import annotations

import tempfile
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


if __name__ == "__main__":
    unittest.main()
