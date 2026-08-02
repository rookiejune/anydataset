from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from anydataset._io.atomic import replace_dir


class AtomicDirectoryTest(unittest.TestCase):
    def test_replace_dir_keeps_empty_target_until_atomic_replace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "target"
            target.mkdir()
            real_replace = os.replace

            def checked_replace(src: str | Path, dst: str | Path) -> None:
                self.assertTrue(Path(dst).exists())
                real_replace(src, dst)

            def write(path: Path) -> None:
                (path / "value.txt").write_text("new", encoding="utf-8")

            with mock.patch(
                "anydataset._io.atomic.os.replace",
                side_effect=checked_replace,
            ) as replace:
                replaced = replace_dir(target, write)

            self.assertEqual(replaced, target)
            self.assertEqual(
                (target / "value.txt").read_text(encoding="utf-8"),
                "new",
            )
            self.assertEqual(replace.call_count, 1)

    def test_replace_dir_preserves_empty_target_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "target"
            target.mkdir()

            def write(path: Path) -> None:
                (path / "value.txt").write_text("new", encoding="utf-8")

            with mock.patch(
                "anydataset._io.atomic.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    replace_dir(target, write)

            self.assertTrue(target.is_dir())
            self.assertEqual(tuple(target.iterdir()), ())
            self.assertEqual({path.name for path in root.iterdir()}, {"target"})


if __name__ == "__main__":
    unittest.main()
