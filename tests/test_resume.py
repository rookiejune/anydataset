import tempfile
import unittest
from pathlib import Path

from anydataset._resume import (
    cleanup_resume_dir,
    dataset_sample_count,
    index_batch_id,
    indexes_complete,
    pending_batch,
    prepare_resume_dir,
    resume_root,
    validate_completed_indexes,
)


class ResumeHelpersTest(unittest.TestCase):
    def test_prepare_resume_dir_uses_hidden_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "dataset"

            path = prepare_resume_dir(target, "fragments")

            self.assertEqual(path, Path(tmpdir) / ".dataset.resume" / "fragments")
            self.assertTrue(path.is_dir())

    def test_prepare_resume_dir_rejects_non_empty_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "dataset"
            target.mkdir()
            (target / "dataset.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Target directory must be empty"):
                prepare_resume_dir(target, "fragments")

    def test_cleanup_resume_dir_removes_hidden_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "dataset"
            root = resume_root(target)
            (root / "fragments").mkdir(parents=True)

            cleanup_resume_dir(target)

            self.assertFalse(root.exists())

    def test_dataset_sample_count_reports_context(self):
        with self.assertRaisesRegex(TypeError, "filter requires a dataset with __len__"):
            dataset_sample_count(iter(()), context="filter")

    def test_validate_completed_indexes_rejects_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "outside dataset"):
            validate_completed_indexes({0, 3}, 3)

    def test_indexes_complete_uses_validated_cardinality(self):
        self.assertTrue(indexes_complete(validate_completed_indexes({0, 2, 1}, 3), 3))
        self.assertFalse(indexes_complete(validate_completed_indexes({0, 2}, 3), 3))

    def test_pending_batch_skips_completed_indexes(self):
        self.assertEqual(
            pending_batch([(0, "a"), (1, "b"), (2, "c")], frozenset({1})),
            ((0, "a"), (2, "c")),
        )

    def test_index_batch_id_is_stable_path_segment(self):
        first = index_batch_id((2, 5, 9))
        second = index_batch_id((2, 5, 9))

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("batch-000000000002-000000000009-"))
        self.assertNotIn("/", first)


if __name__ == "__main__":
    unittest.main()
