import pickle
import tempfile
import unittest
from pathlib import Path

from anydataset._resume import (
    ComplementIndexes,
    append_completed_index_cache,
    cached_completed_indexes,
    cleanup_resume_dir,
    dataset_sample_count,
    format_index_ranges,
    index_batch_id,
    indexes_complete,
    missing_indexes,
    pending_batch,
    prepare_resume_dir,
    quarantine_resume_dir,
    resume_root,
    validate_completed_indexes,
    write_completed_index_cache,
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

    def test_quarantine_resume_dir_renames_hidden_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "dataset"
            root = resume_root(target)
            (root / "fragments").mkdir(parents=True)

            stale = quarantine_resume_dir(target)

            self.assertIsNotNone(stale)
            self.assertFalse(root.exists())
            self.assertTrue(stale.is_dir())
            self.assertTrue(stale.name.startswith(".dataset.resume.stale-"))

    def test_dataset_sample_count_reports_context(self):
        with self.assertRaisesRegex(TypeError, "filter requires a dataset with __len__"):
            dataset_sample_count(iter(()), context="filter")

    def test_validate_completed_indexes_rejects_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "outside dataset"):
            validate_completed_indexes({0, 3}, 3)

    def test_indexes_complete_uses_validated_cardinality(self):
        self.assertTrue(indexes_complete(validate_completed_indexes({0, 2, 1}, 3), 3))
        self.assertFalse(indexes_complete(validate_completed_indexes({0, 2}, 3), 3))

    def test_missing_indexes_keeps_fresh_runs_compact(self):
        missing = missing_indexes(frozenset(), 20_000_000)

        self.assertIsInstance(missing, range)
        self.assertEqual(len(missing), 20_000_000)
        self.assertEqual((missing[0], missing[-1]), (0, 19_999_999))

    def test_missing_indexes_uses_picklable_lazy_complement(self):
        missing = missing_indexes(frozenset({1, 4}), 8)
        restored = pickle.loads(pickle.dumps(missing))

        self.assertEqual(tuple(restored), (0, 2, 3, 5, 6, 7))
        self.assertEqual(restored[2], 3)
        self.assertEqual(restored[-1], 7)
        self.assertEqual(restored[1:5:2], (2, 5))

    def test_missing_indexes_materializes_the_smaller_side(self):
        missing = missing_indexes(frozenset({0, 1, 3, 4, 6, 7}), 8)

        self.assertEqual(missing, (2, 5))

    def test_missing_indexes_rejects_unvalidated_completed_indexes(self):
        with self.assertRaisesRegex(ValueError, "outside dataset"):
            missing_indexes(frozenset({-1}), 8)

    def test_missing_indexes_rejects_negative_expected_count(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            missing_indexes(frozenset(), -1)

    def test_completed_indexes_reject_boolean_values(self):
        with self.assertRaisesRegex(ValueError, "integers"):
            validate_completed_indexes((True,), 2)

    def test_complement_indexes_enforces_compact_sequence_invariants(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            ComplementIndexes(4, (2, 1))
        with self.assertRaisesRegex(ValueError, "outside expected range"):
            ComplementIndexes(4, (4,))

    def test_format_index_ranges_skips_large_contiguous_runs(self):
        self.assertEqual(
            format_index_ranges(range(20_000_000)),
            "0-19999999",
        )
        self.assertEqual(
            format_index_ranges(missing_indexes(frozenset({2, 5}), 20_000_000)),
            "0-1,3-4,6-19999999",
        )

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

    def test_completed_index_cache_round_trips_indexes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            write_completed_index_cache(
                root,
                (
                    ("batch-a", (0, 2)),
                    ("batch-b", (3,)),
                ),
            )

            self.assertEqual(
                cached_completed_indexes(root, ("batch-b", "batch-a")),
                frozenset({0, 2, 3}),
            )

    def test_completed_index_cache_ignores_fragment_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            write_completed_index_cache(root, (("batch-a", (0,)),))

            self.assertIsNone(cached_completed_indexes(root, ("batch-a", "batch-b")))

    def test_append_completed_index_cache_records_new_fragment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            append_completed_index_cache(root, "batch-a", (0, 1))
            append_completed_index_cache(root, "batch-b", (2,))

            self.assertEqual(
                cached_completed_indexes(root, ("batch-a", "batch-b")),
                frozenset({0, 1, 2}),
            )


if __name__ == "__main__":
    unittest.main()
