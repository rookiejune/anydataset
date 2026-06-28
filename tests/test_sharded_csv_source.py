from pathlib import Path
import tempfile
import unittest
from unittest import mock

from anydataset import AnyDataset, IterableAnyDataset, Spec, has_source, resolve_dataset


class ShardedCsvSourceTest(unittest.TestCase):
    def test_registered_as_builtin_source(self):
        self.assertTrue(has_source("sharded_csv"))

    def test_resolves_registered_source_shorthand(self):
        spec = resolve_dataset("sharded_csv:///tmp/data:train")

        self.assertEqual((spec.source, spec.path, spec.split), (
            "sharded_csv",
            "/tmp/data",
            "train",
        ))

    def test_reads_physical_shard_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_0 = root / "shard_0"
            shard_1 = root / "shard_1"
            shard_0.mkdir()
            shard_1.mkdir()
            (shard_0 / "0.csv").write_text(
                "src_lang,src_text,target_lang,target_text\n"
                "en,hello,zh,nihao\n",
                encoding="utf-8",
            )
            (shard_1 / "0.csv").write_text(
                "src_lang,src_text,target_lang,target_text\n"
                "en,tea,zh,cha\n",
                encoding="utf-8",
            )

            dataset = IterableAnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
                cache_root=root / "cache",
            )

            self.assertEqual(list(dataset.iter_shard(2, 1)), ["tea"])

    def test_reads_multiple_csv_files_per_physical_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "10.csv").write_text(
                "src_text\n"
                "ten\n",
                encoding="utf-8",
            )
            (shard_dir / "2.csv").write_text(
                "src_text\n"
                "two\n",
                encoding="utf-8",
            )

            dataset = IterableAnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
                cache_root=root / "cache",
            )

            self.assertEqual(list(dataset), ["two", "ten"])

    def test_supports_map_style_indexing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_0 = root / "shard_0"
            shard_1 = root / "shard_1"
            shard_0.mkdir()
            shard_1.mkdir()
            (shard_0 / "0.csv").write_text(
                "src_text\n"
                "zero\n"
                "one\n",
                encoding="utf-8",
            )
            (shard_1 / "0.csv").write_text(
                "src_text\n"
                "two\n",
                encoding="utf-8",
            )

            dataset = AnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
                cache_root=root / "cache",
            )

            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[0], "zero")
            self.assertEqual(dataset[2], "two")
            self.assertEqual(dataset[-1], "two")
            self.assertEqual(list(dataset.iter_shard(2, 1)), ["one"])

    def test_reuses_cached_row_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache = root / "cache"
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text(
                "src_text\n"
                "zero\n"
                "one\n",
                encoding="utf-8",
            )

            dataset = AnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
                cache_root=cache,
            )
            self.assertEqual(len(dataset), 2)
            index_files = list(cache.rglob("sharded_csv_index.json"))
            self.assertEqual(len(index_files), 1)

            second = AnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
                cache_root=cache,
            )
            with mock.patch.object(second.dataset, "_row_count") as row_count:
                self.assertEqual(len(second), 2)

            row_count.assert_not_called()

    def test_map_style_indexed_iteration_keeps_global_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text(
                "src_text\n"
                "zero\n"
                "one\n"
                "two\n",
                encoding="utf-8",
            )

            dataset = AnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
                cache_root=root / "cache",
            )

            self.assertEqual(list(dataset.iter_indexed_range(1, 3)), [(1, "one"), (2, "two")])
            self.assertEqual(list(dataset.iter_indexed_shard(2, 1)), [(1, "one")])

    def test_reads_split_physical_shard_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "train" / "shard_0"
            shard_dir.mkdir(parents=True)
            (shard_dir / "0.csv").write_text(
                "src_lang,src_text,target_lang,target_text\n"
                "en,hello,zh,nihao\n",
                encoding="utf-8",
            )

            dataset = IterableAnyDataset(
                Spec(source="sharded_csv", path=tmpdir, split="train"),
                parse_fn=lambda row: row["target_text"],
                cache_root=root / "cache",
            )

            self.assertEqual(list(dataset), ["nihao"])

    def test_warns_when_physical_shards_are_not_contiguous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            shard_0 = root / "shard_0"
            shard_2 = root / "shard_2"
            shard_0.mkdir()
            shard_2.mkdir()
            (shard_0 / "0.csv").write_text("src_text\nzero\n", encoding="utf-8")
            (shard_2 / "0.csv").write_text("src_text\ntwo\n", encoding="utf-8")

            dataset = IterableAnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
                cache_root=root / "cache",
            )

            with mock.patch(
                "anydataset.dataset.source.sharded_csv.Path.home",
                return_value=home,
            ):
                self.assertEqual(list(dataset), ["zero", "two"])

            log = home / ".anydataset" / "logs" / "sharded_csv.log"
            self.assertIn(
                f"Missing sharded CSV directories under {root}: shard_1.",
                log.read_text(encoding="utf-8"),
            )

    def test_reuses_physical_shard_scan_between_iterations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            shard_0 = root / "shard_0"
            shard_2 = root / "shard_2"
            shard_0.mkdir()
            shard_2.mkdir()
            (shard_0 / "0.csv").write_text("src_text\nzero\n", encoding="utf-8")
            (shard_2 / "0.csv").write_text("src_text\ntwo\n", encoding="utf-8")

            dataset = IterableAnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
                cache_root=root / "cache",
            )

            with mock.patch(
                "anydataset.dataset.source.sharded_csv.Path.home",
                return_value=home,
            ):
                self.assertEqual(list(dataset), ["zero", "two"])
                self.assertEqual(list(dataset), ["zero", "two"])

            log = home / ".anydataset" / "logs" / "sharded_csv.log"
            self.assertEqual(log.read_text(encoding="utf-8").count("WARNING"), 1)


if __name__ == "__main__":
    unittest.main()
