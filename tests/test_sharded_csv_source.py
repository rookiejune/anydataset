from pathlib import Path
import tempfile
import unittest

from anydataset import IterableAnyDataset, Spec, has_source, resolve_dataset


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

    def test_reads_global_shard_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shard_2"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text(
                "src_lang,src_text,target_lang,target_text\n"
                "en,hello,zh,nihao\n",
                encoding="utf-8",
            )
            (shard_dir / "1.csv").write_text(
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

    def test_reads_split_global_shard_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "train" / "shard_1"
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


if __name__ == "__main__":
    unittest.main()
