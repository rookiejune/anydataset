from pathlib import Path
import tempfile
import unittest

from anydataset import (
    IterableAnyDataset,
    Spec,
    has_source,
    resolve_dataset,
)


class TsvSourceTest(unittest.TestCase):
    def test_registered_as_builtin_source(self):
        self.assertTrue(has_source("tsv"))

    def test_resolves_registered_source_shorthand(self):
        spec = resolve_dataset("tsv:///tmp/data:train")

        self.assertEqual((spec.source, spec.path, spec.split), (
            "tsv",
            "/tmp/data",
            "train",
        ))

    def test_reads_split_tsv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train.tsv").write_text(
                "path\tsentence\n"
                "a.mp3\thello\n"
                "b.mp3\ttea\n",
                encoding="utf-8",
            )

            dataset = IterableAnyDataset(
                Spec(source="tsv", path=tmpdir, split="train"),
                parse_fn=lambda row: row["sentence"],
            )

            self.assertEqual(list(dataset), ["hello", "tea"])

    def test_reads_split_tsv_subdirs_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            en = root / "en"
            zh = root / "zh-CN"
            en.mkdir()
            zh.mkdir()
            en.joinpath("train.tsv").write_text(
                "path\tsentence\n"
                "en.mp3\thello\n",
                encoding="utf-8",
            )
            zh.joinpath("train.tsv").write_text(
                "path\tsentence\n"
                "zh.mp3\tni hao\n",
                encoding="utf-8",
            )

            dataset = IterableAnyDataset(
                Spec(
                    source="tsv",
                    path=tmpdir,
                    split="train",
                    load_options={
                        "subdirs": ("en", "zh-CN"),
                        "root_field": "root",
                    },
                ),
                parse_fn=lambda row: (row["sentence"], row["root"]),
            )

            self.assertEqual(
                list(dataset),
                [
                    ("hello", str(en)),
                    ("ni hao", str(zh)),
                ],
            )

    def test_shards_rows_by_index_modulo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train.tsv").write_text(
                "path\tsentence\n"
                "a.mp3\tzero\n"
                "b.mp3\tone\n"
                "c.mp3\ttwo\n",
                encoding="utf-8",
            )

            dataset = IterableAnyDataset(
                Spec(source="tsv", path=tmpdir, split="train"),
                parse_fn=lambda row: row["sentence"],
            )

            self.assertEqual(list(dataset.iter_shard(2, 1)), ["one"])


if __name__ == "__main__":
    unittest.main()
