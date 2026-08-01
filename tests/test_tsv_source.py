from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest import mock

import anydataset.dataset.source._tabular_parquet as tabular_parquet

from anydataset import (
    AnyDataset,
    Spec,
    resolve_dataset,
)
from anydataset.dataset.source._registry import source_exists


class TsvSourceTest(unittest.TestCase):
    def test_registered_as_builtin_source(self):
        self.assertTrue(source_exists("tsv"))

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

            dataset = AnyDataset(
                Spec(source="tsv", path=tmpdir, split="train"),
                parse_fn=lambda row: row["sentence"],
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0], "hello")
            self.assertEqual(dataset[1], "tea")
            self.assertEqual(list(dataset), ["hello", "tea"])

    def test_reads_quoted_newlines_in_tsv_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train.tsv").write_text(
                'path\tsentence\n'
                'a.mp3\t"hello\nworld"\n',
                encoding="utf-8",
            )

            dataset = AnyDataset(
                Spec(source="tsv", path=tmpdir, split="train"),
                parse_fn=lambda row: row["sentence"],
            )

            self.assertEqual(list(dataset), ["hello\nworld"])

    def test_rejects_unknown_load_options(self):
        dataset = AnyDataset(
            Spec(
                source="tsv",
                path="unused",
                load_options={"unknown": True},
            )
        )

        with self.assertRaisesRegex(TypeError, "Unexpected TSV load option: unknown"):
            dataset.prepare()

    def test_rejects_non_string_encoding(self):
        dataset = AnyDataset(
            Spec(
                source="tsv",
                path="unused",
                load_options={"encoding": None},
            )
        )

        with self.assertRaisesRegex(TypeError, "TSV encoding must be a string"):
            dataset.prepare()

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

            dataset = AnyDataset(
                Spec(
                    source="tsv",
                    path=tmpdir,
                    split="train",
                    load_options={
                        "subdirs": ("en", "zh-CN"),
                        "root_field": "root",
                        "prepare_workers": 0,
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
            self.assertEqual(dataset[1], ("ni hao", str(zh)))

    def test_iter_shard_uses_native_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "train.tsv").write_text(
                "path\tsentence\n"
                "a.mp3\tzero\n"
                "b.mp3\tone\n"
                "c.mp3\ttwo\n",
                encoding="utf-8",
            )

            dataset = AnyDataset(
                Spec(source="tsv", path=tmpdir, split="train"),
                parse_fn=lambda row: row["sentence"],
            )
            prepared = dataset.dataset

            self.assertEqual(
                [(index, row["sentence"]) for index, row in prepared.iter_shard(2, 1)],
                [(1, "one")],
            )
            self.assertEqual(list(dataset.iter_shard(2, 1)), [(1, "one")])

    def test_reuses_prepared_parquet_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = root / "data"
            home = root / "home"
            data.mkdir()
            home.mkdir()
            (data / "train.tsv").write_text(
                "sentence\nhello\ntea\n",
                encoding="utf-8",
            )
            spec = Spec(source="tsv", path=str(data), split="train")
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                first = AnyDataset(spec, parse_fn=lambda row: row["sentence"])
                self.assertEqual(list(first), ["hello", "tea"])
                manifests = list(first.cache_manager.root.rglob("tsv_parquet.json"))
                self.assertEqual(len(manifests), 1)
                manifest = manifests[0]
                data = json.loads(manifest.read_text(encoding="utf-8"))
                for record in data["files"]:
                    record["device"] = -1
                    record["inode"] = -1
                manifest.write_text(json.dumps(data), encoding="utf-8")

                second = AnyDataset(spec, parse_fn=lambda row: row["sentence"])
                with mock.patch.object(
                    tabular_parquet,
                    "convert_file_job",
                ) as convert:
                    self.assertEqual(list(second), ["hello", "tea"])
                convert.assert_not_called()

    def test_root_field_does_not_change_physical_cache_identity(self):
        base = Spec(source="tsv", path="/data/common_voice", split="train")
        with_root = Spec(
            source="tsv",
            path="/data/common_voice",
            split="train",
            load_options={"root_field": "root"},
        )

        self.assertEqual(base.id, with_root.id)
        self.assertEqual(base.cache_relpath, with_root.cache_relpath)
        self.assertNotIn("root_field", with_root.to_dict()["load_options"])


if __name__ == "__main__":
    unittest.main()
