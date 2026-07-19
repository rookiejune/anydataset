import os
from pathlib import Path
import tempfile
import threading
import unittest
from functools import partial
from unittest import mock

from anydataset import (
    AnyDataset,
    IterableAnyDataset,
    Spec,
    has_source,
    resolve_dataset,
)
from anydataset._parallel import can_select_indexes, map_style_indexed_loader
from anydataset.cache import FileLock
from anydataset.dataset.source.sharded_csv import CsvShard, _missing_shard_ranges


class ShardedCsvSourceTest(unittest.TestCase):
    def test_registered_as_builtin_source(self):
        self.assertTrue(has_source("sharded_csv"))

    def test_rejects_unknown_load_options(self):
        dataset = AnyDataset(
            Spec(
                source="sharded_csv",
                path="unused",
                load_options={"unknown": True},
            )
        )

        with self.assertRaisesRegex(
            TypeError,
            "Unexpected sharded_csv load option: unknown",
        ):
            dataset.prepare()

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
            )

            self.assertEqual(list(dataset), ["two", "ten"])

    def test_rejects_equivalent_numeric_csv_file_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard = Path(tmpdir) / "shard_0"
            shard.mkdir()
            (shard / "1.csv").write_text("value\none\n", encoding="utf-8")
            (shard / "01.csv").write_text("value\ntwo\n", encoding="utf-8")
            dataset = AnyDataset(Spec(source="sharded_csv", path=tmpdir))

            with self.assertRaisesRegex(ValueError, "file indexes must be unique"):
                dataset.prepare()

    def test_rejects_equivalent_numeric_shard_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "shard_1").mkdir()
            (root / "shard_01").mkdir()
            dataset = AnyDataset(Spec(source="sharded_csv", path=tmpdir))

            with self.assertRaisesRegex(ValueError, "directory indexes must be unique"):
                dataset.prepare()

    def test_large_missing_shard_gap_is_represented_compactly(self):
        shards = (
            CsvShard(0, Path("shard_0")),
            CsvShard(1_000_000_000, Path("shard_1000000000")),
        )

        self.assertEqual(
            _missing_shard_ranges(shards),
            ((1, 999_999_999),),
        )

    def test_ignores_non_numeric_csv_file_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text(
                "src_text\n"
                "zero\n",
                encoding="utf-8",
            )
            (shard_dir / "metadata.csv").write_text(
                "src_text\n"
                "ignored\n",
                encoding="utf-8",
            )

            dataset = AnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
            )

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                self.assertEqual(len(dataset), 1)
                self.assertEqual(list(dataset), ["zero"])

            log = _single_log(home, "sharded_csv.log")
            self.assertIn(
                f"Ignored non-numeric CSV files under {shard_dir}: metadata.csv.",
                log.read_text(encoding="utf-8"),
            )

    def test_warns_once_for_ignored_non_numeric_csv_file_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text("src_text\nzero\n", encoding="utf-8")
            (shard_dir / "notes.csv").write_text("src_text\nignored\n", encoding="utf-8")

            dataset = IterableAnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
            )

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                self.assertEqual(list(dataset), ["zero"])
                self.assertEqual(list(dataset), ["zero"])

            log = _single_log(home, "sharded_csv.log")
            self.assertEqual(log.read_text(encoding="utf-8").count("WARNING"), 1)

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
            )

            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[0], "zero")
            self.assertEqual(dataset[2], "two")
            self.assertEqual(dataset[-1], "two")
            self.assertEqual(list(dataset.iter_shard(2, 1)), ["one"])

    def test_reuses_prepared_parquet_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
            )
            self.assertEqual(len(dataset), 2)
            manifests = list(
                dataset.cache_manager.root.rglob("sharded_csv_parquet.json")
            )
            parts = list(dataset.cache_manager.root.rglob("*.parquet"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(len(parts), 1)

            second = AnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
            )
            with mock.patch(
                "anydataset.dataset.source.sharded_csv._convert_file_job"
            ) as convert:
                self.assertEqual(len(second), 2)

            convert.assert_not_called()

    def test_rebuilds_changed_parquet_part(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            source = shard_dir / "0.csv"
            source.write_text("src_text\nzero\n", encoding="utf-8")
            spec = Spec(source="sharded_csv", path=tmpdir)

            first = AnyDataset(spec, parse_fn=lambda row: row["src_text"])
            self.assertEqual(list(first), ["zero"])

            source.write_text("src_text\nzero\none\n", encoding="utf-8")
            second = AnyDataset(spec, parse_fn=lambda row: row["src_text"])

            self.assertEqual(list(second), ["zero", "one"])
            self.assertEqual(
                len(list(second.cache_manager.root.rglob("*.parquet"))),
                1,
            )

    def test_multiple_csv_files_use_prepare_process_pool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text("src_text\nzero\n", encoding="utf-8")
            (shard_dir / "1.csv").write_text("src_text\none\n", encoding="utf-8")
            dataset = AnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
            )

            with mock.patch(
                "anydataset.dataset.source.sharded_csv.ProcessPoolExecutor"
            ) as executor:
                pool = executor.return_value.__enter__.return_value
                pool.map.side_effect = lambda function, jobs: map(function, jobs)

                self.assertEqual(list(dataset), ["zero", "one"])

            executor.assert_called_once()

    def test_daemon_worker_prepares_multiple_files_inline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text("src_text\nzero\n", encoding="utf-8")
            (shard_dir / "1.csv").write_text("src_text\none\n", encoding="utf-8")
            dataset = AnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=_src_text,
            )

            with (
                mock.patch(
                    "anydataset.dataset.source.sharded_csv.multiprocessing.current_process"
                ) as current_process,
                mock.patch(
                    "anydataset.dataset.source.sharded_csv.ProcessPoolExecutor"
                ) as executor,
            ):
                current_process.return_value.daemon = True
                self.assertEqual(list(dataset), ["zero", "one"])

            executor.assert_not_called()

    def test_prepared_parquet_supports_index_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text("src_text\nzero\n", encoding="utf-8")
            dataset = AnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
            )

            self.assertTrue(can_select_indexes(dataset))

    def test_prepared_parquet_supports_spawn_loader_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text(
                "src_text\nzero\none\ntwo\nthree\n",
                encoding="utf-8",
            )
            spec = Spec(source="sharded_csv", path=tmpdir)
            factory = partial(AnyDataset, spec, parse_fn=_src_text)
            dataset = factory()
            dataset.prepare()

            batches = map_style_indexed_loader(
                factory,
                sample_count=len(dataset),
                batch_size=1,
                num_workers=2,
                start_method="spawn",
            )

            self.assertEqual(
                [item for batch in batches for item in batch],
                [(0, "zero"), (1, "one"), (2, "two"), (3, "three")],
            )

    def test_prepared_parquet_preserves_string_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text(
                "value,other\n,001\n",
                encoding="utf-8",
            )
            dataset = AnyDataset(Spec(source="sharded_csv", path=tmpdir))

            self.assertEqual(dataset[0], {"value": "", "other": "001"})

    def test_prepared_parquet_supports_header_only_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text("src_text\n", encoding="utf-8")
            dataset = AnyDataset(Spec(source="sharded_csv", path=tmpdir))

            self.assertEqual(len(dataset), 0)

            second = AnyDataset(Spec(source="sharded_csv", path=tmpdir))
            self.assertEqual(len(second), 0)

    def test_prepare_waits_for_concurrent_cache_builder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shard_0"
            shard_dir.mkdir()
            (shard_dir / "0.csv").write_text("src_text\nzero\n", encoding="utf-8")
            spec = Spec(source="sharded_csv", path=tmpdir)
            dataset = AnyDataset(spec)
            cache = dataset.cache_manager.prepare(spec)
            entered = threading.Event()
            release = threading.Event()

            def hold_lock():
                with FileLock(cache.lock_path):
                    entered.set()
                    release.wait()

            holder = threading.Thread(target=hold_lock)
            holder.start()
            self.assertTrue(entered.wait(timeout=1))
            timer = threading.Timer(0.1, release.set)
            timer.start()
            try:
                self.assertEqual(len(dataset), 1)
            finally:
                release.set()
                timer.cancel()
                holder.join()

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
            )

            self.assertEqual(list(dataset.iter_indexed_range(1, 3)), [(1, "one"), (2, "two")])
            self.assertEqual(list(dataset.iter_indexed_shard(2, 1)), [(1, "one")])

    def test_iterable_source_native_shard_keeps_global_indices(self):
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

            dataset = IterableAnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
            )

            self.assertEqual(
                list(dataset.iter_indexed_shard(2, 1)),
                [(1, "one")],
            )

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
            )

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                self.assertEqual(list(dataset), ["zero", "two"])

            log = _single_log(home, "sharded_csv.log")
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
            )

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                self.assertEqual(list(dataset), ["zero", "two"])
                self.assertEqual(list(dataset), ["zero", "two"])

            log = _single_log(home, "sharded_csv.log")
            self.assertEqual(log.read_text(encoding="utf-8").count("WARNING"), 1)


def _single_log(home: Path, name: str) -> Path:
    logs = list((home / "logs").glob(f"*/{name}"))
    if len(logs) != 1:
        raise AssertionError(f"expected one {name}, found: {logs}")
    return logs[0]


def _src_text(row):
    return row["src_text"]


if __name__ == "__main__":
    unittest.main()
