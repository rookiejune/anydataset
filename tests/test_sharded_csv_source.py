import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from functools import partial
from unittest import mock

import anydataset.dataset.source._tabular_parquet as tabular_parquet
import pyarrow.parquet as pyarrow_parquet

from anydataset import (
    AnyDataset,
    Spec,
    resolve_dataset,
)
from anydataset._runtime.parallel import can_select_indexes, map_style_sample_index_loader
from anydataset.cache import FileLock
from anydataset.dataset.source._registry import source_exists
from anydataset.dataset.source.sharded_csv import (
    _CsvShard,
    _ShardedCsvDataset,
    _missing_shard_ranges,
)


class ShardedCsvSourceTest(unittest.TestCase):
    def test_restores_pickle_state_without_prepare_workers(self):
        dataset = _ShardedCsvDataset(Path("unused"))
        state = dataset.__getstate__()
        state.pop("prepare_workers")

        restored = _ShardedCsvDataset.__new__(_ShardedCsvDataset)
        restored.__setstate__(state)

        self.assertIsNone(restored.prepare_workers)

    def test_registered_as_builtin_source(self):
        self.assertTrue(source_exists("sharded_csv"))

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

    def test_prepare_workers_zero_uses_inline_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard = Path(tmpdir) / "shard_0"
            shard.mkdir()
            (shard / "0.csv").write_text("value\nzero\n", encoding="utf-8")
            dataset = AnyDataset(
                Spec(
                    source="sharded_csv",
                    path=tmpdir,
                    load_options={"prepare_workers": 0},
                )
            )

            with mock.patch(
                "anydataset.dataset.source._tabular_parquet.ProcessPoolExecutor",
                side_effect=AssertionError("inline preparation should not spawn"),
            ):
                self.assertEqual(len(dataset), 1)
                self.assertEqual(dataset[0]["value"], "zero")

    def test_rejects_invalid_prepare_workers(self):
        for value in (True, 1.5):
            with self.subTest(value=value):
                dataset = AnyDataset(
                    Spec(
                        source="sharded_csv",
                        path="unused",
                        load_options={"prepare_workers": value},
                    )
                )
                with self.assertRaisesRegex(TypeError, "prepare_workers"):
                    dataset.prepare()

        dataset = AnyDataset(
            Spec(
                source="sharded_csv",
                path="unused",
                load_options={"prepare_workers": -1},
            )
        )
        with self.assertRaisesRegex(ValueError, "prepare_workers"):
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

            dataset = AnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=lambda row: row["src_text"],
            )

            self.assertEqual(list(dataset.iter_shard(2, 1)), [(1, "tea")])

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

            dataset = AnyDataset(
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
            _CsvShard(0, Path("shard_0")),
            _CsvShard(1_000_000_000, Path("shard_1000000000")),
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

            dataset = AnyDataset(
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
            self.assertEqual(list(dataset.iter_shard(2, 1)), [(1, "one")])

    def test_shuffle_uses_parquet_row_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard = Path(tmpdir) / "shard_0"
            shard.mkdir()
            (shard / "0.csv").write_text(
                "value\n0\n1\n2\n3\n4\n",
                encoding="utf-8",
            )
            with mock.patch(
                "anydataset.dataset.source._tabular_parquet.PARQUET_ROW_GROUP_SIZE",
                2,
            ):
                dataset = AnyDataset(Spec(source="sharded_csv", path=tmpdir))
                groups = list(
                    dataset._shuffle(
                        shuffle=True,
                        seed=7,
                        epoch=3,
                        num_replicas=1,
                        rank=0,
                    )
                )
                rank_indexes = [
                    [
                        index
                        for group in dataset._shuffle(
                            shuffle=False,
                            seed=0,
                            epoch=0,
                            num_replicas=2,
                            rank=rank,
                        )
                        for index in group
                    ]
                    for rank in range(2)
                ]

            self.assertEqual(
                sorted(index for group in groups for index in group),
                list(range(5)),
            )
            self.assertEqual(sorted(len(group) for group in groups), [1, 2, 2])
            for group in groups:
                self.assertEqual(min(group) // 2, max(group) // 2)
            self.assertEqual(rank_indexes, [[0, 2, 4], [1, 3]])

    def test_random_access_reuses_offsets_and_parquet_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard = Path(tmpdir) / "shard_0"
            shard.mkdir()
            (shard / "0.csv").write_text(
                "value\n0\n1\n2\n3\n",
                encoding="utf-8",
            )
            with mock.patch(
                "anydataset.dataset.source._tabular_parquet.PARQUET_ROW_GROUP_SIZE",
                2,
            ):
                dataset = AnyDataset(Spec(source="sharded_csv", path=tmpdir))
                prepared = dataset.dataset

            with (
                mock.patch(
                    "anydataset.dataset.source._tabular_parquet.stops",
                    side_effect=AssertionError("row-group stops were recomputed"),
                ),
                mock.patch(
                    "anydataset.dataset.source._tabular_parquet.pq.ParquetFile",
                    wraps=pyarrow_parquet.ParquetFile,
                ) as parquet,
            ):
                self.assertEqual(dataset[0]["value"], "0")
                self.assertEqual(dataset[3]["value"], "3")

            self.assertEqual(parquet.call_count, 1)
            prepared.close()

    def test_iter_shard_reads_each_row_group_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard = Path(tmpdir) / "shard_0"
            shard.mkdir()
            (shard / "0.csv").write_text(
                "value\n0\n1\n2\n3\n4\n",
                encoding="utf-8",
            )
            with mock.patch(
                "anydataset.dataset.source._tabular_parquet.PARQUET_ROW_GROUP_SIZE",
                2,
            ):
                dataset = AnyDataset(Spec(source="sharded_csv", path=tmpdir))
                prepared = dataset.dataset
                with mock.patch.object(
                    prepared,
                    "_read_parquet_group",
                    wraps=prepared._read_parquet_group,
                ) as read_group:
                    rows = list(prepared.iter_shard(3, 1))

            self.assertEqual(
                [(index, row["value"]) for index, row in rows],
                [(1, "1"), (4, "4")],
            )
            self.assertEqual(
                [call.args[1] for call in read_group.call_args_list],
                [0, 2],
            )
            prepared.close()

    def test_parquet_manifest_records_file_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            data = root / "data"
            home.mkdir()
            shard = data / "shard_0"
            shard.mkdir(parents=True)
            (shard / "0.csv").write_text("value\nzero\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                dataset = AnyDataset(Spec(source="sharded_csv", path=str(data)))
                self.assertEqual(len(dataset), 1)
                manifest = next(
                    dataset.cache_manager.root.rglob("sharded_csv_parquet.json")
                )
                record = json.loads(manifest.read_text(encoding="utf-8"))["files"][0]

        self.assertNotIn("device", record)
        self.assertNotIn("inode", record)
        self.assertIn("size", record)
        self.assertIn("mtime_ns", record)
        self.assertIn("ctime_ns", record)

    def test_reuses_prepared_parquet_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            data = root / "data"
            home.mkdir()
            shard_dir = data / "shard_0"
            shard_dir.mkdir(parents=True)
            (shard_dir / "0.csv").write_text(
                "src_text\n"
                "zero\n"
                "one\n",
                encoding="utf-8",
            )
            spec = Spec(source="sharded_csv", path=str(data))

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                dataset = AnyDataset(
                    spec,
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
                    spec,
                    parse_fn=lambda row: row["src_text"],
                )
                with mock.patch.object(
                    tabular_parquet,
                    "convert_file_job",
                ) as convert:
                    self.assertEqual(len(second), 2)

                convert.assert_not_called()

    def test_rebuilds_cache_with_non_integer_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            data = root / "data"
            home.mkdir()
            shard = data / "shard_0"
            shard.mkdir(parents=True)
            (shard / "0.csv").write_text("value\nzero\none\n", encoding="utf-8")
            spec = Spec(
                source="sharded_csv",
                path=str(data),
                load_options={"prepare_workers": 0},
            )
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                first = AnyDataset(spec)
                self.assertEqual(len(first), 2)
                manifest = next(
                    first.cache_manager.root.rglob("sharded_csv_parquet.json")
                )

                for version in (True, 1.0):
                    with self.subTest(version=version):
                        data_json = json.loads(manifest.read_text(encoding="utf-8"))
                        data_json["schema_version"] = version
                        manifest.write_text(json.dumps(data_json), encoding="utf-8")

                        with mock.patch.object(
                            tabular_parquet,
                            "convert_file_job",
                            wraps=tabular_parquet.convert_file_job,
                        ) as convert:
                            rebuilt = AnyDataset(spec)
                            self.assertEqual(len(rebuilt), 2)

                        convert.assert_called_once()
                        repaired = json.loads(manifest.read_text(encoding="utf-8"))
                        self.assertIs(type(repaired["schema_version"]), int)

    def test_rebuilds_changed_parquet_part(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            data = root / "data"
            home.mkdir()
            shard_dir = data / "shard_0"
            shard_dir.mkdir(parents=True)
            source = shard_dir / "0.csv"
            source.write_text("src_text\nzero\n", encoding="utf-8")
            spec = Spec(source="sharded_csv", path=str(data))

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                first = AnyDataset(spec, parse_fn=lambda row: row["src_text"])
                self.assertEqual(list(first), ["zero"])

                source.write_text("src_text\nzero\none\n", encoding="utf-8")
                second = AnyDataset(spec, parse_fn=lambda row: row["src_text"])

                self.assertEqual(list(second), ["zero", "one"])
                self.assertEqual(
                    len(list(second.cache_manager.root.rglob("*.parquet"))),
                    1,
                )

    def test_rebuilds_cache_with_invalid_part_or_row_group_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            data = root / "data"
            home.mkdir()
            shard = data / "shard_0"
            shard.mkdir(parents=True)
            (shard / "0.csv").write_text("value\nzero\none\n", encoding="utf-8")
            spec = Spec(
                source="sharded_csv",
                path=str(data),
                load_options={"prepare_workers": 0},
            )
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                first = AnyDataset(spec)
                self.assertEqual(len(first), 2)
                manifest = next(
                    first.cache_manager.root.rglob("sharded_csv_parquet.json")
                )

                for invalid in ("part", "row_groups", "parquet"):
                    with self.subTest(invalid=invalid):
                        payload = json.loads(manifest.read_text(encoding="utf-8"))
                        record = payload["files"][0]
                        if invalid == "part":
                            record["part"] = (
                                f"../sharded_csv_parquet/{Path(record['part']).name}"
                            )
                        else:
                            if invalid == "row_groups":
                                record["row_groups"] = [1, 1]
                            else:
                                part = (
                                    manifest.parent
                                    / "sharded_csv_parquet"
                                    / record["part"]
                                )
                                part.write_bytes(b"not parquet")
                        manifest.write_text(json.dumps(payload), encoding="utf-8")

                        rebuilt = AnyDataset(spec)
                        self.assertEqual(rebuilt[1]["value"], "one")
                        repaired = json.loads(manifest.read_text(encoding="utf-8"))[
                            "files"
                        ][0]
                        self.assertNotIn("/", repaired["part"])
                        self.assertEqual(repaired["row_groups"], [2])

    def test_rejects_source_change_during_parquet_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "0.csv"
            target = root / "0.parquet"
            source.write_text("value\nbefore\n", encoding="utf-8")
            read_csv = tabular_parquet.read_csv

            def change_source(*args, **kwargs):
                table = read_csv(*args, **kwargs)
                source.write_text("value\nafter\n", encoding="utf-8")
                return table

            with mock.patch.object(
                tabular_parquet,
                "read_csv",
                side_effect=change_source,
            ):
                with self.assertRaisesRegex(ValueError, "changed while preparing"):
                    tabular_parquet.convert_file_job(
                        (0, source, target, ",", "utf-8", "sharded CSV")
                    )

            self.assertFalse(target.exists())

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
                "anydataset.dataset.source._tabular_parquet.ProcessPoolExecutor"
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
                    "anydataset.dataset.source._tabular_parquet.multiprocessing.current_process"
                ) as current_process,
                mock.patch(
                    "anydataset.dataset.source._tabular_parquet.ProcessPoolExecutor"
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

            batches = map_style_sample_index_loader(
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

    def test_map_style_sample_index_iteration_keeps_global_indices(self):
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
            self.assertEqual(list(dataset.iter_shard(2, 1)), [(1, "one")])

    def test_iter_shard_keeps_global_indices(self):
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

            self.assertEqual(
                list(dataset.iter_shard(2, 1)),
                [(1, "one")],
            )
            self.assertEqual(list(dataset.iter_shard(2, 1)), [(1, "one")])
            self.assertEqual(
                list(dataset.iter_shard(2, 1)),
                list(dataset.iter_shard(2, 1)),
            )

    def test_iter_shard_uses_dense_global_modulo(self):
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

            def parse_fn(row):
                return row["src_text"]

            dataset = AnyDataset(
                Spec(source="sharded_csv", path=tmpdir),
                parse_fn=parse_fn,
            )

            self.assertEqual(list(dataset.iter_shard(2, 1)), [(1, "one")])
            self.assertEqual(
                list(dataset.iter_shard(2, 0)),
                [(0, "zero"), (2, "two")],
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

            dataset = AnyDataset(
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

            dataset = AnyDataset(
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

            dataset = AnyDataset(
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
