from __future__ import annotations

import gc
import io
import json
import math
import multiprocessing
import os
import pickle
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock
from collections.abc import Iterator, Sequence
from enum import auto
from functools import partial
from pathlib import Path

import anydataset.filter.runtime.factory as filter_factory_module
import anydataset.filter.api as filter_api_module
import torch
from torch.utils.data import DataLoader

from anydataset import (
    AnyDataset,
    anydataset_home,
    FilterRule,
    Spec,
    register_source,
)
from anydataset._compat import StrEnum
from anydataset.filter import (
    FilterDecision,
    FilteredDataset,
    cleanup_filter_generations,
)
from anydataset.filter.cache.generations import current_filter_generation
from anydataset.filter.cache.resume import (
    iter_filter_fragment_chunks,
    prepare_filter_resume_dir,
    write_filter_fragment,
)
from anydataset.filter.types import _FilterChunk, _FilterMetricsRow
from anydataset.dataset.source._registry import source_exists
from anydataset.provider_service import ProviderServer, RemoteFilterFactory
from anydataset.runtime import Runtime
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
)
from anydataset.filter.runtime.collect import collect_ranges_parallel
from anydataset.filter.runtime.collect import _read_worker_message as read_worker_message
from anydataset.filter.cache.storage import metrics_ready, read_index_rows, write_index_rows
from anydataset.store import DatasetWriter
from anydataset.store.jsonio import (
    read_json as read_store_json,
    write_json as write_store_json,
)


class FilteredDatasetTest(unittest.TestCase):
    def test_rule_apply_partitions_bool_labels(self):
        _register_rows_source("unit_test_filter_bool")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            dataset = _dataset("unit_test_filter_bool", [0, 1, 2, 3])
            rule = FilterRule(
                name="even",
                factory=lambda: lambda sample: _value(sample) % 2 == 0,
            )

            result = rule.apply(dataset_factory=lambda: dataset, device="cpu")
            accepted = result.select_by("accept")
            rejected = result.select_by("reject")

        self.assertEqual(result.labels, ("accept", "reject"))
        self.assertEqual(result.counts, {"accept": 2, "reject": 2})
        self.assertEqual(_values(accepted), [0, 2])
        self.assertEqual(_values(rejected), [1, 3])
        self.assertEqual(accepted.indices, (0, 2))
        self.assertEqual(rejected.indices, (1, 3))

    def test_rule_apply_partitions_string_and_enum_labels(self):
        _register_rows_source("unit_test_filter_labels")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            dataset = _dataset("unit_test_filter_labels", [0, 1, 2, 3])
            rule = FilterRule(
                name="route",
                factory=_route_factory,
            )

            result = rule.apply(dataset_factory=lambda: dataset, device="cpu")
            selected = result.select_by(Route.REVIEW, "reject")

        self.assertEqual(result.labels, ("accept", "review", "reject"))
        self.assertEqual(result.counts, {"accept": 1, "review": 2, "reject": 1})
        self.assertEqual(selected.labels, ("review", "reject"))
        self.assertEqual(_values(selected), [1, 2, 3])
        self.assertEqual(selected.indices, (1, 2, 3))

    def test_multi_label_index_merges_shards_lazily(self):
        _register_rows_source("unit_test_filter_lazy_merge")
        dataset = _dataset("unit_test_filter_lazy_merge", list(range(6)))
        result = FilterRule("mod_three", _mod_three_factory).apply(
            dataset_factory=lambda: dataset,
            device="cpu",
            max_shard_samples=1,
        )

        with mock.patch(
            "anydataset.filter.cache.storage.read_index_rows",
            wraps=read_index_rows,
        ) as read:
            selected = result.select_by("zero", "two")

            self.assertEqual(len(selected), 4)
            self.assertEqual(read.call_count, 0)
            self.assertEqual(selected.global_index(0), 0)
            self.assertEqual(read.call_count, 2)
            self.assertEqual(selected.indices, (0, 2, 3, 5))
            self.assertEqual(read.call_count, 4)

    def test_iter_indices_and_partitions_are_lazy(self):
        _register_rows_source("unit_test_filter_lazy_iterators")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_lazy_iterators", list(range(6)))
            result = (
                FilterRule("mod_three", _mod_three_factory)
                .apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                    max_shard_samples=1,
                )
                .select_by("zero", "two")
            )

            with mock.patch(
                "anydataset.filter.cache.storage.read_index_rows",
                wraps=read_index_rows,
            ) as read:
                indices = result.iter_indices()
                self.assertEqual(read.call_count, 0)
                self.assertEqual(next(indices), 0)
                self.assertGreater(read.call_count, 0)

            partitions = dict(result.iter_partitions())
            self.assertEqual(set(partitions), {"zero", "two"})
            self.assertEqual(tuple(partitions["zero"]), (0, 3))
            self.assertEqual(tuple(partitions["two"]), (2, 5))

    def test_rule_version_and_id_isolated_in_cache_identity(self):
        _register_rows_source("unit_test_filter_rule_identity")
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(
                os.environ,
                {"ANYDATASET_HOME": tmpdir},
            ),
        ):
            dataset = _dataset("unit_test_filter_rule_identity", [0, 1])
            first = FilterRule(
                "same",
                lambda: lambda _sample: True,
                rule_id="quality-check",
                version="v1",
            ).apply(dataset_factory=lambda: dataset, device="cpu")
            second = FilterRule(
                "same",
                lambda: lambda _sample: False,
                rule_id="quality-check",
                version="v2",
            ).apply(dataset_factory=lambda: dataset, device="cpu")

            metadata = json.loads(
                (first.cache_path / "rule.json").read_text(encoding="utf-8")
            )

        self.assertNotEqual(first.cache_path, second.cache_path)
        self.assertEqual(
            metadata["rule"],
            {
                "name": "same",
                "rule_id": "quality-check",
                "version": "v1",
            },
        )
        self.assertEqual(first.rule.rule_id, "quality-check")
        self.assertEqual(first.rule.version, "v1")

    def test_rule_content_id_isolated_in_cache_identity(self):
        _register_rows_source("unit_test_filter_content_id")
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}),
        ):
            dataset = _dataset("unit_test_filter_content_id", [0, 1])
            first = FilterRule(
                "same",
                lambda: lambda _sample: True,
                content_id="sha-a",
            ).apply(dataset_factory=lambda: dataset, device="cpu")
            second = FilterRule(
                "same",
                lambda: lambda _sample: False,
                content_id="sha-b",
            ).apply(dataset_factory=lambda: dataset, device="cpu")
            metadata = json.loads(
                (first.cache_path / "rule.json").read_text(encoding="utf-8")
            )

        self.assertNotEqual(first.cache_path, second.cache_path)
        self.assertEqual(
            metadata["rule"],
            {"name": "same", "content_id": "sha-a"},
        )
        self.assertEqual(first.rule.content_id, "sha-a")

    def test_rebuild_forces_new_generation_under_same_identity(self):
        _register_rows_source("unit_test_filter_rebuild")
        dataset = _dataset("unit_test_filter_rebuild", [0, 1])
        rule = FilterRule("all", _true_factory)
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                first = rule.apply(dataset_factory=lambda: dataset, device="cpu")
                first_path = first.cache_path
                second = rule.apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                    rebuild=True,
                )

                self.assertNotEqual(first_path, second.cache_path)
                self.assertEqual(_values(second), [0, 1])

    def test_select_unknown_label_returns_empty_dataset(self):
        _register_rows_source("unit_test_filter_unknown")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_unknown", [0])
            result = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            ).apply(dataset_factory=lambda: dataset, device="cpu")

            selected = result.select_by("review")

        self.assertEqual(len(selected), 0)
        self.assertEqual(selected.indices, ())

    def test_select_deduplicates_labels(self):
        _register_rows_source("unit_test_filter_deduplicate")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_deduplicate", [0, 1])
            result = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            ).apply(dataset_factory=lambda: dataset, device="cpu")

            selected = result.select_by(True, "accept")

        self.assertEqual(selected.labels, ("accept",))
        self.assertEqual(selected.indices, (0, 1))

    def test_rule_apply_selects_all_labels_by_default(self):
        _register_rows_source("unit_test_filter_all_labels")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_all_labels", [0, 1])
            result = FilterRule(
                name="even",
                factory=lambda: lambda sample: _value(sample) % 2 == 0,
            ).apply(dataset_factory=lambda: dataset, device="cpu")

        self.assertEqual(result.available_labels, ("accept", "reject"))
        self.assertEqual(result.labels, ("accept", "reject"))
        self.assertEqual(_values(result), [0, 1])

    def test_rule_predicate_receives_full_sample(self):
        _register_rows_source("unit_test_filter_full_sample")
        seen = []
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_full_sample", [0])
            rule = FilterRule(
                name="full",
                factory=lambda: lambda sample: seen.append(sample) or True,
            )

            rule.apply(dataset_factory=lambda: dataset, device="cpu")

        sample = seen[0]
        audio = sample[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(audio.views, {AudioView.WAVEFORM: 0})
        self.assertEqual(audio.meta, {AudioMeta.LABEL: 0})

    def test_rule_apply_reuses_ready_cache(self):
        _register_rows_source("unit_test_filter_reuses")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            dataset = _dataset("unit_test_filter_reuses", [0, 1, 2, 3])
            first_rule = FilterRule(
                name="gte_two",
                factory=lambda: lambda sample: _value(sample) >= 2,
            )
            first_rule.apply(dataset_factory=lambda: dataset, device="cpu")
            second_rule = FilterRule(
                name="gte_two",
                factory=lambda: lambda sample: calls.append(sample) or False,
            )

            result = second_rule.apply(dataset_factory=lambda: dataset, device="cpu")

        self.assertEqual(_values(result.select_by("accept")), [2, 3])
        self.assertEqual(calls, [])

    def test_concurrent_cold_cache_waits_and_reuses_result(self):
        _register_rows_source("unit_test_filter_concurrent_cache")
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            dataset = _dataset("unit_test_filter_concurrent_cache", [0, 1])
            entered = threading.Event()
            release = threading.Event()
            calls = []
            results = []
            errors = []

            def predicate_factory():
                def predicate(sample):
                    calls.append(_value(sample))
                    if len(calls) == 1:
                        entered.set()
                        if not release.wait(timeout=5):
                            raise TimeoutError("test predicate was not released")
                    return True

                return predicate

            rule = FilterRule("concurrent", predicate_factory)

            def apply():
                try:
                    results.append(
                        rule.apply(dataset_factory=lambda: dataset, device="cpu")
                    )
                except Exception as exc:
                    errors.append(exc)

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                first = threading.Thread(target=apply)
                first.start()
                self.assertTrue(entered.wait(timeout=5))
                second = threading.Thread(target=apply)
                second.start()
                time.sleep(0.05)
                self.assertTrue(second.is_alive())
                release.set()
                first.join(timeout=10)
                second.join(timeout=10)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(calls, [0, 1])
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].cache_path, results[1].cache_path)
            pointer = json.loads(
                (results[0].cache_path.parents[1] / "current.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(pointer["generation"], results[0].cache_path.name)

    def test_rule_apply_rechecks_if_cache_snapshot_disappears(self):
        _register_rows_source("unit_test_filter_snapshot_recheck")
        dataset = _dataset("unit_test_filter_snapshot_recheck", [0])
        calls = []
        rule = FilterRule(
            "snapshot_recheck",
            lambda: lambda sample: calls.append(_value(sample)) or True,
        )
        first = rule.apply(dataset_factory=lambda: dataset, device="cpu")
        removed = False

        def disappearing_read_json(path):
            nonlocal removed
            path = Path(path)
            if path.name == "rule.json" and not removed:
                removed = True
                path.unlink()
            return read_store_json(path)

        with mock.patch(
            "anydataset.filter.cache.ready.read_json",
            side_effect=disappearing_read_json,
        ):
            restored = rule.apply(dataset_factory=lambda: dataset, device="cpu")

        self.assertNotEqual(restored.cache_path, first.cache_path)
        self.assertEqual(_values(restored), [0])
        self.assertEqual(calls, [0, 0])

    def test_current_generation_rejects_non_integer_schema_version(self):
        for version in (True, 1.0):
            with self.subTest(version=version):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    write_store_json(
                        root / "current.json",
                        {
                            "schema_version": version,
                            "generation": "0" * 32,
                        },
                    )

                    with self.assertRaisesRegex(ValueError, "schema_version mismatch"):
                        current_filter_generation(root)

    def test_live_lazy_index_keeps_immutable_generation(self):
        _register_rows_source("unit_test_filter_live_snapshot")
        dataset = _dataset("unit_test_filter_live_snapshot", [0, 1, 2, 3])
        rule = FilterRule(
            "all_with_metrics",
            lambda: (
                lambda sample: FilterDecision(
                    label=True,
                    metrics={"score": _value(sample)},
                )
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                live = rule.apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                    max_shard_samples=2,
                )
                self.assertEqual(live.global_index(0), 0)
                live_path = live.cache_path

                current = rule.apply(
                    dataset_factory=lambda: dataset,
                    metrics=True,
                    device="cpu",
                    max_shard_samples=3,
                )

                self.assertNotEqual(current.cache_path, live_path)
                self.assertEqual(live.global_index(2), 2)
                self.assertTrue(live_path.is_dir())
                self.assertNotIn(
                    live_path,
                    cleanup_filter_generations(current.cache_path),
                )

    def test_generation_cleanup_waits_for_live_factory(self):
        _register_rows_source("unit_test_filter_generation_cleanup")
        dataset = _dataset("unit_test_filter_generation_cleanup", [0, 1])
        rule = FilterRule("all", _true_factory)
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                first = rule.apply(dataset_factory=lambda: dataset, device="cpu")
                old_path = first.cache_path
                factory = first.dataset_factory
                current_pointer = old_path.parents[1] / "current.json"
                current_pointer.unlink()
                current = rule.apply(dataset_factory=lambda: dataset, device="cpu")

                del first
                gc.collect()
                self.assertEqual(cleanup_filter_generations(current.cache_path), ())
                restored = factory()
                self.assertEqual(_values(restored), [0, 1])

                del restored
                del factory
                gc.collect()
                removed = cleanup_filter_generations(current.cache_path)

                self.assertEqual(removed, (old_path,))
                self.assertFalse(old_path.exists())

    def test_publish_collects_unleased_generation(self):
        _register_rows_source("unit_test_filter_generation_publish_cleanup")
        dataset = _dataset("unit_test_filter_generation_publish_cleanup", [0])
        rule = FilterRule("all", _true_factory)
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                old_path = rule.apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                ).cache_path
                gc.collect()
                (old_path.parents[1] / "current.json").unlink()

                current = rule.apply(dataset_factory=lambda: dataset, device="cpu")

                self.assertNotEqual(current.cache_path, old_path)
                self.assertFalse(old_path.exists())

    @unittest.skipUnless(
        "fork" in multiprocessing.get_all_start_methods(), "requires fork"
    )
    def test_fork_child_close_does_not_release_parent_generation_lease(self):
        _register_rows_source("unit_test_filter_fork_generation_lease")
        dataset = _dataset("unit_test_filter_fork_generation_lease", [0, 1])
        rule = FilterRule("all", _metric_factory)
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                first = rule.apply(dataset_factory=lambda: dataset, device="cpu")
                old_path = first.cache_path
                context = multiprocessing.get_context("fork")
                closed = context.Event()
                process = context.Process(
                    target=_close_filter_generation_lease,
                    args=(first, closed),
                )
                process.start()
                self.assertTrue(closed.wait(timeout=5))
                process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)

                current = rule.apply(
                    dataset_factory=lambda: dataset,
                    metrics=True,
                    device="cpu",
                )

                self.assertNotEqual(current.cache_path, old_path)
                self.assertTrue(old_path.is_dir())
                self.assertEqual(first.global_index(1), 1)

                del first
                gc.collect()
                self.assertEqual(
                    cleanup_filter_generations(current.cache_path),
                    (old_path,),
                )

    @unittest.skipUnless(
        "fork" in multiprocessing.get_all_start_methods(), "requires fork"
    )
    def test_fork_parent_close_does_not_release_child_generation_lease(self):
        _register_rows_source("unit_test_filter_fork_child_generation_lease")
        dataset = _dataset("unit_test_filter_fork_child_generation_lease", [0, 1])
        rule = FilterRule("all", _metric_factory)
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                first = rule.apply(dataset_factory=lambda: dataset, device="cpu")
                old_path = first.cache_path
                context = multiprocessing.get_context("fork")
                ready = context.Event()
                release = context.Event()
                process = context.Process(
                    target=_use_filter_generation_after_release,
                    args=(first, ready, release),
                )
                process.start()
                self.assertTrue(ready.wait(timeout=5))

                current = rule.apply(
                    dataset_factory=lambda: dataset,
                    metrics=True,
                    device="cpu",
                )
                del first
                gc.collect()

                self.assertNotEqual(current.cache_path, old_path)
                self.assertEqual(cleanup_filter_generations(current.cache_path), ())

                release.set()
                process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)
                self.assertEqual(
                    cleanup_filter_generations(current.cache_path),
                    (old_path,),
                )

    def test_metrics_iterator_keeps_its_generation(self):
        _register_rows_source("unit_test_filter_metrics_generation")
        dataset = _dataset("unit_test_filter_metrics_generation", [0, 1])
        rule = FilterRule("metrics", _metric_factory)
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                first = rule.apply(
                    dataset_factory=lambda: dataset,
                    metrics=True,
                    device="cpu",
                )
                old_path = first.cache_path
                (old_path.parents[1] / "current.json").unlink()
                current = rule.apply(
                    dataset_factory=lambda: dataset,
                    metrics=True,
                    device="cpu",
                )

                rows = first.iter_metrics()
                del first
                gc.collect()
                first_row = next(rows)

                self.assertNotEqual(current.cache_path, old_path)
                self.assertEqual(cleanup_filter_generations(current.cache_path), ())
                self.assertEqual(
                    [row["index"] for row in (first_row, *rows)],
                    [0, 1],
                )
                self.assertTrue(old_path.is_dir())

                del rows
                gc.collect()
                self.assertEqual(
                    cleanup_filter_generations(current.cache_path),
                    (old_path,),
                )

    def test_lazy_index_validates_actual_shard_count(self):
        _register_rows_source("unit_test_filter_shard_count")
        dataset = _dataset("unit_test_filter_shard_count", [0])
        rule = FilterRule("all", _true_factory)
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                first = rule.apply(dataset_factory=lambda: dataset, device="cpu")
                manifest = json.loads(
                    (first.cache_path / "partitions.json").read_text(encoding="utf-8")
                )
                relpath = manifest["partitions"][0]["files"][0]["file"]
                write_index_rows(first.cache_path / relpath, [0, 0])

                restored = rule.apply(dataset_factory=lambda: dataset, device="cpu")
                with self.assertRaisesRegex(ValueError, "row count"):
                    _ = restored.indices

    def test_rule_apply_rebuilds_partition_count_mismatch(self):
        _register_rows_source("unit_test_filter_partition_count")
        dataset = _dataset("unit_test_filter_partition_count", [0, 1])
        calls = []
        rule = FilterRule(
            "partition_count",
            lambda: lambda sample: calls.append(_value(sample)) or True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                first = rule.apply(dataset_factory=lambda: dataset, device="cpu")
                path = first.cache_path / "partitions.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifest["partitions"][0]["count"] = 0
                path.write_text(json.dumps(manifest), encoding="utf-8")

                restored = rule.apply(dataset_factory=lambda: dataset, device="cpu")

        self.assertEqual(restored.counts, {"accept": 2})
        self.assertEqual(calls, [0, 1, 0, 1])

    def test_rule_apply_rebuilds_metrics_count_mismatch(self):
        _register_rows_source("unit_test_filter_metrics_count")
        dataset = _dataset("unit_test_filter_metrics_count", [0, 1])
        calls = []
        rule = FilterRule(
            "metrics_count",
            lambda: (
                lambda sample: FilterDecision(
                    label=True,
                    metrics={"score": calls.append(_value(sample)) or _value(sample)},
                )
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                first = rule.apply(
                    dataset_factory=lambda: dataset,
                    metrics=True,
                    device="cpu",
                )
                path = first.metrics_path / "metrics.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifest["count"] = 0
                manifest["files"] = []
                path.write_text(json.dumps(manifest), encoding="utf-8")

                restored = rule.apply(
                    dataset_factory=lambda: dataset,
                    metrics=True,
                    device="cpu",
                )
                rows = list(restored.iter_metrics())

        self.assertEqual(len(rows), 2)
        self.assertEqual(calls, [0, 1, 0, 1])

    def test_metrics_manifest_rejects_non_integer_schema_version(self):
        for version in (True, 1.0):
            with self.subTest(version=version):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir)
                    write_store_json(
                        path / "metrics.json",
                        {
                            "schema_version": version,
                            "count": 0,
                            "files": [],
                        },
                    )

                    self.assertFalse(metrics_ready(path, expected_count=0))

    def test_filter_resume_rejects_duplicate_partition_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            write_filter_fragment(
                path,
                (0,),
                _FilterChunk(partitions={"accept": [0]}, metrics=()),
            )
            fragment = next(child for child in path.iterdir() if child.is_dir())
            manifest_path = fragment / "fragment.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["partitions"].append(dict(manifest["partitions"][0]))
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate.*partition label"):
                tuple(iter_filter_fragment_chunks(path, metrics=False))

    def test_filter_resume_fragment_rejects_non_integer_schema_version(self):
        for version in (True, 1.0):
            with self.subTest(version=version):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir)
                    write_filter_fragment(
                        path,
                        (0,),
                        _FilterChunk(partitions={"accept": [0]}, metrics=()),
                    )
                    fragment = next(child for child in path.iterdir() if child.is_dir())
                    manifest_path = fragment / "fragment.json"
                    manifest = read_store_json(manifest_path)
                    manifest["schema_version"] = version
                    write_store_json(manifest_path, manifest)

                    with self.assertRaisesRegex(ValueError, "schema_version mismatch"):
                        tuple(iter_filter_fragment_chunks(path, metrics=False))

    def test_filter_resume_metadata_rejects_non_integer_schema_version(self):
        for version in (True, 1.0):
            with self.subTest(version=version):
                with tempfile.TemporaryDirectory() as tmpdir:
                    cache_path = Path(tmpdir) / "cache"
                    path = prepare_filter_resume_dir(
                        cache_path,
                        {"identity": "same"},
                        metrics=False,
                    )
                    metadata_path = path / "resume.json"
                    metadata = read_store_json(metadata_path)
                    metadata["schema_version"] = version
                    write_store_json(metadata_path, metadata)
                    stale = path / "stale"
                    stale.write_text("stale", encoding="utf-8")

                    restored = prepare_filter_resume_dir(
                        cache_path,
                        {"identity": "same"},
                        metrics=False,
                    )

                    self.assertEqual(restored, path)
                    self.assertFalse(stale.exists())

    def test_filter_resume_validates_rows_against_scan_count(self):
        cases = (
            (
                _FilterChunk(partitions={"accept": [0, 1]}, metrics=()),
                False,
                "partition rows",
            ),
            (
                _FilterChunk(partitions={"accept": [0]}, metrics=()),
                True,
                "metrics rows",
            ),
            (
                _FilterChunk(partitions={"accept": [0, 0]}, metrics=()),
                False,
                "duplicate partition index",
            ),
            (
                _FilterChunk(
                    partitions={"accept": [0]},
                    metrics=(_FilterMetricsRow(index=0, label="reject", metrics={}),),
                ),
                True,
                "metrics do not match partitions",
            ),
        )
        for chunk, metrics, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir)
                    scan_indexes = (0, 1) if "duplicate" in message else (0,)
                    write_filter_fragment(path, scan_indexes, chunk)

                    with self.assertRaisesRegex(ValueError, message):
                        tuple(iter_filter_fragment_chunks(path, metrics=metrics))

    def test_rule_apply_resumes_from_completed_chunks(self):
        _register_rows_source("unit_test_filter_resume")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            calls = root / "calls.txt"
            reads = root / "reads.txt"
            marker = root / "failed.txt"
            dataset = _dataset(
                "unit_test_filter_resume",
                [0, 1, 2, 3],
                read_log=reads,
            )
            rule = FilterRule(
                name="resume_even",
                factory=lambda: _FailOnceFilter(calls, marker, fail_value=2),
            )

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                with self.assertRaisesRegex(RuntimeError, "stop after first chunk"):
                    rule.apply(
                        dataset_factory=lambda: dataset,
                        device="cpu",
                        commit_samples=2,
                    )

                self.assertEqual(
                    calls.read_text(encoding="utf-8").splitlines(),
                    ["0", "1", "2"],
                )

                result = rule.apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                    commit_samples=2,
                )
                log_text = _read_filter_log()
                events = _read_events()

            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["0", "1", "2", "2", "3"],
            )
            self.assertEqual(
                reads.read_text(encoding="utf-8").splitlines(),
                ["0", "1", "2", "2", "3"],
            )
            self.assertEqual(result.counts, {"accept": 2, "reject": 2})
            self.assertEqual(result.select_by("accept").indices, (0, 2))
            self.assertIn("ranges=2-3", log_text)
            resume_event = [
                entry for entry in events if entry["event"] == "filter_resume"
            ][-1]
            self.assertEqual(resume_event["fields"]["expected"], 4)
            self.assertEqual(resume_event["fields"]["completed"], 2)
            self.assertEqual(resume_event["fields"]["missing"], 2)
            self.assertEqual(resume_event["fields"]["ranges"], "2-3")
            cache_root = result.cache_path.parents[1]
            self.assertFalse(
                (cache_root.parent / f".{cache_root.name}.resume").exists()
            )

    def test_rule_apply_resumes_metrics_chunks(self):
        _register_rows_source("unit_test_filter_resume_metrics")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calls = root / "metric-calls.txt"
            marker = root / "metric-failed.txt"
            dataset = _dataset("unit_test_filter_resume_metrics", [0, 1, 2, 3])
            rule = FilterRule(
                name="resume_metrics",
                factory=lambda: _FailOnceMetricFilter(calls, marker, fail_value=2),
            )

            with self.assertRaisesRegex(RuntimeError, "stop after first chunk"):
                rule.apply(
                    dataset_factory=lambda: dataset,
                    metrics=True,
                    device="cpu",
                    commit_samples=2,
                )

            result = rule.apply(
                dataset_factory=lambda: dataset,
                metrics=True,
                device="cpu",
                commit_samples=2,
            )
            rows = list(result.iter_metrics())

            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["0", "1", "2", "2", "3"],
            )
            self.assertEqual(
                rows,
                [
                    {"index": 0, "label": "accept", "metrics": {"score": 0}},
                    {"index": 1, "label": "reject", "metrics": {"score": 1}},
                    {"index": 2, "label": "accept", "metrics": {"score": 2}},
                    {"index": 3, "label": "reject", "metrics": {"score": 3}},
                ],
            )

    def test_chained_filter_resume_skips_view_indexes(self):
        _register_rows_source("unit_test_filter_chain_resume")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calls = root / "chain-calls.txt"
            marker = root / "chain-failed.txt"
            dataset = _dataset("unit_test_filter_chain_resume", [0, 1, 2, 3, 4])
            first = (
                FilterRule(
                    name="gte_two",
                    factory=lambda: lambda sample: _value(sample) >= 2,
                )
                .apply(dataset_factory=lambda: dataset, device="cpu")
                .select_by("accept")
            )
            calls.write_text("", encoding="utf-8")
            second = FilterRule(
                name="resume_chain_even",
                factory=lambda: _FailOnceFilter(calls, marker, fail_value=4),
            )

            with self.assertRaisesRegex(RuntimeError, "stop after first chunk"):
                second.apply(
                    dataset_factory=first.dataset_factory,
                    device="cpu",
                    commit_samples=2,
                )

            result = second.apply(
                dataset_factory=first.dataset_factory,
                device="cpu",
                commit_samples=2,
            )

            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["2", "3", "4", "4"],
            )
            self.assertEqual(result.select_by("accept").indices, (2, 4))

    def test_rule_apply_creates_predicate_from_factory(self):
        _register_rows_source("unit_test_filter_factory")
        events = []
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_factory", [0, 1])

            def factory():
                events.append("factory")
                return lambda sample: events.append(_value(sample)) or True

            rule = FilterRule(name="factory", factory=factory)
            rule.apply(dataset_factory=lambda: dataset, device="cpu")
            rule.apply(dataset_factory=lambda: dataset, device="cpu")

        self.assertEqual(events, ["factory", 0, 1])

    def test_rule_apply_rebuilds_when_base_count_changes(self):
        _register_rows_source("unit_test_filter_rebuilds")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            first = _dataset("unit_test_filter_rebuilds", [0, 1, 2])
            rule = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            )
            rule.apply(dataset_factory=lambda: first, device="cpu")
            second = _dataset("unit_test_filter_rebuilds", [0, 1, 2, 3])

            result = rule.apply(dataset_factory=lambda: second, device="cpu")

        self.assertEqual(_values(result.select_by("accept")), [0, 1, 2, 3])

    def test_rule_apply_reuses_same_name_cache(self):
        _register_rows_source("unit_test_filter_same_name")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            dataset = _dataset("unit_test_filter_same_name", [0])
            first_rule = FilterRule(
                name="same_name",
                factory=lambda: lambda sample: True,
            )
            first_result = first_rule.apply(
                dataset_factory=lambda: dataset, device="cpu"
            )
            second_rule = FilterRule(
                name="same_name",
                factory=lambda: lambda sample: calls.append(sample) or False,
            )

            second_result = second_rule.apply(
                dataset_factory=lambda: dataset, device="cpu"
            )

        self.assertEqual(first_result.cache_path, second_result.cache_path)
        self.assertEqual(calls, [])
        self.assertEqual(_values(second_result.select_by("accept")), [0])

    def test_filtered_dataset_selects_from_filter_cache(self):
        _register_rows_source("unit_test_filter_direct_select")
        calls = []
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_direct_select", [0, 1, 2])
            rule = FilterRule(
                name="even",
                factory=lambda: (
                    lambda sample: (
                        calls.append(_value(sample)) or _value(sample) % 2 == 0
                    )
                ),
            )

            result = rule.apply(dataset_factory=lambda: dataset, device="cpu")
            filtered = result.select_by("accept")

        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(_values(filtered), [0, 2])
        self.assertEqual(filtered.indices, (0, 2))

    def test_filtered_dataset_constructor_matches_apply_select_by(self):
        _register_rows_source("unit_test_filter_direct_constructor")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_direct_constructor", [0, 1, 2, 3])
            rule = FilterRule(
                name="even",
                factory=lambda: lambda sample: _value(sample) % 2 == 0,
            )

            selected = rule.apply(
                dataset_factory=lambda: dataset,
                device="cpu",
            ).select_by("accept")
            direct = FilteredDataset(
                rule.name,
                rule.factory,
                dataset_factory=lambda: dataset,
                labels="accept",
                device="cpu",
            )

        self.assertEqual(selected.cache_path, direct.cache_path)
        self.assertEqual(selected.labels, direct.labels)
        self.assertEqual(selected.indices, direct.indices)
        self.assertEqual(_values(direct), [0, 2])

    def test_filtered_dataset_constructor_rejects_invalid_rule_name(self):
        _register_rows_source("unit_test_filter_direct_reject")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_direct_reject", [0])

            with self.assertRaises(TypeError):
                FilteredDataset(1, _true_factory, dataset_factory=lambda: dataset)

    def test_filtered_dataset_constructor_rejects_unknown_apply_kwargs(self):
        _register_rows_source("unit_test_filter_direct_bad_option")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_direct_bad_option", [0])

            with self.assertRaises(TypeError):
                FilteredDataset(
                    "bad",
                    _true_factory,
                    dataset_factory=lambda: dataset,
                    unknown=True,
                )

    def test_filtered_dataset_shards_selected_order(self):
        _register_rows_source("unit_test_filter_shards")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_shards", [0, 1, 2, 3, 4])
            filtered = (
                FilterRule(
                    name="all",
                    factory=lambda: lambda sample: True,
                )
                .apply(dataset_factory=lambda: dataset, device="cpu")
                .select_by("accept")
            )

            shard = [
                (index, _value(sample))
                for index, sample in filtered.iter_shard(2, 1)
            ]

        self.assertEqual(shard, [(1, 1), (3, 3)])

    def test_filtered_dataset_iter_shard_keeps_physical_indices(self):
        _register_rows_source("unit_test_filter_iter_shard_indices")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_iter_shard_indices", [0, 1, 2, 3, 4])
            filtered = (
                FilterRule(
                    name="even",
                    factory=lambda: lambda sample: _value(sample) % 2 == 0,
                )
                .apply(dataset_factory=lambda: dataset, device="cpu")
                .select_by("accept")
            )

            shard = [
                (index, filtered.global_index(index), _value(sample))
                for index, sample in filtered.iter_shard(2, 1)
            ]

        self.assertEqual(shard, [(1, 2, 2)])

    def test_filtered_dataset_pickle_rebuilds_cache_from_factory(self):
        dataset_factory = partial(
            _unpicklable_dataset,
            "unit_test_filter_pickle",
            list(range(4)),
        )
        filtered = (
            FilterRule(
                name="even",
                factory=_metric_factory,
            )
            .apply(
                dataset_factory=dataset_factory,
                input_id="pickle-input-v1",
                metrics=True,
                device="cpu",
            )
            .select_by("accept")
        )

        restored_cache = pickle.loads(pickle.dumps(filtered._cache))
        restored = pickle.loads(pickle.dumps(filtered))

        self.assertEqual(restored_cache.labels, ("accept", "reject"))
        self.assertEqual(restored.cache_path, filtered.cache_path)
        self.assertEqual(restored.metrics_path, filtered.metrics_path)
        self.assertEqual(restored.input_id, "pickle-input-v1")
        self.assertEqual(_values(restored), [0, 2])
        self.assertEqual(len(list(restored.iter_metrics())), 4)

    def test_filter_rule_restores_legacy_slots_pickle(self):
        current = FilterRule(
            "current",
            _true_factory,
            rule_id="stable-rule",
            version="v2",
        )
        current_restored = pickle.loads(pickle.dumps(current))

        self.assertEqual(current_restored, current)
        self.assertIs(current_restored.factory, _true_factory)

        legacy_rule_type = type(
            "FilterRule",
            (),
            {
                "__module__": filter_api_module.__name__,
                "__slots__": ("_factory", "_name"),
                "__init__": _init_legacy_filter_rule,
            },
        )
        with mock.patch.object(filter_api_module, "FilterRule", legacy_rule_type):
            payload = pickle.dumps(legacy_rule_type("legacy", _true_factory))

        restored = pickle.loads(payload)

        self.assertIsInstance(restored, FilterRule)
        self.assertEqual(restored.name, "legacy")
        self.assertIs(restored.factory, _true_factory)
        self.assertEqual(restored.rule_id, "legacy")
        self.assertIsNone(restored.version)

    def test_filter_restores_legacy_reduce_signatures(self):
        dataset_factory = partial(
            _unpicklable_dataset,
            "unit_test_filter_legacy_pickle",
            list(range(4)),
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.dict(
                os.environ,
                {"ANYDATASET_HOME": tmpdir},
            ),
        ):
            filtered = (
                FilterRule("even", _metric_factory)
                .apply(
                    dataset_factory=dataset_factory,
                    input_id="legacy-input",
                    device="cpu",
                )
                .select_by("accept")
            )
            cache_pickle = _LegacyReduce(
                filter_api_module._restore_filter_cache,
                (
                    dataset_factory,
                    filtered.rule.name,
                    filtered.cache_path,
                    filtered.metrics_path,
                    filtered.input_id,
                ),
            )
            restored_cache = pickle.loads(pickle.dumps(cache_pickle))

            current_factory = filtered.dataset_factory
            factory_pickle = _LegacyReduce(
                filter_factory_module.restore_filtered_dataset_factory,
                (
                    current_factory.base,
                    current_factory.rule_name,
                    current_factory.labels,
                    current_factory.cache_path,
                    current_factory.metrics_path,
                    current_factory.input_id,
                ),
            )
            restored_factory = pickle.loads(pickle.dumps(factory_pickle))
            restored = restored_factory()

            self.assertEqual(restored_cache.rule.name, "even")
            self.assertEqual(restored_cache.rule.rule_id, "even")
            self.assertIsNone(restored_cache.rule.version)
            self.assertEqual(restored_cache.labels, ("accept", "reject"))
            self.assertEqual(restored_factory.rule_id, "even")
            self.assertIsNone(restored_factory.version)
            self.assertEqual(restored.rule.rule_id, "even")
            self.assertIsNone(restored.rule.version)
            self.assertEqual(_values(restored), [0, 2])

    def test_filtered_dataset_factory_preserves_metrics(self):
        dataset_factory = partial(
            _unpicklable_dataset,
            "unit_test_filter_factory_metrics",
            list(range(4)),
        )
        filtered = FilterRule(
            name="even",
            factory=_metric_factory,
        ).apply(
            dataset_factory=dataset_factory,
            metrics=True,
            device="cpu",
        )

        factory = pickle.loads(pickle.dumps(filtered.dataset_factory))
        restored = factory()

        self.assertEqual(restored.metrics_path, filtered.metrics_path)
        self.assertEqual(list(restored.iter_metrics()), list(filtered.iter_metrics()))

    def test_filtered_dataset_spawn_loader_reads_selected_samples(self):
        dataset_factory = partial(
            _unpicklable_dataset,
            "unit_test_filter_spawn_loader",
            list(range(4)),
        )
        filtered = (
            FilterRule(
                name="even",
                factory=_mod_three_factory,
            )
            .apply(
                dataset_factory=dataset_factory,
                device="cpu",
            )
            .select_by("zero", "two")
        )

        loader = DataLoader(
            filtered,
            batch_size=None,
            num_workers=1,
            multiprocessing_context="spawn",
        )

        self.assertEqual([_value(sample) for sample in loader], [0, 2, 3])

    def test_result_and_filtered_dataset_attributes_are_read_only(self):
        _register_rows_source("unit_test_filter_readonly")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_readonly", [0])
            result = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            ).apply(dataset_factory=lambda: dataset, device="cpu")
            filtered = result.select_by("accept")

        with self.assertRaises(AttributeError):
            result.labels = ()
        with self.assertRaises(TypeError):
            result.counts["accept"] = 0
        with self.assertRaises(AttributeError):
            filtered.indices = ()
        with self.assertRaises(AttributeError):
            filtered.cache_path = Path("/tmp/changed")

    def test_filtered_dataset_repr_uses_count(self):
        _register_rows_source("unit_test_filter_repr")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_repr", [0, 1, 2])
            filtered = (
                FilterRule(
                    name="all",
                    factory=lambda: lambda sample: True,
                )
                .apply(dataset_factory=lambda: dataset, device="cpu")
                .select_by("accept")
            )

            text = repr(filtered)

        self.assertIn("count=3", text)
        self.assertNotIn("indices=", text)

    def test_rule_metadata_is_written_under_filter_cache_path(self):
        _register_rows_source("unit_test_filter_metadata")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            dataset = _dataset("unit_test_filter_metadata", [0, 1])
            rule = FilterRule(
                name="keep_v1",
                factory=lambda: lambda sample: True,
            )

            result = rule.apply(dataset_factory=lambda: dataset, device="cpu")
            metadata = json.loads(
                (result.cache_path / "rule.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (result.cache_path / "partitions.json").read_text(encoding="utf-8")
            )

        cache_root = result.cache_path.parents[1]
        current = json.loads((cache_root / "current.json").read_text(encoding="utf-8"))
        self.assertEqual(
            result.cache_path.parents[3], anydataset_home() / "cache" / "filters"
        )
        self.assertEqual(result.cache_path.parent.name, "generations")
        self.assertEqual(current["schema_version"], 1)
        self.assertEqual(current["generation"], result.cache_path.name)
        self.assertEqual(metadata["schema_version"], 5)
        self.assertEqual(
            metadata["base"]["identity"]["type"],
            "anydataset.dataset.abc.AnyDataset",
        )
        self.assertEqual(metadata["base"]["spec_id"], dataset.spec.id)
        self.assertEqual(metadata["base"]["identity"]["spec_id"], dataset.spec.id)
        self.assertEqual(metadata["base"]["sample_count"], 2)
        self.assertEqual(metadata["rule"]["name"], "keep_v1")
        self.assertEqual(set(metadata["rule"]), {"name"})
        self.assertEqual(manifest["partitions"][0]["label"], "accept")
        self.assertEqual(manifest["partitions"][0]["count"], 2)
        self.assertEqual(len(manifest["partitions"][0]["files"]), 1)

    def test_store_provenance_versions_filter_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store"
            sample = {
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.LONGCAT: torch.tensor([[1]])}
                )
            }
            DatasetWriter(
                path,
                dataset_id="toy",
                provenance={"input_id": "input-v1", "provider_id": "provider-v1"},
            ).write([sample])
            rule = FilterRule(
                name="has-longcat",
                factory=lambda: (
                    lambda value: (
                        AudioView.LONGCAT in value[Role.DEFAULT, Modality.AUDIO].views
                    )
                ),
            )
            first = rule.apply(
                dataset_factory=lambda: AnyDataset(
                    Spec(source="store", path=str(path))
                ),
                device="cpu",
            )

            manifest = read_store_json(path / "dataset.json")
            manifest["provenance"]["input_id"] = "input-v2"
            write_store_json(path / "dataset.json", manifest)
            second = rule.apply(
                dataset_factory=lambda: AnyDataset(
                    Spec(source="store", path=str(path))
                ),
                device="cpu",
            )

        self.assertNotEqual(first.cache_path, second.cache_path)

    def test_legacy_store_rejected_for_filter_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store"
            DatasetWriter(path, dataset_id="toy").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={AudioView.LONGCAT: torch.tensor([[1]])}
                        )
                    }
                ]
            )
            manifest = read_store_json(path / "dataset.json")
            manifest["schema_version"] = 2
            del manifest["provenance"]
            write_store_json(path / "dataset.json", manifest)
            rule = FilterRule(name="all", factory=lambda: lambda _sample: True)

            with self.assertRaisesRegex(ValueError, "schema_version 2 is legacy"):
                rule.apply(
                    dataset_factory=lambda: AnyDataset(
                        Spec(source="store", path=str(path))
                    ),
                    device="cpu",
                )

    def test_legacy_store_dataset_rejected_for_filter_cache_identity(self):
        from anydataset.store.reader import read_store_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store"
            DatasetWriter(path, dataset_id="toy").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={AudioView.LONGCAT: torch.tensor([[1]])}
                        )
                    }
                ]
            )
            manifest = read_store_json(path / "dataset.json")
            manifest["schema_version"] = 2
            del manifest["provenance"]
            write_store_json(path / "dataset.json", manifest)
            rule = FilterRule(name="all", factory=lambda: lambda _sample: True)

            with self.assertRaisesRegex(ValueError, "schema_version 2 is legacy"):
                AnyDataset(Spec(source="store", path=str(path))).prepare()

            allowed = read_store_dataset(path, legacy_policy="allow")
            with self.assertRaisesRegex(ValueError, "schema_version 2 is legacy"):
                rule.apply(dataset_factory=lambda: allowed, device="cpu")

    def test_store_view_selection_versions_filter_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "store"
            DatasetWriter(path, dataset_id="toy").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={
                                AudioView.WAVEFORM: (torch.tensor([[1.0]]), 16000),
                                AudioView.LONGCAT: torch.tensor([[1]]),
                            }
                        )
                    }
                ]
            )
            waveform = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            longcat = (Role.DEFAULT, Modality.AUDIO, AudioView.LONGCAT)
            rule = FilterRule(name="all", factory=lambda: lambda _sample: True)

            first = rule.apply(
                dataset_factory=lambda: AnyDataset.from_store(
                    path,
                    views=(waveform,),
                ),
                device="cpu",
            )
            second = rule.apply(
                dataset_factory=lambda: AnyDataset.from_store(
                    path,
                    views=(longcat,),
                ),
                device="cpu",
            )

        self.assertNotEqual(first.cache_path, second.cache_path)

    def test_rule_apply_writes_partition_shards(self):
        _register_rows_source("unit_test_filter_partition_shards")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            dataset = _dataset(
                "unit_test_filter_partition_shards",
                [0, 1, 2, 3, 4],
            )
            rule = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            )

            result = rule.apply(
                dataset_factory=lambda: dataset,
                device="cpu",
                commit_samples=2,
                max_shard_samples=2,
            )
            manifest = json.loads(
                (result.cache_path / "partitions.json").read_text(encoding="utf-8")
            )
            selected = result.select_by("accept")

        files = manifest["partitions"][0]["files"]
        self.assertEqual([file["count"] for file in files], [2, 2, 1])
        self.assertEqual(selected.indices, (0, 1, 2, 3, 4))

    def test_rule_apply_rejects_duplicate_partition_labels(self):
        _register_rows_source("unit_test_filter_duplicate_partition")
        dataset = _dataset("unit_test_filter_duplicate_partition", [0])
        rule = FilterRule("all", _true_factory)
        result = rule.apply(dataset_factory=lambda: dataset, device="cpu")
        path = result.cache_path / "partitions.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        duplicate = dict(manifest["partitions"][0])
        duplicate["count"] = 0
        manifest["partitions"].append(duplicate)
        path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Duplicate filter partition label"):
            rule.apply(dataset_factory=lambda: dataset, device="cpu")

    def test_rule_apply_filters_with_workers(self):
        with tempfile.TemporaryDirectory():
            dataset_factory = partial(
                _dataset,
                "unit_test_filter_workers",
                list(range(12)),
            )
            rule = FilterRule(
                name="mod_three",
                factory=_mod_three_factory,
            )

            result = rule.apply(
                dataset_factory=dataset_factory,
                device=("cpu:0", "cpu:1"),
                max_shard_samples=2,
            )
            selected = result.select_by("one", "two")

        self.assertEqual(result.counts, {"zero": 4, "one": 4, "two": 4})
        self.assertEqual(selected.indices, (1, 2, 4, 5, 7, 8, 10, 11))

    def test_rule_apply_workers_cover_tail_samples(self):
        with tempfile.TemporaryDirectory():
            dataset_factory = partial(
                _dataset,
                "unit_test_filter_worker_tail",
                list(range(5)),
            )
            result = FilterRule(
                name="all",
                factory=_true_factory,
            ).apply(dataset_factory=dataset_factory, device=("cpu:0", "cpu:1"))

        self.assertEqual(result.counts, {"accept": 5})
        self.assertEqual(result.select_by("accept").indices, (0, 1, 2, 3, 4))

    def test_rule_apply_single_device_loader_workers_cover_all_samples(self):
        with tempfile.TemporaryDirectory():
            dataset_factory = partial(
                _dataset,
                "unit_test_filter_loader_workers",
                list(range(7)),
            )
            result = FilterRule(
                name="mod_three",
                factory=_mod_three_factory,
            ).apply(
                dataset_factory=dataset_factory,
                device="cpu",
                num_workers=2,
                batch_size=2,
            )

        self.assertEqual(result.counts, {"zero": 3, "one": 2, "two": 2})
        self.assertEqual(result.select_by("one", "two").indices, (1, 2, 4, 5))

    def test_rule_apply_uses_predicate_call_batch_in_sample_order(self):
        with tempfile.TemporaryDirectory():
            dataset = _dataset(
                "unit_test_filter_predicate_batch",
                list(range(7)),
            )

            result = FilterRule(
                name="batch_mod_three",
                factory=_BatchModThree,
            ).apply(
                dataset_factory=lambda: dataset,
                device="cpu",
                batch_size=3,
            )

        self.assertEqual(result.counts, {"zero": 3, "one": 2, "two": 2})
        self.assertEqual(result.select_by("one").indices, (1, 4))
        self.assertEqual(result.select_by("two").indices, (2, 5))

    def test_rule_apply_rejects_wrong_predicate_batch_output_count(self):
        with tempfile.TemporaryDirectory():
            dataset = _dataset(
                "unit_test_filter_predicate_batch_count",
                [0, 1, 2],
            )

            with self.assertRaisesRegex(
                ValueError,
                "one output per input sample",
            ):
                FilterRule(
                    name="bad_batch_count",
                    factory=_ShortBatchFilter,
                ).apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                    batch_size=3,
                )

    def test_rule_apply_requires_ordered_predicate_batch_output(self):
        with tempfile.TemporaryDirectory():
            dataset = _dataset(
                "unit_test_filter_predicate_batch_order",
                [0, 1],
            )

            with self.assertRaisesRegex(TypeError, "ordered sequence"):
                FilterRule(
                    name="unordered_batch",
                    factory=_IterableBatchFilter,
                ).apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                    batch_size=2,
                )

    def test_rule_apply_parallel_loader_workers_cover_all_samples(self):
        with tempfile.TemporaryDirectory():
            dataset_factory = partial(
                _dataset,
                "unit_test_filter_parallel_loader_workers",
                list(range(10)),
            )
            result = FilterRule(
                name="mod_three",
                factory=_mod_three_factory,
            ).apply(
                dataset_factory=dataset_factory,
                device=("cpu:0", "cpu:1"),
                num_workers=2,
                batch_size=2,
            )

        self.assertEqual(result.counts, {"zero": 4, "one": 3, "two": 3})
        self.assertEqual(result.select_by("one", "two").indices, (1, 2, 4, 5, 7, 8))

    def test_rule_apply_remote_filter_with_fork_loader(self):
        with tempfile.TemporaryDirectory():
            dataset_factory = partial(
                _dataset,
                "unit_test_filter_remote_fork_loader",
                list(range(6)),
            )
            address = Path("/tmp") / f"anydataset-filter-{os.getpid()}-{id(self)}.sock"
            server = ProviderServer(
                address=address,
                provider_factory=_RemoteModThreeFactory(),
                device="cpu",
            )

            with server:
                result = FilterRule(
                    name="remote_mod_three",
                    factory=RemoteFilterFactory({"cpu": address}),
                ).apply(
                    dataset_factory=dataset_factory,
                    device="cpu",
                    num_workers=1,
                    batch_size=2,
                    runtime=Runtime(
                        server_start_method="spawn",
                    ),
                )

        self.assertEqual(result.counts, {"zero": 2, "one": 2, "two": 2})
        self.assertEqual(result.select_by("one", "two").indices, (1, 2, 4, 5))

    def test_rule_apply_writes_metrics_side_output(self):
        _register_rows_source("unit_test_filter_metrics")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_metrics", [0, 1, 2])
            rule = FilterRule(
                name="with_metrics",
                factory=_metric_factory,
            )

            result = rule.apply(
                dataset_factory=lambda: dataset,
                metrics=True,
                device="cpu",
                max_shard_samples=2,
            )
            rows = list(result.iter_metrics())
            manifest = json.loads(
                (result.metrics_path / "metrics.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.metrics_path, result.cache_path / "metrics")
        self.assertEqual(result.counts, {"accept": 2, "reject": 1})
        self.assertEqual(
            rows,
            [
                {
                    "index": 0,
                    "label": "accept",
                    "metrics": {"score": 0, "tags": ["even"]},
                },
                {
                    "index": 1,
                    "label": "reject",
                    "metrics": {"score": 1, "tags": ["odd"]},
                },
                {
                    "index": 2,
                    "label": "accept",
                    "metrics": {"score": 2, "tags": ["even"]},
                },
            ],
        )
        self.assertEqual(manifest["count"], 3)
        self.assertEqual([file["count"] for file in manifest["files"]], [2, 1])

    def test_rule_apply_reuses_metrics_cache(self):
        _register_rows_source("unit_test_filter_metrics_reuse")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            dataset = _dataset("unit_test_filter_metrics_reuse", [0, 1])
            FilterRule(
                name="same",
                factory=_metric_factory,
            ).apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")
            result = FilterRule(
                name="same",
                factory=lambda: (
                    lambda sample: (
                        calls.append(sample)
                        or FilterDecision(
                            label=False,
                            metrics={"score": -1},
                        )
                    )
                ),
            ).apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")
            rows = list(result.iter_metrics())

        self.assertEqual(calls, [])
        self.assertEqual([row["label"] for row in rows], ["accept", "reject"])

    def test_rule_apply_rebuilds_for_metrics_cache(self):
        _register_rows_source("unit_test_filter_metrics_rebuild")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            dataset = _dataset("unit_test_filter_metrics_rebuild", [0])
            FilterRule(
                name="same",
                factory=lambda: lambda sample: True,
            ).apply(dataset_factory=lambda: dataset, device="cpu")
            result = FilterRule(
                name="same",
                factory=lambda: (
                    lambda sample: calls.append(sample) or _metric_decision(sample)
                ),
            ).apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")
            rows = list(result.iter_metrics())

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            rows,
            [
                {
                    "index": 0,
                    "label": "accept",
                    "metrics": {"score": 0, "tags": ["even"]},
                }
            ],
        )

    def test_rule_apply_logs_cache_build_reason(self):
        _register_rows_source("unit_test_filter_cache_log")
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                dataset = _dataset("unit_test_filter_cache_log", [0])
                rule = FilterRule(
                    name="log_reason",
                    factory=lambda: lambda sample: True,
                )

                rule.apply(dataset_factory=lambda: dataset, device="cpu")
                log_text = _read_filter_log()
                events = _read_events()

        self.assertIn("building filter cache", log_text)
        self.assertIn("reason='current generation pointer is missing'", log_text)
        self.assertIn("rule='log_reason'", log_text)
        miss_event = [
            entry for entry in events if entry["event"] == "filter_cache_miss"
        ][0]
        self.assertEqual(miss_event["fields"]["rule"], "log_reason")
        self.assertEqual(miss_event["fields"]["sample_count"], 1)
        self.assertEqual(miss_event["fields"]["metrics"], False)
        self.assertEqual(
            miss_event["fields"]["reason"],
            "current generation pointer is missing",
        )

    def test_rule_apply_logs_metrics_rebuild_reason(self):
        _register_rows_source("unit_test_filter_metrics_log")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_metrics_log", [0])
            rule = FilterRule(
                name="log_metrics",
                factory=_metric_factory,
            )

            rule.apply(dataset_factory=lambda: dataset, device="cpu")
            rule.apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")
            log_text = _read_filter_log()

        self.assertIn("reason='metrics cache is missing or incomplete'", log_text)

    def test_rule_apply_reports_scan_and_writer_progress(self):
        _register_rows_source("unit_test_filter_progress")
        dataset = _dataset("unit_test_filter_progress", [0, 1, 2])
        stdout = io.StringIO()

        with (
            mock.patch("anydataset._runtime.progress._NON_INTERACTIVE_PROGRESS_INTERVAL", 0.0),
            redirect_stdout(stdout),
        ):
            FilterRule(
                name="progress",
                factory=lambda: lambda sample: True,
            ).apply(
                dataset_factory=lambda: dataset,
                device="cpu",
                commit_samples=2,
            )

        output = stdout.getvalue()
        self.assertIn("filter samples: 3 sample/3 (100.0%)", output)
        self.assertIn("scan=3", output)
        self.assertIn("writer=3", output)

    def test_rule_apply_requires_decisions_when_metrics_enabled(self):
        _register_rows_source("unit_test_filter_metrics_required")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_metrics_required", [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: True,
            )

            with self.assertRaises(TypeError):
                rule.apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")

    def test_filter_metrics_must_be_json_serializable(self):
        _register_rows_source("unit_test_filter_metrics_json")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_metrics_json", [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: (
                    lambda sample: FilterDecision(
                        label=True,
                        metrics={"score": math.nan},
                    )
                ),
            )

            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")

    def test_filter_metrics_keys_must_be_strings(self):
        _register_rows_source("unit_test_filter_metrics_keys")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_metrics_keys", [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: (
                    lambda sample: FilterDecision(
                        label=True,
                        metrics={1: "bad"},
                    )
                ),
            )

            with self.assertRaises(TypeError):
                rule.apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")

    def test_metrics_are_not_available_without_metrics_option(self):
        _register_rows_source("unit_test_filter_metrics_disabled")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_metrics_disabled", [0])
            result = FilterRule(
                name="disabled",
                factory=_metric_factory,
            ).apply(dataset_factory=lambda: dataset, device="cpu")

            with self.assertRaises(ValueError):
                list(result.iter_metrics())

    def test_rule_apply_writes_metrics_with_workers(self):
        with tempfile.TemporaryDirectory():
            dataset_factory = partial(
                _dataset,
                "unit_test_filter_metrics_workers",
                list(range(6)),
            )
            result = FilterRule(
                name="with_workers",
                factory=_metric_factory,
            ).apply(
                dataset_factory=dataset_factory,
                metrics=True,
                device=("cpu:0", "cpu:1"),
                max_shard_samples=2,
            )

            rows = list(result.iter_metrics())

        self.assertEqual([row["index"] for row in rows], list(range(6)))
        self.assertEqual([row["label"] for row in rows], ["accept", "reject"] * 3)

    def test_rule_apply_sets_ddp_environment_for_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            dataset_factory = partial(
                _dataset,
                "unit_test_filter_worker_env",
                [0, 1, 2, 3],
            )
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                result = FilterRule(
                    name="worker_env",
                    factory=_env_factory,
                ).apply(
                    dataset_factory=dataset_factory,
                    metrics=True,
                    device=("cpu:0", "cpu:1"),
                )
                log_texts = [
                    log.read_text(encoding="utf-8") for log in _filter_worker_logs(home)
                ]
            rows = list(result.iter_metrics())

        world_sizes = {row["metrics"]["world_size"] for row in rows}
        devices = {row["metrics"]["device"] for row in rows}
        ranks = {row["metrics"]["rank"] for row in rows}
        self.assertEqual(world_sizes, {"2"})
        self.assertEqual(devices, {"cpu:0", "cpu:1"})
        self.assertEqual(ranks, {"0", "1"})
        for text in log_texts:
            self.assertIn("starting shard", text)
            self.assertIn("finished shard", text)

    def test_rule_apply_workers_use_dataset_factory_not_dataset_pickle(self):
        with tempfile.TemporaryDirectory():
            dataset_factory = partial(
                _unpicklable_dataset,
                "unit_test_filter_unpicklable_dataset",
                list(range(4)),
            )

            result = FilterRule(
                name="mod_three",
                factory=_mod_three_factory,
            ).apply(dataset_factory=dataset_factory, device=("cpu:0", "cpu:1"))

        self.assertEqual(result.counts, {"zero": 2, "one": 1, "two": 1})

    def test_read_worker_message_rejects_unexpected_payload(self):
        output = mock.Mock()
        output.get.return_value = {"unexpected": True}
        with self.assertRaisesRegex(RuntimeError, "unexpected message"):
            read_worker_message(
                output,
                (),
                {},
                set(),
                rank=0,
                validate_modulo=False,
                worker_timeout=None,
                last_message=0.0,
            )

    def test_collect_ranges_parallel_orders_selected_indexes_by_position(self):
        with tempfile.TemporaryDirectory():
            dataset_factory = partial(
                _dataset,
                "unit_test_filter_parallel_selected",
                list(range(10)),
            )

            chunks = list(
                collect_ranges_parallel(
                    dataset_factory,
                    _metric_factory,
                    ("cpu:0", "cpu:1"),
                    True,
                    3,
                    sample_count=10,
                    sample_indexes=(2, 5, 9),
                    batch_size=2,
                    num_workers=0,
                    prefetch_factor=None,
                    runtime=Runtime(),
                    use_map_style_loader=True,
                )
            )

            indexes = [row.index for chunk in chunks for row in chunk.metrics]

            self.assertEqual(indexes, [2, 5, 9])

    def test_collect_ranges_parallel_cleans_workers_after_partial_start(self):
        context = mock.Mock()
        first = mock.Mock()
        first.is_alive.return_value = True
        second = mock.Mock()
        second.start.side_effect = RuntimeError("start failed")
        context.Process.side_effect = (first, second)
        dataset_factory = partial(
            _dataset,
            "unit_test_filter_partial_start",
            [0, 1],
        )

        with mock.patch(
            "anydataset.filter.runtime.collect.multiprocessing_context",
            return_value=context,
        ):
            chunks = collect_ranges_parallel(
                dataset_factory,
                _true_factory,
                ("cpu:0", "cpu:1"),
                False,
                1,
                sample_count=2,
                batch_size=1,
                num_workers=0,
                prefetch_factor=None,
                runtime=Runtime(),
                use_map_style_loader=True,
            )
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                next(iter(chunks))

        first.terminate.assert_called_once_with()
        first.join.assert_called_once_with()
        second.join.assert_not_called()

    def test_collect_ranges_parallel_keeps_compact_sample_indexes(self):
        context = mock.Mock()
        process = mock.Mock()
        process.start.side_effect = RuntimeError("stop before worker start")
        context.Process.return_value = process
        sample_indexes = range(20_000_000)
        dataset_factory = partial(
            _dataset,
            "unit_test_filter_compact_indexes",
            [0],
        )

        with mock.patch(
            "anydataset.filter.runtime.collect.multiprocessing_context",
            return_value=context,
        ):
            chunks = collect_ranges_parallel(
                dataset_factory,
                _true_factory,
                ("cpu",),
                False,
                1,
                sample_count=len(sample_indexes),
                sample_indexes=sample_indexes,
                batch_size=1,
                num_workers=0,
                prefetch_factor=None,
                runtime=Runtime(),
                use_map_style_loader=True,
            )
            with self.assertRaisesRegex(RuntimeError, "before worker start"):
                next(iter(chunks))

        config = context.Process.call_args.kwargs["args"][4]
        self.assertIs(config.sample_indexes, sample_indexes)

    def test_rule_apply_parallel_resume_skips_completed_predicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            calls = root / "parallel-calls.txt"
            marker = root / "parallel-failed.txt"
            dataset_factory = partial(
                _dataset,
                "unit_test_filter_parallel_resume",
                list(range(6)),
            )
            rule = FilterRule(
                name="parallel_resume_even",
                factory=_FailOnceFilterFactory(calls, marker, fail_value=2),
            )

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                with self.assertRaisesRegex(RuntimeError, "stop after first chunk"):
                    rule.apply(
                        dataset_factory=dataset_factory,
                        device=("cpu:0", "cpu:1"),
                        commit_samples=2,
                        write_workers=0,
                    )

                result = rule.apply(
                    dataset_factory=dataset_factory,
                    device=("cpu:0", "cpu:1"),
                    commit_samples=2,
                    write_workers=0,
                )

            self.assertEqual(result.counts, {"accept": 3, "reject": 3})
            self.assertEqual(result.select_by("accept").indices, (0, 2, 4))

    def test_rule_apply_parallel_worker_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            dataset_factory = partial(
                _dataset,
                "unit_test_filter_worker_timeout",
                list(range(2)),
            )
            rule = FilterRule(
                name="timeout",
                factory=_SlowFilterFactory(delay=5.0),
            )

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                with self.assertRaisesRegex(TimeoutError, "timed out"):
                    rule.apply(
                        dataset_factory=dataset_factory,
                        device=("cpu:0", "cpu:1"),
                        worker_timeout=0.1,
                    )

    def test_rule_apply_writes_empty_metrics_manifest(self):
        _register_rows_source("unit_test_filter_metrics_empty")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_metrics_empty", [])
            result = FilterRule(
                name="empty",
                factory=_metric_factory,
            ).apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")
            rows = list(result.iter_metrics())
            manifest = json.loads(
                (result.metrics_path / "metrics.json").read_text(encoding="utf-8")
            )

        self.assertEqual(rows, [])
        self.assertEqual(manifest["count"], 0)
        self.assertEqual(manifest["files"], [])

    def test_rule_apply_rejects_invalid_parallel_options(self):
        _register_rows_source("unit_test_filter_parallel_options")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_parallel_options", [0])
            rule = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            )

            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, device=())
            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, commit_samples=0)
            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, max_shard_samples=0)
            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, batch_size=0)
            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, num_workers=-1)
            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, prefetch_factor=0)
            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, write_workers=-1)
            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, write_prefetch=0)
            with self.assertRaises(TypeError):
                rule.apply(dataset_factory=lambda: dataset, input_id=1)
            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, input_id="")

    def test_filtered_dataset_rejects_empty_selection(self):
        _register_rows_source("unit_test_filter_requires_labels")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_requires_labels", [0])
            rule = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            )
            result = rule.apply(dataset_factory=lambda: dataset, device="cpu")

            with self.assertRaises(ValueError):
                result.select_by()

    def test_filter_rule_can_apply_to_filtered_dataset(self):
        _register_rows_source("unit_test_filter_chain")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            dataset = _dataset("unit_test_filter_chain", [0, 1, 2, 3, 4])
            first = (
                FilterRule(
                    name="gte_two",
                    factory=lambda: lambda sample: _value(sample) >= 2,
                )
                .apply(dataset_factory=lambda: dataset, device="cpu")
                .select_by("accept")
            )
            seen = []
            second_rule = FilterRule(
                name="even_after_gte_two",
                factory=lambda: lambda sample: _track_even(sample, seen),
            )

            result = second_rule.apply(
                dataset_factory=first.dataset_factory, device="cpu"
            )
            selected = result.select_by("accept")

        self.assertEqual(seen, [2, 3, 4])
        self.assertEqual(_values(selected), [2, 4])
        self.assertEqual(selected.indices, (2, 4))
        self.assertEqual(result.counts, {"accept": 2, "reject": 1})

    def test_chained_accept_filters_commute(self):
        _register_rows_source("unit_test_filter_commute")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_commute", list(range(12)))
            even = FilterRule(
                name="even",
                factory=lambda: lambda sample: _value(sample) % 2 == 0,
            )
            gte_six = FilterRule(
                name="gte_six",
                factory=lambda: lambda sample: _value(sample) >= 6,
            )

            even_then_gte = gte_six.apply(
                dataset_factory=even.apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                )
                .select_by("accept")
                .dataset_factory,
                device="cpu",
            ).select_by("accept")
            gte_then_even = even.apply(
                dataset_factory=gte_six.apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                )
                .select_by("accept")
                .dataset_factory,
                device="cpu",
            ).select_by("accept")

        self.assertEqual(_values(even_then_gte), [6, 8, 10])
        self.assertEqual(_values(gte_then_even), [6, 8, 10])
        self.assertEqual(even_then_gte.indices, gte_then_even.indices)

    def test_chained_filter_metrics_use_global_indices(self):
        _register_rows_source("unit_test_filter_chain_metrics")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            dataset = _dataset("unit_test_filter_chain_metrics", [0, 1, 2, 3, 4])
            first = (
                FilterRule(
                    name="gte_two",
                    factory=lambda: lambda sample: _value(sample) >= 2,
                )
                .apply(dataset_factory=lambda: dataset, device="cpu")
                .select_by("accept")
            )

            result = FilterRule(
                name="even_after_gte_two",
                factory=_metric_factory,
            ).apply(dataset_factory=first.dataset_factory, metrics=True, device="cpu")
            rows = list(result.iter_metrics())

        self.assertEqual(
            rows,
            [
                {
                    "index": 2,
                    "label": "accept",
                    "metrics": {"score": 2, "tags": ["even"]},
                },
                {
                    "index": 3,
                    "label": "reject",
                    "metrics": {"score": 3, "tags": ["odd"]},
                },
                {
                    "index": 4,
                    "label": "accept",
                    "metrics": {"score": 4, "tags": ["even"]},
                },
            ],
        )

    def test_chained_filter_cache_is_distinct_from_physical_filter_cache(self):
        _register_rows_source("unit_test_filter_chain_cache")
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir)
            dataset = _dataset("unit_test_filter_chain_cache", [0, 1, 2, 3])
            first = (
                FilterRule(
                    name="gte_two",
                    factory=lambda: lambda sample: _value(sample) >= 2,
                )
                .apply(dataset_factory=lambda: dataset, device="cpu")
                .select_by("accept")
            )
            second_rule = FilterRule(
                name="even",
                factory=lambda: lambda sample: _value(sample) % 2 == 0,
            )

            physical = second_rule.apply(dataset_factory=lambda: dataset, device="cpu")
            chained = second_rule.apply(
                dataset_factory=first.dataset_factory, device="cpu"
            )
            metadata = json.loads(
                (chained.cache_path / "rule.json").read_text(encoding="utf-8")
            )

        self.assertNotEqual(physical.cache_path, chained.cache_path)
        self.assertEqual(physical.counts, {"accept": 2, "reject": 2})
        self.assertEqual(chained.counts, {"accept": 1, "reject": 1})
        self.assertEqual(metadata["base"]["sample_count"], 2)
        self.assertEqual(metadata["base"]["view"]["kind"], "filtered")
        self.assertEqual(metadata["base"]["view"]["rule"], {"name": "gte_two"})
        self.assertEqual(metadata["base"]["view"]["labels"], ["accept"])
        self.assertEqual(
            metadata["base"]["view"]["generation"],
            first.cache_path.name,
        )
        self.assertEqual(metadata["base"]["view"]["view_schema_version"], 3)

    def test_chained_filter_cache_invalidates_on_upstream_generation_republish(self):
        _register_rows_source("unit_test_filter_chain_generation")
        dataset = _dataset("unit_test_filter_chain_generation", [0, 1, 2, 3])
        upstream_rule = FilterRule(
            name="gte_two",
            factory=lambda: lambda sample: _value(sample) >= 2,
        )
        downstream_rule = FilterRule(
            name="even",
            factory=lambda: lambda sample: _value(sample) % 2 == 0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": tmpdir}):
                upstream = upstream_rule.apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                ).select_by("accept")
                first = downstream_rule.apply(
                    dataset_factory=upstream.dataset_factory,
                    device="cpu",
                )
                first_path = first.cache_path
                first_generation = json.loads(
                    (first_path / "rule.json").read_text(encoding="utf-8")
                )["base"]["view"]["generation"]
                pointer = upstream.cache_path.parents[1] / "current.json"
                pointer.unlink()
                republished = upstream_rule.apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                ).select_by("accept")
                self.assertNotEqual(republished.cache_path.name, first_generation)
                self.assertEqual(len(republished), 2)

                second = downstream_rule.apply(
                    dataset_factory=republished.dataset_factory,
                    device="cpu",
                )
                second_meta = json.loads(
                    (second.cache_path / "rule.json").read_text(encoding="utf-8")
                )

        self.assertNotEqual(first_path, second.cache_path)
        self.assertEqual(second.counts, {"accept": 1, "reject": 1})
        self.assertEqual(
            second_meta["base"]["view"]["generation"],
            republished.cache_path.name,
        )
        self.assertNotEqual(
            second_meta["base"]["view"]["generation"],
            first_generation,
        )

    def test_filter_rule_exposes_name_contract_only(self):
        rule = FilterRule(
            name="same",
            factory=lambda: lambda sample: True,
        )

        self.assertEqual(rule.name, "same")
        self.assertFalse(hasattr(rule, "identity"))
        self.assertFalse(hasattr(rule, "id"))

    def test_filter_rule_equality_uses_identity_fields(self):
        first = FilterRule(
            name="same",
            factory=lambda: lambda sample: True,
            content_id="sha-a",
        )
        second = FilterRule(
            name="same",
            factory=lambda: lambda sample: False,
            content_id="sha-a",
        )
        third = FilterRule(
            name="same",
            factory=lambda: lambda sample: True,
            content_id="sha-b",
        )

        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))
        self.assertNotEqual(first, third)

    def test_filter_rule_attributes_are_read_only(self):
        rule = FilterRule(
            name="readonly",
            factory=lambda: lambda sample: True,
        )

        with self.assertRaises(AttributeError):
            rule.name = "changed"
        with self.assertRaises(AttributeError):
            rule.factory = lambda: lambda sample: False

    def test_filter_predicate_must_return_supported_label(self):
        _register_rows_source("unit_test_filter_predicate_type")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_predicate_type", [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: 1,
            )

            with self.assertRaises(TypeError):
                rule.apply(dataset_factory=lambda: dataset, device="cpu")

    def test_filter_label_must_not_be_empty(self):
        _register_rows_source("unit_test_filter_empty_label")
        with tempfile.TemporaryDirectory():
            dataset = _dataset("unit_test_filter_empty_label", [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: "",
            )

            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, device="cpu")


class Route(StrEnum):
    REVIEW = auto()


class _RowsSource:
    def prepare(self, spec: Spec, cache_path: Path):
        rows = [{"value": value} for value in spec.load_options["values"]]
        raw_log = spec.load_options.get("read_log")
        if raw_log is None:
            return rows
        return _ReadTrackingRows(rows, Path(str(raw_log)))


def _register_rows_source(name: str) -> None:
    if not source_exists(name):
        register_source(name, _RowsSource)


def _dataset(
    source: str,
    values: list[int],
    *,
    read_log: Path | None = None,
) -> AnyDataset:
    _register_rows_source(source)
    load_options: dict[str, object] = {"values": values}
    if read_log is not None:
        load_options["read_log"] = str(read_log)
    return AnyDataset(
        Spec(source=source, path="/tmp/rows", load_options=load_options),
        parse_fn=_parse,
    )


def _read_filter_log() -> str:
    logs = sorted(anydataset_home().glob("logs/*/filter.log"))
    if not logs:
        return ""
    return "\n".join(path.read_text(encoding="utf-8") for path in logs)


def _read_events() -> list[dict[str, object]]:
    logs = sorted(anydataset_home().glob("logs/*/events.jsonl"))
    rows: list[dict[str, object]] = []
    for path in logs:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    return rows


def _filter_worker_logs(home: Path) -> list[Path]:
    logs = sorted((home / "logs").glob("*/filter/part-*.log"))
    if len(logs) != 2:
        raise AssertionError(f"expected two filter worker logs, found: {logs}")
    return logs


def _close_filter_generation_lease(dataset, closed) -> None:
    dataset._cache._lease.close()
    closed.set()


def _use_filter_generation_after_release(dataset, ready, release) -> None:
    ready.set()
    if not release.wait(timeout=5):
        raise TimeoutError("filter generation lease was not released")
    if dataset.global_index(1) != 1:
        raise AssertionError("forked filter dataset returned the wrong index")


class _UnpicklableAnyDataset(AnyDataset):
    def __getstate__(self):
        raise TypeError("dataset instance must not be pickled")


class _LegacyReduce:
    def __init__(self, restore, args) -> None:
        self.restore = restore
        self.args = args

    def __reduce__(self):
        return self.restore, self.args


def _init_legacy_filter_rule(self, name, factory) -> None:
    self._name = name
    self._factory = factory


class _LazyIndex(Sequence[int]):
    def __init__(self, values: tuple[int, ...]) -> None:
        self._values = values
        self.iterated = False

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> int:
        return self._values[index]

    def __iter__(self) -> Iterator[int]:
        self.iterated = True
        return iter(self._values)


class _ReadTrackingRows:
    def __init__(self, rows: list[dict[str, int]], calls: Path) -> None:
        self.rows = rows
        self.calls = calls

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        with self.calls.open("a", encoding="utf-8") as file:
            file.write(f"{index}\n")
        return self.rows[index]


class _FailOnceFilter:
    def __init__(self, calls: Path, marker: Path, *, fail_value: int) -> None:
        self.calls = calls
        self.marker = marker
        self.fail_value = fail_value

    def __call__(self, sample):
        value = _value(sample)
        with self.calls.open("a", encoding="utf-8") as file:
            file.write(f"{value}\n")
        if value == self.fail_value and not self.marker.exists():
            self.marker.write_text("failed\n", encoding="utf-8")
            raise RuntimeError("stop after first chunk")
        return value % 2 == 0


class _FailOnceFilterFactory:
    def __init__(self, calls: Path, marker: Path, *, fail_value: int) -> None:
        self.calls = calls
        self.marker = marker
        self.fail_value = fail_value

    def __call__(self):
        return _FailOnceFilter(self.calls, self.marker, fail_value=self.fail_value)


class _SlowFilter:
    def __init__(self, *, delay: float) -> None:
        self.delay = delay

    def __call__(self, sample):
        time.sleep(self.delay)
        return True


class _SlowFilterFactory:
    def __init__(self, *, delay: float) -> None:
        self.delay = delay

    def __call__(self):
        return _SlowFilter(delay=self.delay)


class _BatchModThree:
    def __call__(self, sample):
        raise AssertionError("batched predicate should use call_batch()")

    def call_batch(self, samples):
        return tuple(_mod_three(sample) for sample in samples)


class _ShortBatchFilter:
    def __call__(self, sample):
        raise AssertionError("batched predicate should use call_batch()")

    def call_batch(self, samples):
        return [True] * (len(samples) - 1)


class _IterableBatchFilter:
    def __call__(self, sample):
        raise AssertionError("batched predicate should use call_batch()")

    def call_batch(self, samples):
        return (True for _sample in samples)


class _FailOnceMetricFilter(_FailOnceFilter):
    def __call__(self, sample):
        value = _value(sample)
        with self.calls.open("a", encoding="utf-8") as file:
            file.write(f"{value}\n")
        if value == self.fail_value and not self.marker.exists():
            self.marker.write_text("failed\n", encoding="utf-8")
            raise RuntimeError("stop after first chunk")
        return FilterDecision(label=value % 2 == 0, metrics={"score": value})


def _unpicklable_dataset(
    source: str,
    values: list[int],
) -> AnyDataset:
    _register_rows_source(source)
    return _UnpicklableAnyDataset(
        Spec(source=source, path="/tmp/rows", load_options={"values": values}),
        parse_fn=_parse,
    )


def _parse(row):
    value = row["value"]
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: value},
            meta={AudioMeta.LABEL: value},
        )
    }


def _route(sample):
    value = _value(sample)
    if value == 0:
        return True
    if value in {1, 2}:
        return Route.REVIEW
    return "reject"


def _route_factory():
    return _route


def _mod_three(sample):
    value = _value(sample)
    if value % 3 == 0:
        return "zero"
    if value % 3 == 1:
        return "one"
    return "two"


def _mod_three_factory():
    return _mod_three


class _RemoteModThreeFactory:
    def __call__(self, device: str):
        return _mod_three


def _true_decision(sample):
    return True


def _true_factory():
    return _true_decision


def _metric_decision(sample):
    value = _value(sample)
    label = value % 2 == 0
    tag = "even" if label else "odd"
    return FilterDecision(
        label=label,
        metrics={
            "score": value,
            "tags": [tag],
        },
    )


def _metric_factory():
    return _metric_decision


def _env_decision(sample):
    device = os.environ["ANYDATASET_FILTER_DEVICE"]
    return FilterDecision(
        label=device,
        metrics={
            "device": device,
            "local_rank": os.environ["LOCAL_RANK"],
            "rank": os.environ["RANK"],
            "world_size": os.environ["WORLD_SIZE"],
        },
    )


def _env_factory():
    return _env_decision


def _track_even(sample, seen):
    value = _value(sample)
    seen.append(value)
    return value % 2 == 0


def _value(sample):
    return sample[Role.DEFAULT, Modality.AUDIO].meta[AudioMeta.LABEL]


def _values(dataset):
    return [_value(sample) for sample in dataset]


if __name__ == "__main__":
    unittest.main()
