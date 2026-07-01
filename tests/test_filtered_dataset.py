from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from collections.abc import Iterator, Sequence
from enum import StrEnum, auto
from functools import partial
from pathlib import Path

import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioMeta,
    AudioView,
    anydataset_home,
    FilterDecision,
    FilteredDataset,
    FilterRule,
    Modality,
    ProviderServer,
    RemoteFilterFactory,
    Role,
    Runtime,
    Spec,
    TextItem,
    TextView,
    has_source,
    register_source,
)
from anydataset.store import DatasetWriter


class FilteredDatasetTest(unittest.TestCase):
    def test_rule_apply_partitions_bool_labels(self):
        _register_rows_source("unit_test_filter_bool")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
            root = Path(tmpdir)
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

    def test_select_unknown_label_returns_empty_dataset(self):
        _register_rows_source("unit_test_filter_unknown")
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
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
            root = Path(tmpdir)
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

    def test_rule_apply_resumes_from_completed_chunks(self):
        _register_rows_source("unit_test_filter_resume")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calls = root / "calls.txt"
            marker = root / "failed.txt"
            dataset = _dataset("unit_test_filter_resume", [0, 1, 2, 3])
            rule = FilterRule(
                name="resume_even",
                factory=lambda: _FailOnceFilter(calls, marker, fail_value=2),
            )

            with self.assertRaisesRegex(RuntimeError, "stop after first chunk"):
                rule.apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                    commit_samples=2,
                )

            self.assertEqual(calls.read_text(encoding="utf-8").splitlines(), ["0", "1", "2"])

            result = rule.apply(
                dataset_factory=lambda: dataset,
                device="cpu",
                commit_samples=2,
            )

            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["0", "1", "2", "2", "3"],
            )
            self.assertEqual(result.counts, {"accept": 2, "reject": 2})
            self.assertEqual(result.select_by("accept").indices, (0, 2))
            self.assertFalse(
                (result.cache_path.parent / f".{result.cache_path.name}.resume").exists()
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
            first = FilterRule(
                name="gte_two",
                factory=lambda: lambda sample: _value(sample) >= 2,
            ).apply(dataset_factory=lambda: dataset, device="cpu").select_by("accept")
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
        with tempfile.TemporaryDirectory() as tmpdir:
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
            root = Path(tmpdir)
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
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_same_name", [0])
            first_rule = FilterRule(
                name="same_name",
                factory=lambda: lambda sample: True,
            )
            first_result = first_rule.apply(dataset_factory=lambda: dataset, device="cpu")
            second_rule = FilterRule(
                name="same_name",
                factory=lambda: lambda sample: calls.append(sample) or False,
            )

            second_result = second_rule.apply(dataset_factory=lambda: dataset, device="cpu")

        self.assertEqual(first_result.cache_path, second_result.cache_path)
        self.assertEqual(calls, [])
        self.assertEqual(_values(second_result.select_by("accept")), [0])

    def test_filtered_dataset_selects_from_filter_cache(self):
        _register_rows_source("unit_test_filter_direct_select")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_direct_select", [0, 1, 2])
            rule = FilterRule(
                name="even",
                factory=lambda: lambda sample: calls.append(_value(sample)) or _value(sample) % 2 == 0,
            )

            result = rule.apply(dataset_factory=lambda: dataset, device="cpu")
            filtered = result.select_by("accept")

        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(_values(filtered), [0, 2])
        self.assertEqual(filtered.indices, (0, 2))

    def test_filtered_dataset_constructor_matches_apply_select_by(self):
        _register_rows_source("unit_test_filter_direct_constructor")
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_direct_reject", [0])

            with self.assertRaises(TypeError):
                FilteredDataset(1, _true_factory, dataset_factory=lambda: dataset)

    def test_filtered_dataset_constructor_rejects_unknown_apply_kwargs(self):
        _register_rows_source("unit_test_filter_direct_bad_option")
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_shards", [0, 1, 2, 3, 4])
            filtered = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            ).apply(dataset_factory=lambda: dataset, device="cpu").select_by("accept")

            shard = [_value(sample) for sample in filtered.iter_shard(2, 1)]

        self.assertEqual(shard, [1, 3])

    def test_filtered_dataset_indexed_shards_keep_physical_indices(self):
        _register_rows_source("unit_test_filter_indexed_shards")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_indexed_shards", [0, 1, 2, 3, 4])
            filtered = FilterRule(
                name="even",
                factory=lambda: lambda sample: _value(sample) % 2 == 0,
            ).apply(dataset_factory=lambda: dataset, device="cpu").select_by("accept")

            shard = [
                (index, filtered.global_index(index), _value(sample))
                for index, sample in filtered.iter_indexed_shard(2, 1)
            ]

        self.assertEqual(shard, [(1, 2, 2)])

    def test_result_and_filtered_dataset_attributes_are_read_only(self):
        _register_rows_source("unit_test_filter_readonly")
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_repr", [0, 1, 2])
            filtered = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            ).apply(dataset_factory=lambda: dataset, device="cpu").select_by("accept")

            text = repr(filtered)

        self.assertIn("count=3", text)
        self.assertNotIn("indices=", text)

    def test_rule_metadata_is_written_under_filter_cache_path(self):
        _register_rows_source("unit_test_filter_metadata")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_metadata", [0, 1])
            rule = FilterRule(
                name="keep_v1",
                factory=lambda: lambda sample: True,
            )

            result = rule.apply(dataset_factory=lambda: dataset, device="cpu")
            metadata = json.loads((result.cache_path / "rule.json").read_text(encoding="utf-8"))
            manifest = json.loads((result.cache_path / "partitions.json").read_text(encoding="utf-8"))

        self.assertEqual(result.cache_path.parents[1], anydataset_home() / "cache" / "filters")
        self.assertEqual(metadata["schema_version"], 5)
        self.assertEqual(metadata["base"]["identity"]["type"], "anydataset.dataset.abc.AnyDataset")
        self.assertEqual(metadata["base"]["spec_id"], dataset.spec.id)
        self.assertEqual(metadata["base"]["identity"]["spec_id"], dataset.spec.id)
        self.assertEqual(metadata["base"]["sample_count"], 2)
        self.assertEqual(metadata["rule"]["name"], "keep_v1")
        self.assertEqual(set(metadata["rule"]), {"name"})
        self.assertEqual(manifest["partitions"][0]["label"], "accept")
        self.assertEqual(manifest["partitions"][0]["count"], 2)
        self.assertEqual(len(manifest["partitions"][0]["files"]), 1)

    def test_rule_apply_writes_partition_shards(self):
        _register_rows_source("unit_test_filter_partition_shards")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset(
                "unit_test_filter_partition_shards",
                [0, 1, 2, 3, 4],
            )
            rule = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            )

            result = rule.apply(dataset_factory=lambda: dataset, device="cpu", commit_samples=2, max_shard_samples=2)
            manifest = json.loads((result.cache_path / "partitions.json").read_text(encoding="utf-8"))
            selected = result.select_by("accept")

        files = manifest["partitions"][0]["files"]
        self.assertEqual([file["count"] for file in files], [2, 2, 1])
        self.assertEqual(selected.indices, (0, 1, 2, 3, 4))

    def test_rule_apply_filters_with_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
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

    def test_rule_apply_parallel_loader_workers_cover_all_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics", [0, 1, 2])
            rule = FilterRule(
                name="with_metrics",
                factory=_metric_factory,
            )

            result = rule.apply(dataset_factory=lambda: dataset, metrics=True, device="cpu", max_shard_samples=2)
            rows = list(result.iter_metrics())
            manifest = json.loads(
                (result.metrics_path / "metrics.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.metrics_path, result.cache_path / "metrics")
        self.assertEqual(result.counts, {"accept": 2, "reject": 1})
        self.assertEqual(
            rows,
            [
                {"index": 0, "label": "accept", "metrics": {"score": 0, "tags": ["even"]}},
                {"index": 1, "label": "reject", "metrics": {"score": 1, "tags": ["odd"]}},
                {"index": 2, "label": "accept", "metrics": {"score": 2, "tags": ["even"]}},
            ],
        )
        self.assertEqual(manifest["count"], 3)
        self.assertEqual([file["count"] for file in manifest["files"]], [2, 1])

    def test_rule_apply_reuses_metrics_cache(self):
        _register_rows_source("unit_test_filter_metrics_reuse")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_metrics_reuse", [0, 1])
            FilterRule(
                name="same",
                factory=_metric_factory,
            ).apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")
            result = FilterRule(
                name="same",
                factory=lambda: lambda sample: calls.append(sample) or FilterDecision(
                    label=False,
                    metrics={"score": -1},
                ),
            ).apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")
            rows = list(result.iter_metrics())

        self.assertEqual(calls, [])
        self.assertEqual([row["label"] for row in rows], ["accept", "reject"])

    def test_rule_apply_rebuilds_for_metrics_cache(self):
        _register_rows_source("unit_test_filter_metrics_rebuild")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_metrics_rebuild", [0])
            FilterRule(
                name="same",
                factory=lambda: lambda sample: True,
            ).apply(dataset_factory=lambda: dataset, device="cpu")
            result = FilterRule(
                name="same",
                factory=lambda: lambda sample: calls.append(sample) or _metric_decision(sample),
            ).apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")
            rows = list(result.iter_metrics())

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            rows,
            [{"index": 0, "label": "accept", "metrics": {"score": 0, "tags": ["even"]}}],
        )

    def test_rule_apply_logs_cache_build_reason(self):
        _register_rows_source("unit_test_filter_cache_log")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_cache_log", [0])
            rule = FilterRule(
                name="log_reason",
                factory=lambda: lambda sample: True,
            )

            rule.apply(dataset_factory=lambda: dataset, device="cpu")
            log_text = _read_filter_log()

        self.assertIn("building filter cache", log_text)
        self.assertIn("reason='ready marker is missing'", log_text)
        self.assertIn("rule='log_reason'", log_text)

    def test_rule_apply_logs_metrics_rebuild_reason(self):
        _register_rows_source("unit_test_filter_metrics_log")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics_log", [0])
            rule = FilterRule(
                name="log_metrics",
                factory=_metric_factory,
            )

            rule.apply(dataset_factory=lambda: dataset, device="cpu")
            rule.apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")
            log_text = _read_filter_log()

        self.assertIn("reason='metrics cache is missing or incomplete'", log_text)

    def test_rule_apply_requires_decisions_when_metrics_enabled(self):
        _register_rows_source("unit_test_filter_metrics_required")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics_required", [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: True,
            )

            with self.assertRaises(TypeError):
                rule.apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")

    def test_filter_metrics_must_be_json_serializable(self):
        _register_rows_source("unit_test_filter_metrics_json")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics_json", [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: FilterDecision(
                    label=True,
                    metrics={"score": math.nan},
                ),
            )

            with self.assertRaises(ValueError):
                rule.apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")

    def test_filter_metrics_keys_must_be_strings(self):
        _register_rows_source("unit_test_filter_metrics_keys")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics_keys", [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: FilterDecision(
                    label=True,
                    metrics={1: "bad"},
                ),
            )

            with self.assertRaises(TypeError):
                rule.apply(dataset_factory=lambda: dataset, metrics=True, device="cpu")

    def test_metrics_are_not_available_without_metrics_option(self):
        _register_rows_source("unit_test_filter_metrics_disabled")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics_disabled", [0])
            result = FilterRule(
                name="disabled",
                factory=_metric_factory,
            ).apply(dataset_factory=lambda: dataset, device="cpu")

            with self.assertRaises(ValueError):
                list(result.iter_metrics())

    def test_rule_apply_writes_metrics_with_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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
            dataset_factory = partial(
                _dataset,
                "unit_test_filter_worker_env",
                [0, 1, 2, 3],
            )
            result = FilterRule(
                name="worker_env",
                factory=_env_factory,
            ).apply(
                dataset_factory=dataset_factory,
                metrics=True,
                device=("cpu:0", "cpu:1"),
            )
            rows = list(result.iter_metrics())

        world_sizes = {row["metrics"]["world_size"] for row in rows}
        devices = {row["metrics"]["device"] for row in rows}
        ranks = {row["metrics"]["rank"] for row in rows}
        self.assertEqual(world_sizes, {"2"})
        self.assertEqual(devices, {"cpu:0", "cpu:1"})
        self.assertEqual(ranks, {"0", "1"})

    def test_rule_apply_workers_use_dataset_factory_not_dataset_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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

    def test_rule_apply_writes_empty_metrics_manifest(self):
        _register_rows_source("unit_test_filter_metrics_empty")
        with tempfile.TemporaryDirectory() as tmpdir:
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
        with tempfile.TemporaryDirectory() as tmpdir:
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

    def test_filtered_dataset_rejects_empty_selection(self):
        _register_rows_source("unit_test_filter_requires_labels")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_requires_labels", [0])
            rule = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            )
            result = rule.apply(dataset_factory=lambda: dataset, device="cpu")

            with self.assertRaises(ValueError):
                result.select_by()

    def test_filtered_dataset_reads_store_merge_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            delta = root / "delta"
            waveform = torch.tensor([[1.0, 2.0]])
            DatasetWriter(delta, dataset_id="toy", split="train").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={
                                AudioView.LONGCAT: {
                                    "semantic_codes": torch.tensor([1, 2]),
                                }
                            },
                            meta={AudioMeta.LABEL: "speech"},
                        )
                    }
                ]
            )
            source = [
                {
                    (Role.DEFAULT, Modality.AUDIO): AudioItem(
                        views={AudioView.WAVEFORM: (waveform, 4)},
                        meta={AudioMeta.LABEL: "speech"},
                    ),
                    (Role.DEFAULT, Modality.TEXT): TextItem(
                        views={TextView.TEXT: "hello"},
                    ),
                }
            ]
            merged = AnyDataset(
                f"store://{delta}:train",
            ).merge(source)
            rule = FilterRule(
                name="has_longcat",
                factory=lambda: lambda sample: AudioView.LONGCAT
                in sample[Role.DEFAULT, Modality.AUDIO].views,
            )

            result = rule.apply(
                dataset_factory=lambda: merged,
                device="cpu",
            )
            filtered = result.select_by("accept")
            sample = filtered[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(set(audio.views), {AudioView.WAVEFORM, AudioView.LONGCAT})
        self.assertEqual(sample[Role.DEFAULT, Modality.TEXT].views[TextView.TEXT], "hello")

    def test_merge_filter_identity_is_order_and_grouping_independent(self):
        _register_rows_source("unit_test_filter_merge_identity")
        base = _dataset("unit_test_filter_merge_identity", [0])
        first = [{"a": 1}]
        second = [{"b": 2}]
        rule = FilterRule(
            name="all",
            factory=lambda: lambda sample: True,
        )

        left = base.merge(first).merge(second)
        right = base.merge(second).merge(first)
        left_grouped = rule.apply(dataset_factory=lambda: left, device="cpu")
        right_grouped = rule.apply(dataset_factory=lambda: right, device="cpu")

        self.assertEqual(left_grouped.cache_path, right_grouped.cache_path)

    def test_filter_rule_can_apply_to_filtered_dataset(self):
        _register_rows_source("unit_test_filter_chain")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_chain", [0, 1, 2, 3, 4])
            first = FilterRule(
                name="gte_two",
                factory=lambda: lambda sample: _value(sample) >= 2,
            ).apply(dataset_factory=lambda: dataset, device="cpu").select_by("accept")
            seen = []
            second_rule = FilterRule(
                name="even_after_gte_two",
                factory=lambda: lambda sample: _track_even(sample, seen),
            )

            result = second_rule.apply(dataset_factory=first.dataset_factory, device="cpu")
            selected = result.select_by("accept")

        self.assertEqual(seen, [2, 3, 4])
        self.assertEqual(_values(selected), [2, 4])
        self.assertEqual(selected.indices, (2, 4))
        self.assertEqual(result.counts, {"accept": 2, "reject": 1})

    def test_chained_accept_filters_commute(self):
        _register_rows_source("unit_test_filter_commute")
        with tempfile.TemporaryDirectory() as tmpdir:
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
                ).select_by("accept").dataset_factory,
                device="cpu",
            ).select_by("accept")
            gte_then_even = even.apply(
                dataset_factory=gte_six.apply(
                    dataset_factory=lambda: dataset,
                    device="cpu",
                ).select_by("accept").dataset_factory,
                device="cpu",
            ).select_by("accept")

        self.assertEqual(_values(even_then_gte), [6, 8, 10])
        self.assertEqual(_values(gte_then_even), [6, 8, 10])
        self.assertEqual(even_then_gte.indices, gte_then_even.indices)

    def test_chained_filter_metrics_use_global_indices(self):
        _register_rows_source("unit_test_filter_chain_metrics")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_chain_metrics", [0, 1, 2, 3, 4])
            first = FilterRule(
                name="gte_two",
                factory=lambda: lambda sample: _value(sample) >= 2,
            ).apply(dataset_factory=lambda: dataset, device="cpu").select_by("accept")

            result = FilterRule(
                name="even_after_gte_two",
                factory=_metric_factory,
            ).apply(dataset_factory=first.dataset_factory, metrics=True, device="cpu")
            rows = list(result.iter_metrics())

        self.assertEqual(
            rows,
            [
                {"index": 2, "label": "accept", "metrics": {"score": 2, "tags": ["even"]}},
                {"index": 3, "label": "reject", "metrics": {"score": 3, "tags": ["odd"]}},
                {"index": 4, "label": "accept", "metrics": {"score": 4, "tags": ["even"]}},
            ],
        )

    def test_chained_filter_cache_is_distinct_from_physical_filter_cache(self):
        _register_rows_source("unit_test_filter_chain_cache")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_chain_cache", [0, 1, 2, 3])
            first = FilterRule(
                name="gte_two",
                factory=lambda: lambda sample: _value(sample) >= 2,
            ).apply(dataset_factory=lambda: dataset, device="cpu").select_by("accept")
            second_rule = FilterRule(
                name="even",
                factory=lambda: lambda sample: _value(sample) % 2 == 0,
            )

            physical = second_rule.apply(dataset_factory=lambda: dataset, device="cpu")
            chained = second_rule.apply(dataset_factory=first.dataset_factory, device="cpu")
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

    def test_filter_rule_exposes_name_contract_only(self):
        rule = FilterRule(
            name="same",
            factory=lambda: lambda sample: True,
        )

        self.assertEqual(rule.name, "same")
        self.assertFalse(hasattr(rule, "identity"))
        self.assertFalse(hasattr(rule, "id"))

    def test_filter_rule_equality_uses_name(self):
        first = FilterRule(
            name="same",
            factory=lambda: lambda sample: True,
        )
        second = FilterRule(
            name="same",
            factory=lambda: lambda sample: False,
        )

        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))

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
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_predicate_type", [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: 1,
            )

            with self.assertRaises(TypeError):
                rule.apply(dataset_factory=lambda: dataset, device="cpu")

    def test_filter_label_must_not_be_empty(self):
        _register_rows_source("unit_test_filter_empty_label")
        with tempfile.TemporaryDirectory() as tmpdir:
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
        return [{"value": value} for value in spec.load_options["values"]]


def _register_rows_source(name: str) -> None:
    if not has_source(name):
        register_source(name, _RowsSource)


def _dataset(source: str, values: list[int]) -> AnyDataset:
    _register_rows_source(source)
    return AnyDataset(
        Spec(source=source, path="/tmp/rows", load_options={"values": values}),
        parse_fn=_parse,
    )


def _read_filter_log() -> str:
    logs = sorted(anydataset_home().glob("logs/*/filter.log"))
    if not logs:
        return ""
    return "\n".join(path.read_text(encoding="utf-8") for path in logs)


class _UnpicklableAnyDataset(AnyDataset):
    def __getstate__(self):
        raise TypeError("dataset instance must not be pickled")


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
