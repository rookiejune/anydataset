from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from enum import StrEnum, auto
from pathlib import Path

import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioMeta,
    AudioView,
    FilterDecision,
    FilteredDataset,
    FilterRule,
    Modality,
    Role,
    Spec,
    TextItem,
    TextView,
    register_source,
)
from anydataset.store import DatasetWriter


class FilteredDatasetTest(unittest.TestCase):
    def test_rule_apply_partitions_bool_labels(self):
        _register_rows_source("unit_test_filter_bool")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_bool", root, [0, 1, 2, 3])
            rule = FilterRule(
                name="even",
                factory=lambda: lambda sample: _value(sample) % 2 == 0,
            )

            result = rule.apply(dataset, device="cpu")
            accepted = result.select("accept")
            rejected = result.select("reject")

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
            dataset = _dataset("unit_test_filter_labels", root, [0, 1, 2, 3])
            rule = FilterRule(
                name="route",
                factory=_route_factory,
            )

            result = rule.apply(dataset, device="cpu")
            selected = result.select(Route.REVIEW, "reject")

        self.assertEqual(result.labels, ("accept", "review", "reject"))
        self.assertEqual(result.counts, {"accept": 1, "review": 2, "reject": 1})
        self.assertEqual(selected.labels, ("review", "reject"))
        self.assertEqual(_values(selected), [1, 2, 3])
        self.assertEqual(selected.indices, (1, 2, 3))

    def test_select_unknown_label_returns_empty_dataset(self):
        _register_rows_source("unit_test_filter_unknown")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_unknown", Path(tmpdir), [0])
            result = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            ).apply(dataset, device="cpu")

            selected = result.select("review")

        self.assertEqual(len(selected), 0)
        self.assertEqual(selected.indices, ())

    def test_select_deduplicates_labels(self):
        _register_rows_source("unit_test_filter_deduplicate")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_deduplicate", Path(tmpdir), [0, 1])
            result = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            ).apply(dataset, device="cpu")

            selected = result.select(True, "accept")

        self.assertEqual(selected.labels, ("accept",))
        self.assertEqual(selected.indices, (0, 1))

    def test_rule_predicate_receives_full_sample(self):
        _register_rows_source("unit_test_filter_full_sample")
        seen = []
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_full_sample", Path(tmpdir), [0])
            rule = FilterRule(
                name="full",
                factory=lambda: lambda sample: seen.append(sample) or True,
            )

            rule.apply(dataset, device="cpu")

        sample = seen[0]
        audio = sample[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(audio.views, {AudioView.WAVEFORM: 0})
        self.assertEqual(audio.meta, {AudioMeta.LABEL: 0})

    def test_rule_apply_reuses_ready_cache(self):
        _register_rows_source("unit_test_filter_reuses")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_reuses", root, [0, 1, 2, 3])
            first_rule = FilterRule(
                name="gte_two",
                factory=lambda: lambda sample: _value(sample) >= 2,
            )
            first_rule.apply(dataset, device="cpu")
            second_rule = FilterRule(
                name="gte_two",
                factory=lambda: lambda sample: calls.append(sample) or False,
            )

            result = second_rule.apply(dataset, device="cpu")

        self.assertEqual(_values(result.select("accept")), [2, 3])
        self.assertEqual(calls, [])

    def test_rule_apply_creates_predicate_from_factory(self):
        _register_rows_source("unit_test_filter_factory")
        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_factory", Path(tmpdir), [0, 1])

            def factory():
                events.append("factory")
                return lambda sample: events.append(_value(sample)) or True

            rule = FilterRule(name="factory", factory=factory)
            rule.apply(dataset, device="cpu")
            rule.apply(dataset, device="cpu")

        self.assertEqual(events, ["factory", 0, 1])

    def test_rule_apply_rebuilds_when_base_count_changes(self):
        _register_rows_source("unit_test_filter_rebuilds")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = _dataset("unit_test_filter_rebuilds", root, [0, 1, 2])
            rule = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            )
            rule.apply(first, device="cpu")
            second = _dataset("unit_test_filter_rebuilds", root, [0, 1, 2, 3])

            result = rule.apply(second, device="cpu")

        self.assertEqual(_values(result.select("accept")), [0, 1, 2, 3])

    def test_rule_apply_reuses_same_name_cache(self):
        _register_rows_source("unit_test_filter_same_name")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_same_name", root, [0])
            first_rule = FilterRule(
                name="same_name",
                factory=lambda: lambda sample: True,
            )
            first_result = first_rule.apply(dataset, device="cpu")
            second_rule = FilterRule(
                name="same_name",
                factory=lambda: lambda sample: calls.append(sample) or False,
            )

            second_result = second_rule.apply(dataset, device="cpu")

        self.assertEqual(first_result.cache_path, second_result.cache_path)
        self.assertEqual(calls, [])
        self.assertEqual(_values(second_result.select("accept")), [0])

    def test_filtered_dataset_builds_cache_when_missing(self):
        _register_rows_source("unit_test_filter_direct_build")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_direct_build", Path(tmpdir), [0, 1, 2])
            rule = FilterRule(
                name="even",
                factory=lambda: lambda sample: calls.append(_value(sample)) or _value(sample) % 2 == 0,
            )

            filtered = FilteredDataset(dataset, rule, labels="accept", device="cpu")

        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(_values(filtered), [0, 2])
        self.assertEqual(filtered.indices, (0, 2))

    def test_filtered_dataset_reuses_ready_cache_by_rule_name(self):
        _register_rows_source("unit_test_filter_direct_reuse")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_direct_reuse", root, [0, 1, 2])
            FilterRule(
                name="same",
                factory=lambda: lambda sample: _value(sample) >= 1,
            ).apply(dataset, device="cpu")
            filtered = FilteredDataset(
                dataset,
                FilterRule(
                    name="same",
                    factory=lambda: lambda sample: calls.append(sample) or False,
                ),
                labels="accept",
                device="cpu",
            )

        self.assertEqual(calls, [])
        self.assertEqual(_values(filtered), [1, 2])

    def test_filtered_dataset_shards_after_remap(self):
        _register_rows_source("unit_test_filter_shards")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_shards", root, [0, 1, 2, 3, 4])
            filtered = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            ).apply(dataset, device="cpu").select("accept")

            shard = [_value(sample) for sample in filtered.iter_shard(2, 1)]

        self.assertEqual(shard, [1, 3])

    def test_result_and_filtered_dataset_attributes_are_read_only(self):
        _register_rows_source("unit_test_filter_readonly")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_readonly", Path(tmpdir), [0])
            result = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            ).apply(dataset, device="cpu")
            filtered = result.select("accept")

        with self.assertRaises(AttributeError):
            result.labels = ()
        with self.assertRaises(TypeError):
            result.partitions["accept"] = ()
        with self.assertRaises(AttributeError):
            filtered.indices = ()
        with self.assertRaises(AttributeError):
            filtered.cache_path = Path("/tmp/changed")

    def test_filtered_dataset_repr_uses_count(self):
        _register_rows_source("unit_test_filter_repr")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_repr", Path(tmpdir), [0, 1, 2])
            filtered = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            ).apply(dataset, device="cpu").select("accept")

            text = repr(filtered)

        self.assertIn("count=3", text)
        self.assertNotIn("indices=", text)

    def test_rule_metadata_is_written_under_physical_cache_path(self):
        _register_rows_source("unit_test_filter_metadata")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_metadata", root, [0, 1])
            rule = FilterRule(
                name="keep_v1",
                factory=lambda: lambda sample: True,
            )

            result = rule.apply(dataset, device="cpu")
            metadata = json.loads((result.cache_path / "rule.json").read_text(encoding="utf-8"))
            manifest = json.loads((result.cache_path / "partitions.json").read_text(encoding="utf-8"))

        self.assertEqual(result.cache_path.parent, root / dataset.spec.cache_relpath / "filters")
        self.assertEqual(metadata["schema_version"], 4)
        self.assertEqual(metadata["base"]["spec_id"], dataset.spec.id)
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
                root,
                [0, 1, 2, 3, 4],
            )
            rule = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            )

            result = rule.apply(dataset, device="cpu", commit_samples=2, max_shard_samples=2)
            manifest = json.loads((result.cache_path / "partitions.json").read_text(encoding="utf-8"))
            selected = result.select("accept")

        files = manifest["partitions"][0]["files"]
        self.assertEqual([file["count"] for file in files], [2, 2, 1])
        self.assertEqual(selected.indices, (0, 1, 2, 3, 4))

    def test_rule_apply_filters_with_workers(self):
        _register_rows_source("unit_test_filter_workers")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset(
                "unit_test_filter_workers",
                Path(tmpdir),
                list(range(12)),
            )
            rule = FilterRule(
                name="mod_three",
                factory=_mod_three_factory,
            )

            result = rule.apply(dataset, device=("cpu:0", "cpu:1"), max_shard_samples=2)
            selected = result.select("one", "two")

        self.assertEqual(result.counts, {"zero": 4, "one": 4, "two": 4})
        self.assertEqual(selected.indices, (1, 2, 4, 5, 7, 8, 10, 11))

    def test_rule_apply_writes_metrics_side_output(self):
        _register_rows_source("unit_test_filter_metrics")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics", Path(tmpdir), [0, 1, 2])
            rule = FilterRule(
                name="with_metrics",
                factory=_metric_factory,
            )

            result = rule.apply(dataset, metrics=True, device="cpu", max_shard_samples=2)
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
            dataset = _dataset("unit_test_filter_metrics_reuse", root, [0, 1])
            FilterRule(
                name="same",
                factory=_metric_factory,
            ).apply(dataset, metrics=True, device="cpu")
            result = FilterRule(
                name="same",
                factory=lambda: lambda sample: calls.append(sample) or FilterDecision(
                    label=False,
                    metrics={"score": -1},
                ),
            ).apply(dataset, metrics=True, device="cpu")
            rows = list(result.iter_metrics())

        self.assertEqual(calls, [])
        self.assertEqual([row["label"] for row in rows], ["accept", "reject"])

    def test_rule_apply_rebuilds_for_metrics_cache(self):
        _register_rows_source("unit_test_filter_metrics_rebuild")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_metrics_rebuild", root, [0])
            FilterRule(
                name="same",
                factory=lambda: lambda sample: True,
            ).apply(dataset, device="cpu")
            result = FilterRule(
                name="same",
                factory=lambda: lambda sample: calls.append(sample) or _metric_decision(sample),
            ).apply(dataset, metrics=True, device="cpu")
            rows = list(result.iter_metrics())

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            rows,
            [{"index": 0, "label": "accept", "metrics": {"score": 0, "tags": ["even"]}}],
        )

    def test_rule_apply_requires_decisions_when_metrics_enabled(self):
        _register_rows_source("unit_test_filter_metrics_required")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics_required", Path(tmpdir), [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: True,
            )

            with self.assertRaises(TypeError):
                rule.apply(dataset, metrics=True, device="cpu")

    def test_filter_metrics_must_be_json_serializable(self):
        _register_rows_source("unit_test_filter_metrics_json")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics_json", Path(tmpdir), [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: FilterDecision(
                    label=True,
                    metrics={"score": math.nan},
                ),
            )

            with self.assertRaises(ValueError):
                rule.apply(dataset, metrics=True, device="cpu")

    def test_filter_metrics_keys_must_be_strings(self):
        _register_rows_source("unit_test_filter_metrics_keys")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics_keys", Path(tmpdir), [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: FilterDecision(
                    label=True,
                    metrics={1: "bad"},
                ),
            )

            with self.assertRaises(TypeError):
                rule.apply(dataset, metrics=True, device="cpu")

    def test_metrics_are_not_available_without_metrics_option(self):
        _register_rows_source("unit_test_filter_metrics_disabled")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics_disabled", Path(tmpdir), [0])
            result = FilterRule(
                name="disabled",
                factory=_metric_factory,
            ).apply(dataset, device="cpu")

            with self.assertRaises(ValueError):
                list(result.iter_metrics())

    def test_rule_apply_writes_metrics_with_workers(self):
        _register_rows_source("unit_test_filter_metrics_workers")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset(
                "unit_test_filter_metrics_workers",
                Path(tmpdir),
                list(range(6)),
            )
            result = FilterRule(
                name="with_workers",
                factory=_metric_factory,
            ).apply(dataset, metrics=True, device=("cpu:0", "cpu:1"), max_shard_samples=2)

            rows = list(result.iter_metrics())

        self.assertEqual([row["index"] for row in rows], list(range(6)))
        self.assertEqual([row["label"] for row in rows], ["accept", "reject"] * 3)

    def test_rule_apply_sets_ddp_environment_for_workers(self):
        _register_rows_source("unit_test_filter_worker_env")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset(
                "unit_test_filter_worker_env",
                Path(tmpdir),
                [0, 1, 2, 3],
            )
            result = FilterRule(
                name="worker_env",
                factory=_env_factory,
            ).apply(dataset, metrics=True, device=("cpu:0", "cpu:1"))
            rows = list(result.iter_metrics())

        world_sizes = {row["metrics"]["world_size"] for row in rows}
        devices = {row["metrics"]["device"] for row in rows}
        ranks = {row["metrics"]["rank"] for row in rows}
        self.assertEqual(world_sizes, {"2"})
        self.assertEqual(devices, {"cpu:0", "cpu:1"})
        self.assertEqual(ranks, {"0", "1"})

    def test_rule_apply_writes_empty_metrics_manifest(self):
        _register_rows_source("unit_test_filter_metrics_empty")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_metrics_empty", Path(tmpdir), [])
            result = FilterRule(
                name="empty",
                factory=_metric_factory,
            ).apply(dataset, metrics=True, device="cpu")
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
            dataset = _dataset("unit_test_filter_parallel_options", Path(tmpdir), [0])
            rule = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            )

            with self.assertRaises(ValueError):
                rule.apply(dataset, device=())
            with self.assertRaises(ValueError):
                rule.apply(dataset, commit_samples=0)
            with self.assertRaises(ValueError):
                rule.apply(dataset, max_shard_samples=0)

    def test_filtered_dataset_requires_labels(self):
        _register_rows_source("unit_test_filter_requires_labels")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_requires_labels", Path(tmpdir), [0])
            rule = FilterRule(
                name="all",
                factory=lambda: lambda sample: True,
            )

            with self.assertRaises(ValueError):
                FilteredDataset(dataset, rule, labels=())

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
                cache_root=root / "dataset-cache",
            ).merge(source)
            rule = FilterRule(
                name="has_longcat",
                factory=lambda: lambda sample: AudioView.LONGCAT
                in sample[Role.DEFAULT, Modality.AUDIO].views,
            )

            filtered = FilteredDataset(
                merged,
                rule,
                labels="accept",
                device="cpu",
                cache_root=root / "filter-cache",
            )
            sample = filtered[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(set(audio.views), {AudioView.WAVEFORM, AudioView.LONGCAT})
        self.assertEqual(sample[Role.DEFAULT, Modality.TEXT].views[TextView.TEXT], "hello")

    def test_filter_rule_can_apply_to_filtered_dataset(self):
        _register_rows_source("unit_test_filter_chain")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_chain", root, [0, 1, 2, 3, 4])
            first = FilterRule(
                name="gte_two",
                factory=lambda: lambda sample: _value(sample) >= 2,
            ).apply(dataset, device="cpu").select("accept")
            seen = []
            second_rule = FilterRule(
                name="even_after_gte_two",
                factory=lambda: lambda sample: _track_even(sample, seen),
            )

            result = second_rule.apply(first, device="cpu")
            selected = result.select("accept")

        self.assertEqual(seen, [2, 3, 4])
        self.assertEqual(_values(selected), [2, 4])
        self.assertEqual(selected.indices, (0, 2))
        self.assertEqual(result.counts, {"accept": 2, "reject": 1})

    def test_chained_filter_cache_is_distinct_from_physical_filter_cache(self):
        _register_rows_source("unit_test_filter_chain_cache")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_chain_cache", root, [0, 1, 2, 3])
            first = FilterRule(
                name="gte_two",
                factory=lambda: lambda sample: _value(sample) >= 2,
            ).apply(dataset, device="cpu").select("accept")
            second_rule = FilterRule(
                name="even",
                factory=lambda: lambda sample: _value(sample) % 2 == 0,
            )

            physical = second_rule.apply(dataset, device="cpu")
            chained = second_rule.apply(first, device="cpu")
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
            dataset = _dataset("unit_test_filter_predicate_type", Path(tmpdir), [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: 1,
            )

            with self.assertRaises(TypeError):
                rule.apply(dataset, device="cpu")

    def test_filter_label_must_not_be_empty(self):
        _register_rows_source("unit_test_filter_empty_label")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_empty_label", Path(tmpdir), [0])
            rule = FilterRule(
                name="bad",
                factory=lambda: lambda sample: "",
            )

            with self.assertRaises(ValueError):
                rule.apply(dataset, device="cpu")


class Route(StrEnum):
    REVIEW = auto()


class _RowsSource:
    def prepare(self, spec: Spec, cache_path: Path):
        return [{"value": value} for value in spec.load_options["values"]]


def _register_rows_source(name: str) -> None:
    register_source(name, _RowsSource)


def _dataset(source: str, cache_root: Path, values: list[int]) -> AnyDataset:
    return AnyDataset(
        Spec(source=source, path="/tmp/rows", load_options={"values": values}),
        cache_root=cache_root,
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
