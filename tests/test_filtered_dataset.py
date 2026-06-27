from __future__ import annotations

import json
import tempfile
import unittest
from enum import StrEnum, auto
from pathlib import Path

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioMeta,
    AudioReq,
    AudioView,
    FilterRule,
    ImageReq,
    ImageView,
    Modality,
    Role,
    Spec,
    register_source,
)


class FilteredDatasetTest(unittest.TestCase):
    def test_rule_apply_partitions_bool_labels(self):
        _register_rows_source("unit_test_filter_bool")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_bool", root, [0, 1, 2, 3])
            rule = FilterRule(
                name="even",
                schema=_label_schema(),
                predicate=lambda sample: _value(sample) % 2 == 0,
            )

            result = rule.apply(dataset)
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
                schema=_label_schema(),
                predicate=_route,
            )

            result = rule.apply(dataset)
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
                schema=_label_schema(),
                predicate=lambda sample: True,
            ).apply(dataset)

            selected = result.select("review")

        self.assertEqual(len(selected), 0)
        self.assertEqual(selected.indices, ())

    def test_select_deduplicates_labels(self):
        _register_rows_source("unit_test_filter_deduplicate")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_deduplicate", Path(tmpdir), [0, 1])
            result = FilterRule(
                name="all",
                schema=_label_schema(),
                predicate=lambda sample: True,
            ).apply(dataset)

            selected = result.select(True, "accept")

        self.assertEqual(selected.labels, ("accept",))
        self.assertEqual(selected.indices, (0, 1))

    def test_rule_predicate_receives_schema_resolved_sample(self):
        _register_rows_source("unit_test_filter_schema_sample")
        seen = []
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_schema_sample", Path(tmpdir), [0])
            rule = FilterRule(
                name="schema",
                schema=_label_schema(),
                predicate=lambda sample: seen.append(sample) or True,
            )

            rule.apply(dataset)

        sample = seen[0]
        audio = sample[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(audio.views, {})
        self.assertEqual(audio.meta, {AudioMeta.LABEL: 0})

    def test_rule_apply_reuses_ready_cache(self):
        _register_rows_source("unit_test_filter_reuses")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_reuses", root, [0, 1, 2, 3])
            first_rule = FilterRule(
                name="gte_two",
                schema=_label_schema(),
                predicate=lambda sample: _value(sample) >= 2,
            )
            first_rule.apply(dataset)
            second_rule = FilterRule(
                name="gte_two",
                schema=_label_schema(),
                predicate=lambda sample: calls.append(sample) or False,
            )

            result = second_rule.apply(dataset)

        self.assertEqual(_values(result.select("accept")), [2, 3])
        self.assertEqual(calls, [])

    def test_rule_apply_rebuilds_when_base_count_changes(self):
        _register_rows_source("unit_test_filter_rebuilds")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = _dataset("unit_test_filter_rebuilds", root, [0, 1, 2])
            rule = FilterRule(
                name="all",
                schema=_label_schema(),
                predicate=lambda sample: True,
            )
            rule.apply(first)
            second = _dataset("unit_test_filter_rebuilds", root, [0, 1, 2, 3])

            result = rule.apply(second)

        self.assertEqual(_values(result.select("accept")), [0, 1, 2, 3])

    def test_rule_apply_rebuilds_when_schema_changes(self):
        _register_rows_source("unit_test_filter_schema_rebuilds")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_schema_rebuilds", root, [0])
            label_rule = FilterRule(
                name="same_name",
                schema=_label_schema(),
                predicate=lambda sample: True,
            )
            label_result = label_rule.apply(dataset)
            waveform_rule = FilterRule(
                name="same_name",
                schema={
                    (Role.DEFAULT, Modality.AUDIO): AudioReq(
                        views=frozenset({AudioView.WAVEFORM}),
                    )
                },
                predicate=lambda sample: False,
            )

            waveform_result = waveform_rule.apply(dataset)

        self.assertNotEqual(label_result.cache_path, waveform_result.cache_path)
        self.assertEqual(_values(label_result.select("accept")), [0])
        self.assertEqual(_values(waveform_result.select("reject")), [0])

    def test_filtered_dataset_shards_after_remap(self):
        _register_rows_source("unit_test_filter_shards")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_shards", root, [0, 1, 2, 3, 4])
            filtered = FilterRule(
                name="all",
                schema=_label_schema(),
                predicate=lambda sample: True,
            ).apply(dataset).select("accept")

            shard = [_value(sample) for sample in filtered.iter_shard(2, 1)]

        self.assertEqual(shard, [1, 3])

    def test_result_and_filtered_dataset_attributes_are_read_only(self):
        _register_rows_source("unit_test_filter_readonly")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_readonly", Path(tmpdir), [0])
            result = FilterRule(
                name="all",
                schema=_label_schema(),
                predicate=lambda sample: True,
            ).apply(dataset)
            filtered = result.select("accept")

        with self.assertRaises(AttributeError):
            result.labels = ()
        with self.assertRaises(TypeError):
            result.partitions["accept"] = ()
        with self.assertRaises(AttributeError):
            filtered.indices = ()
        with self.assertRaises(AttributeError):
            filtered.cache_path = Path("/tmp/changed")

    def test_rule_metadata_is_written_under_physical_cache_path(self):
        _register_rows_source("unit_test_filter_metadata")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_metadata", root, [0, 1])
            rule = FilterRule(
                name="keep_v1",
                schema=_label_schema(),
                predicate=lambda sample: True,
            )

            result = rule.apply(dataset)
            metadata = json.loads((result.cache_path / "rule.json").read_text(encoding="utf-8"))
            manifest = json.loads((result.cache_path / "partitions.json").read_text(encoding="utf-8"))

        self.assertEqual(result.cache_path.parent, root / dataset.spec.cache_relpath / "filters")
        self.assertEqual(metadata["schema_version"], 3)
        self.assertEqual(metadata["base"]["spec_id"], dataset.spec.id)
        self.assertEqual(metadata["base"]["sample_count"], 2)
        self.assertEqual(metadata["rule"]["name"], "keep_v1")
        self.assertEqual(
            metadata["rule"]["schema"],
            [
                {
                    "role": "default",
                    "modality": "audio",
                    "views": [],
                    "meta": ["label"],
                }
            ],
        )
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
                schema=_label_schema(),
                predicate=lambda sample: True,
            )

            result = rule.apply(dataset, commit_samples=2, max_shard_samples=2)
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
                schema=_label_schema(),
                predicate=_mod_three,
            )

            result = rule.apply(dataset, num_workers=2, max_shard_samples=2)
            selected = result.select("one", "two")

        self.assertEqual(result.counts, {"zero": 4, "one": 4, "two": 4})
        self.assertEqual(selected.indices, (1, 2, 4, 5, 7, 8, 10, 11))

    def test_rule_apply_rejects_invalid_parallel_options(self):
        _register_rows_source("unit_test_filter_parallel_options")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_parallel_options", Path(tmpdir), [0])
            rule = FilterRule(
                name="all",
                schema=_label_schema(),
                predicate=lambda sample: True,
            )

            with self.assertRaises(ValueError):
                rule.apply(dataset, num_workers=0)
            with self.assertRaises(ValueError):
                rule.apply(dataset, commit_samples=0)
            with self.assertRaises(ValueError):
                rule.apply(dataset, max_shard_samples=0)

    def test_filter_rule_identity_sorts_schema_entries(self):
        first = FilterRule(
            name="same",
            schema={
                (Role.DEFAULT, Modality.AUDIO): AudioReq(
                    meta=frozenset({AudioMeta.LABEL}),
                ),
                (Role.DEFAULT, Modality.IMAGE): ImageReq(
                    views=frozenset({ImageView.PIXEL}),
                ),
            },
            predicate=lambda sample: True,
        )
        second = FilterRule(
            name="same",
            schema={
                (Role.DEFAULT, Modality.IMAGE): ImageReq(
                    views=frozenset({ImageView.PIXEL}),
                ),
                (Role.DEFAULT, Modality.AUDIO): AudioReq(
                    meta=frozenset({AudioMeta.LABEL}),
                ),
            },
            predicate=lambda sample: True,
        )

        self.assertEqual(first.id, second.id)

    def test_filter_rule_schema_view_is_read_only(self):
        rule = FilterRule(
            name="readonly",
            schema=_label_schema(),
            predicate=lambda sample: True,
        )

        with self.assertRaises(TypeError):
            rule.schema[Role.DEFAULT, Modality.AUDIO] = AudioReq()

    def test_filter_rule_attributes_are_read_only(self):
        rule = FilterRule(
            name="readonly",
            schema=_label_schema(),
            predicate=lambda sample: True,
        )

        with self.assertRaises(AttributeError):
            rule.name = "changed"
        with self.assertRaises(AttributeError):
            rule.predicate = lambda sample: False

    def test_filter_predicate_must_return_supported_label(self):
        _register_rows_source("unit_test_filter_predicate_type")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_predicate_type", Path(tmpdir), [0])
            rule = FilterRule(
                name="bad",
                schema=_label_schema(),
                predicate=lambda sample: 1,
            )

            with self.assertRaises(TypeError):
                rule.apply(dataset)

    def test_filter_label_must_not_be_empty(self):
        _register_rows_source("unit_test_filter_empty_label")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_empty_label", Path(tmpdir), [0])
            rule = FilterRule(
                name="bad",
                schema=_label_schema(),
                predicate=lambda sample: "",
            )

            with self.assertRaises(ValueError):
                rule.apply(dataset)


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


def _label_schema():
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioReq(
            meta=frozenset({AudioMeta.LABEL}),
        )
    }


def _route(sample):
    value = _value(sample)
    if value == 0:
        return True
    if value in {1, 2}:
        return Route.REVIEW
    return "reject"


def _mod_three(sample):
    value = _value(sample)
    if value % 3 == 0:
        return "zero"
    if value % 3 == 1:
        return "one"
    return "two"


def _value(sample):
    return sample[Role.DEFAULT, Modality.AUDIO].meta[AudioMeta.LABEL]


def _values(dataset):
    return [_value(sample) for sample in dataset]


if __name__ == "__main__":
    unittest.main()
