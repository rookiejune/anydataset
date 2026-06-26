import json
import tempfile
import unittest
from pathlib import Path

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioMeta,
    AudioReq,
    AudioView,
    FilterRule,
    Modality,
    Role,
    Spec,
    cached_filter,
    register_source,
)


class FilteredDatasetTest(unittest.TestCase):
    def test_cached_filter_builds_remapped_dataset(self):
        _register_rows_source("unit_test_filter_builds")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_builds", root, [0, 1, 2, 3])
            rule = FilterRule(
                name="even",
                version="1",
                config=(("mod", 2),),
                predicate=_is_even,
            )

            filtered = cached_filter(dataset, rule)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(_values(filtered), [0, 2])
        self.assertEqual(filtered.indices, (0, 2))

    def test_cached_filter_reuses_ready_cache(self):
        _register_rows_source("unit_test_filter_reuses")
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_reuses", root, [0, 1, 2, 3])
            first_rule = FilterRule(
                name="gte",
                version="1",
                config=(("min", 2),),
                predicate=lambda sample: _value(sample) >= 2,
            )
            cached_filter(dataset, first_rule)
            second_rule = FilterRule(
                name="gte",
                version="1",
                config=(("min", 2),),
                predicate=lambda sample: calls.append(sample) or False,
            )

            filtered = cached_filter(dataset, second_rule)

        self.assertEqual(_values(filtered), [2, 3])
        self.assertEqual(calls, [])

    def test_cached_filter_rebuilds_when_base_count_changes(self):
        _register_rows_source("unit_test_filter_rebuilds")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = _dataset("unit_test_filter_rebuilds", root, [0, 1, 2])
            rule = FilterRule(
                name="all",
                version="1",
                config=(),
                predicate=lambda sample: True,
            )
            cached_filter(first, rule)
            second = _dataset("unit_test_filter_rebuilds", root, [0, 1, 2, 3])

            filtered = cached_filter(second, rule)

        self.assertEqual(_values(filtered), [0, 1, 2, 3])

    def test_filtered_dataset_shards_after_remap(self):
        _register_rows_source("unit_test_filter_shards")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_shards", root, [0, 1, 2, 3, 4])
            filtered = cached_filter(
                dataset,
                FilterRule(
                    name="all",
                    version="1",
                    config=(),
                    predicate=lambda sample: True,
                ),
            )

            shard = [_value(sample) for sample in filtered.iter_shard(2, 1)]

        self.assertEqual(shard, [1, 3])

    def test_filtered_dataset_attributes_are_read_only(self):
        _register_rows_source("unit_test_filter_readonly")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_readonly", Path(tmpdir), [0])
            filtered = cached_filter(
                dataset,
                FilterRule(
                    name="all",
                    version="1",
                    config=(),
                    predicate=lambda sample: True,
                ),
            )

        with self.assertRaises(AttributeError):
            filtered.indices = ()
        with self.assertRaises(AttributeError):
            filtered.cache_path = Path("/tmp/changed")

    def test_rule_metadata_is_written_under_physical_cache_path(self):
        _register_rows_source("unit_test_filter_metadata")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = _dataset("unit_test_filter_metadata", root, [0])
            rule = FilterRule(
                name="keep",
                version="2026-06-26",
                config=(("labels", ("a",)),),
                predicate=lambda sample: True,
            )

            filtered = cached_filter(dataset, rule)
            metadata_path = filtered.cache_path / "rule.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(filtered.cache_path.parent, root / dataset.spec.cache_relpath / "filters")
        self.assertEqual(metadata["base"]["spec_id"], dataset.spec.id)
        self.assertEqual(metadata["base"]["sample_count"], 1)
        self.assertEqual(metadata["rule"]["name"], "keep")
        self.assertEqual(metadata["rule"]["version"], "2026-06-26")
        self.assertEqual(metadata["rule"]["config"], {"labels": ["a"]})

    def test_filter_rule_requires_json_serializable_config(self):
        with self.assertRaises(TypeError):
            FilterRule(
                name="bad",
                version="1",
                config=(("values", {1}),),
                predicate=lambda sample: True,
            )

    def test_filter_rule_rejects_mutable_config_values(self):
        with self.assertRaises(TypeError):
            FilterRule(
                name="labels",
                version="1",
                config=(("labels", ["a"]),),
                predicate=lambda sample: True,
            )

    def test_filter_rule_identity_sorts_config_keys(self):
        first = FilterRule(
            name="labels",
            version="1",
            config=(("labels", ("a",)), ("nested", (("min", 1),))),
            predicate=lambda sample: True,
        )
        second = FilterRule(
            name="labels",
            version="1",
            config=(("nested", (("min", 1),)), ("labels", ("a",))),
            predicate=lambda sample: True,
        )

        self.assertEqual(first.id, second.id)
        self.assertEqual(first.identity["config"], {"labels": ["a"], "nested": {"min": 1}})

    def test_filter_rule_rejects_duplicate_config_keys(self):
        with self.assertRaises(ValueError):
            FilterRule(
                name="duplicate",
                version="1",
                config=(("min", 1), ("min", 2)),
                predicate=lambda sample: True,
            )

    def test_filter_rule_config_view_is_read_only(self):
        rule = FilterRule(
            name="readonly",
            version="1",
            config=(("nested", (("min", 1),)),),
            predicate=lambda sample: True,
        )

        with self.assertRaises(TypeError):
            rule.config["nested"] = {}
        with self.assertRaises(TypeError):
            rule.config["nested"]["min"] = 2

    def test_filter_rule_attributes_are_read_only(self):
        rule = FilterRule(
            name="readonly",
            version="1",
            config=(),
            predicate=lambda sample: True,
        )

        with self.assertRaises(AttributeError):
            rule.name = "changed"
        with self.assertRaises(AttributeError):
            rule.predicate = lambda sample: False

    def test_filter_predicate_must_return_bool(self):
        _register_rows_source("unit_test_filter_predicate_type")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = _dataset("unit_test_filter_predicate_type", Path(tmpdir), [0])
            rule = FilterRule(
                name="bad",
                version="1",
                config=(),
                predicate=lambda sample: 1,
            )

            with self.assertRaises(TypeError):
                cached_filter(dataset, rule)


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


def _value(sample):
    return sample[Role.DEFAULT, Modality.AUDIO].meta[AudioMeta.LABEL]


def _values(dataset):
    return [_value(sample) for sample in dataset]


def _is_even(sample) -> bool:
    return _value(sample) % 2 == 0


if __name__ == "__main__":
    unittest.main()
