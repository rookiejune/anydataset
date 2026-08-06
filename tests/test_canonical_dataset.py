import pickle
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch

from anydataset import (
    AnyDataset,
    IterableAnyDataset,
    Preset,
    Source,
    Spec,
    resolve_dataset,
)
from anydataset._runtime.sharding import runtime_shard
from anydataset.dataset import MapStyleABC
from anydataset.dataset.collate import FieldGroup, FieldRef, collate_fn
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioReq,
    AudioView,
    ImageItem,
    ImageView,
    Lang,
    Modality,
    Role,
    TextView,
    TextItem,
    TextMeta,
    TextReq,
)
from anydataset.presets import WMT19


def _audio_codec_schema():
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioReq(
            views=frozenset({AudioView.WAVEFORM}),
        )
    }


def _machine_translation_schema():
    text = TextReq(views=frozenset({TextView.TEXT}))
    return {
        (Role.SOURCE, Modality.TEXT): text,
        (Role.TARGET, Modality.TEXT): text,
    }


class _ExtendedTextReq(TextReq):
    tag: str


class CanonicalDatasetTest(unittest.TestCase):
    def test_generic_dataset_rejects_preset_inputs(self):
        for dataset_type in (AnyDataset, IterableAnyDataset):
            for value in (Preset.MNIST, "mnist", "mnist:validation"):
                with self.subTest(dataset_type=dataset_type, value=value):
                    with self.assertRaisesRegex(TypeError, "AnyDataset.preset"):
                        dataset_type(value)

        dataset = AnyDataset(
            Preset.MNIST.spec(),
            parse_fn=lambda row: row["label"],
        )
        dataset._dataset = [{"label": 7}]
        self.assertEqual(dataset[0], 7)

    def test_dataset_uses_falsey_callable_parser(self):
        dataset = AnyDataset(
            Spec(source=Source.HF, path="unused"),
            parse_fn=_FalseyParser(),
        )
        dataset._dataset = [{"value": 3}]

        self.assertEqual(dataset[0], {"parsed": 3})

    def test_dataset_rejects_non_callable_parser(self):
        with self.assertRaisesRegex(TypeError, "parse_fn"):
            AnyDataset(Spec(source=Source.HF, path="unused"), parse_fn=0)

    def test_dataset_pickle_preserves_subclass_state_and_drops_cached_data(self):
        dataset = _StatefulAnyDataset(Spec(source=Source.HF, path="unused"))
        dataset.extra = "needed"
        dataset._dataset = [{"cached": True}]

        restored = pickle.loads(pickle.dumps(dataset))

        self.assertEqual(restored.extra, "needed")
        self.assertIsNone(restored._dataset)
        self.assertIsNone(restored._cache_manager)

    def test_items_and_requirements_reject_cross_modality_keys(self):
        cases = (
            lambda: AudioItem(views={TextView.TEXT: "wrong"}),
            lambda: ImageItem(meta={TextMeta.LANG: "wrong"}),
            lambda: TextItem(views={ImageView.PIXEL: "wrong"}),
            lambda: AudioReq(views=frozenset({TextView.TEXT})),
            lambda: TextReq(meta=frozenset({AudioMeta.LABEL})),
        )

        for create in cases:
            with self.subTest(create=create):
                with self.assertRaises(TypeError):
                    create()

    def test_text_language_meta_requires_lang_enum(self):
        item = TextItem(
            views={TextView.TEXT: "hello"},
            meta={TextMeta.LANG: Lang.EN},
        )

        self.assertEqual(item.meta[TextMeta.LANG], Lang.EN)
        batched = TextItem(meta={TextMeta.LANG: [Lang.EN, Lang.DE]})
        self.assertEqual(batched.meta[TextMeta.LANG], [Lang.EN, Lang.DE])
        with self.assertRaisesRegex(TypeError, "TextMeta.LANG"):
            TextItem(
                views={TextView.TEXT: "hello"},
                meta={TextMeta.LANG: "en"},
            )
        with self.assertRaisesRegex(TypeError, "TextMeta.LANG"):
            TextItem(meta={TextMeta.LANG: [Lang.EN, "de"]})

    def test_items_and_requirements_are_immutable(self):
        source_views = {TextView.TEXT: "before"}
        source_meta = {TextMeta.LANG: Lang.EN}
        item = TextItem(views=source_views, meta=source_meta)
        requirement = TextReq(views={TextView.TEXT})

        source_views[TextView.TEXT] = "after"
        source_meta.clear()
        with self.assertRaises(FrozenInstanceError):
            item.views = {TextView.TEXT: "after"}
        with self.assertRaises(TypeError):
            item.views[TextView.TEXT] = "after"
        with self.assertRaises(TypeError):
            del item.meta[TextMeta.LANG]
        with self.assertRaises(FrozenInstanceError):
            requirement.views = frozenset()
        with self.assertRaises(FrozenInstanceError):
            del item.meta
        restored_item = pickle.loads(pickle.dumps(item))
        with self.assertRaises(TypeError):
            restored_item.views[TextView.TEXT] = "after"
        restored = pickle.loads(pickle.dumps(requirement))
        with self.assertRaises(FrozenInstanceError):
            restored.meta = frozenset()
        legacy_item = TextItem.__new__(TextItem)
        legacy_item.__setstate__(
            {
                "views": {TextView.TEXT: "legacy"},
                "meta": {},
            }
        )
        with self.assertRaises(FrozenInstanceError):
            legacy_item.views = {}
        legacy_requirement = TextReq.__new__(TextReq)
        legacy_requirement.__setstate__(
            {
                "views": frozenset({TextView.TEXT}),
                "meta": frozenset(),
            }
        )
        with self.assertRaises(FrozenInstanceError):
            legacy_requirement.views = frozenset()
        extended_requirement = _ExtendedTextReq.__new__(_ExtendedTextReq)
        extended_requirement.__setstate__(
            {
                "views": frozenset({TextView.TEXT}),
                "meta": frozenset(),
                "tag": "preserved",
            }
        )
        with self.assertRaises(FrozenInstanceError):
            extended_requirement.tag = "changed"
        invalid_legacy_item = AudioItem.__new__(AudioItem)
        with self.assertRaises(TypeError):
            invalid_legacy_item.__setstate__(
                {
                    "views": {TextView.TEXT: "wrong"},
                    "meta": {},
                }
            )
        with self.assertRaises(TypeError):
            legacy_item.views[TextView.TEXT] = "changed"
        with self.assertRaises(TypeError):
            hash(item)

        self.assertEqual(item.views[TextView.TEXT], "before")
        self.assertEqual(item.meta[TextMeta.LANG], Lang.EN)
        self.assertEqual(restored_item, item)
        self.assertEqual(requirement.views, frozenset({TextView.TEXT}))
        self.assertEqual(hash(requirement), hash(TextReq(views={TextView.TEXT})))
        self.assertEqual(legacy_item.views[TextView.TEXT], "legacy")
        self.assertEqual(hash(legacy_requirement), hash(requirement))
        self.assertEqual(extended_requirement.tag, "preserved")

    def test_dataset_write_uses_dataset_specific_default_split(self):
        canonical = AnyDataset(Spec(source=Source.HF, path="unused", split="train"))
        generic = _EmptyMapDataset()

        with mock.patch(
            "anydataset.dataset.abc._write_dataset",
            return_value=Path("output"),
        ) as write_dataset:
            canonical.write("output")
            canonical_split = write_dataset.call_args.kwargs["split"]
            generic.write("output")
            generic_split = write_dataset.call_args.kwargs["split"]
            canonical.write("output", split="validation")
            explicit_split = write_dataset.call_args.kwargs["split"]

        self.assertEqual(canonical_split, "train")
        self.assertIsNone(generic_split)
        self.assertEqual(explicit_split, "validation")

    def test_resolves_preset_to_spec(self):
        spec = resolve_dataset("fleurs:validation")

        self.assertEqual(spec.source, Source.HF)
        self.assertEqual(spec.path, "google/fleurs")
        self.assertEqual(spec.split, "validation")
        self.assertNotIn("streaming", spec.load_options)
        self.assertEqual(spec.load_options["config_name"], "en_us")
        self.assertFalse(hasattr(spec, "name"))
        self.assertFalse(hasattr(spec, "key"))

    def test_spec_id_is_stable_physical_identity(self):
        spec = Preset.FSD50K.spec(split="dev")
        same = Spec(
            source=Source.HF_FILES,
            path="Fhrozen/FSD50k",
            split="dev",
            version="main",
            load_options={
                "repo_type": "dataset",
                "path_template": "clips/{split}",
                "suffixes": (".wav",),
            },
        )
        different = Spec(
            source=Source.HF_FILES,
            path="Fhrozen/FSD50k",
            split="eval",
            version="main",
            load_options={
                "repo_type": "dataset",
                "path_template": "clips/{split}",
                "suffixes": (".wav",),
            },
        )
        different_revision = Preset.FSD50K.spec(split="dev", revision="v1")

        self.assertEqual(spec.id, same.id)
        self.assertNotEqual(spec.id, different.id)
        self.assertNotEqual(spec.id, different_revision.id)
        self.assertEqual(spec.to_dict()["id"], spec.id)

    def test_spec_id_ignores_operational_prepare_workers(self):
        base = Spec(source="sharded_csv", path="/data/bitext")
        with_workers = Spec(
            source="sharded_csv",
            path="/data/bitext",
            load_options={"prepare_workers": 4},
        )
        zero_workers = Spec(
            source="sharded_csv",
            path="/data/bitext",
            load_options={"prepare_workers": 0},
        )

        self.assertEqual(base.id, with_workers.id)
        self.assertEqual(base.id, zero_workers.id)
        self.assertEqual(base.cache_relpath, with_workers.cache_relpath)
        self.assertNotIn("prepare_workers", base.to_dict()["load_options"])
        self.assertNotIn("prepare_workers", with_workers.to_dict()["load_options"])

    def test_spec_root_field_is_physical_for_non_tsv_sources(self):
        base = Spec(source="custom_rows", path="/data/rows")
        with_root = Spec(
            source="custom_rows",
            path="/data/rows",
            load_options={"root_field": "root"},
        )

        self.assertNotEqual(base.id, with_root.id)
        self.assertEqual(
            with_root.to_dict()["load_options"]["root_field"],
            "root",
        )

    def test_spec_load_options_are_frozen(self):
        spec = Spec(
            source=Source.HF,
            path="org/data",
            load_options={"streaming": True},
        )

        with self.assertRaises(TypeError):
            spec.load_options["streaming"] = False

    def test_spec_fields_are_immutable(self):
        spec = Spec(source=Source.HF, path="org/data")

        with self.assertRaises(FrozenInstanceError):
            spec.path = "other/data"
        with self.assertRaises(FrozenInstanceError):
            del spec.path

    def test_spec_rejects_invalid_physical_fields(self):
        cases = (
            ({"path": Path("data")}, TypeError, "Spec.path"),
            ({"path": ""}, ValueError, "Spec.path"),
            ({"path": "data", "split": ""}, ValueError, "Spec.split"),
            ({"path": "data", "version": 1}, TypeError, "Spec.version"),
            ({"path": "data", "load_options": []}, TypeError, "load_options"),
            (
                {"path": "data", "load_options": {"nested": {1: "value"}}},
                TypeError,
                "keys must be strings",
            ),
        )

        for kwargs, error, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(error, message):
                    Spec(source=Source.HF, **kwargs)

    def test_spec_load_options_are_deeply_frozen(self):
        spec = Spec(
            source=Source.HF,
            path="org/data",
            load_options={
                "files": ["a.jsonl"],
                "options": {"streaming": True},
            },
        )

        self.assertEqual(spec.load_options["files"], ("a.jsonl",))
        with self.assertRaises(AttributeError):
            spec.load_options["files"].append("b.jsonl")
        with self.assertRaises(TypeError):
            spec.load_options["options"]["streaming"] = False

    def test_spec_nested_load_options_are_picklable_and_remain_frozen(self):
        spec = Spec(
            source=Source.HF,
            path="org/data",
            load_options={"data_files": {"train": ["a.jsonl"]}},
        )

        restored = pickle.loads(pickle.dumps(spec))

        self.assertEqual(restored, spec)
        self.assertEqual(restored.id, spec.id)
        self.assertEqual(restored.load_options["data_files"]["train"], ("a.jsonl",))
        with self.assertRaises(TypeError):
            restored.load_options["data_files"]["train"] = ("b.jsonl",)

    def test_spec_id_does_not_change_when_source_options_are_mutated(self):
        load_options = {"files": ["a.jsonl"]}
        spec = Spec(source=Source.HF, path="org/data", load_options=load_options)
        before = spec.id
        values = {spec: "cached"}

        load_options["files"].append("b.jsonl")

        self.assertEqual(spec.id, before)
        self.assertEqual(values[spec], "cached")
        self.assertEqual(spec.to_dict()["load_options"]["files"], ["a.jsonl"])

    def test_spec_to_dict_does_not_expose_identity_state(self):
        spec = Spec(
            source=Source.HF,
            path="org/data",
            load_options={"files": ["a.jsonl"]},
        )
        payload = spec.to_dict()

        payload["load_options"]["files"].append("b.jsonl")

        self.assertEqual(spec.to_dict()["load_options"]["files"], ["a.jsonl"])

    def test_spec_id_accepts_path_load_options(self):
        spec = Spec(
            source=Source.HF,
            path="org/data",
            load_options={"data_dir": Path("/tmp/data")},
        )

        self.assertEqual(spec.to_dict()["load_options"]["data_dir"], "/tmp/data")
        self.assertEqual(
            spec.id,
            Spec(
                source=Source.HF,
                path="org/data",
                load_options={"data_dir": "/tmp/data"},
            ).id,
        )

    def test_audio_codec_schema_uses_role_modality_keys(self):
        schema = _audio_codec_schema()

        req = schema[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(req.views, frozenset({AudioView.WAVEFORM}))
        self.assertEqual(req.meta, frozenset())

    def test_machine_translation_schema_uses_source_target_text_roles(self):
        schema = _machine_translation_schema()

        source = schema[Role.SOURCE, Modality.TEXT]
        target = schema[Role.TARGET, Modality.TEXT]
        self.assertEqual(source.views, frozenset({TextView.TEXT}))
        self.assertEqual(target.views, frozenset({TextView.TEXT}))

    def test_wmt19_preset_resolves_to_hf_spec(self):
        spec = resolve_dataset("wmt19:validation")

        self.assertEqual(spec.source, Source.HF)
        self.assertEqual(spec.path, "wmt/wmt19")
        self.assertEqual(spec.split, "validation")
        self.assertEqual(spec.load_options["config_name"], "cs-en")
        self.assertNotIn("streaming", spec.load_options)

    def test_wmt19_preset_maps_translation_roles(self):
        sample = WMT19().parse_fn(
            {
                "translation": {
                    "cs": "Caj je horky.",
                    "en": "The tea is hot.",
                }
            }
        )

        source = sample[Role.SOURCE, Modality.TEXT]
        target = sample[Role.TARGET, Modality.TEXT]
        self.assertEqual(source.views[TextView.TEXT], "Caj je horky.")
        self.assertEqual(target.views[TextView.TEXT], "The tea is hot.")

    def test_wmt19_preset_uses_config_language_pair(self):
        dataset = WMT19(config_name="DE-EN")
        sample = dataset.parse_fn(
            {
                "translation": {
                    "de": "Der Tee ist heiss.",
                    "en": "The tea is hot.",
                }
            }
        )

        source = sample[Role.SOURCE, Modality.TEXT]
        target = sample[Role.TARGET, Modality.TEXT]
        self.assertEqual(dataset.spec.load_options["config_name"], "de-en")
        self.assertEqual(source.views[TextView.TEXT], "Der Tee ist heiss.")
        self.assertEqual(target.views[TextView.TEXT], "The tea is hot.")

    def test_wmt19_preset_accepts_explicit_language_pair(self):
        dataset = WMT19(source_lang="DE", target_lang="EN")

        self.assertEqual(dataset.spec.load_options["config_name"], "de-en")

    def test_wmt19_preset_normalizes_lang_enum_values(self):
        dataset = WMT19(source_lang=Lang.ZH, target_lang=Lang.EN)

        sample = dataset.parse_fn(
            {"translation": {"zh": "Ni hao.", "en": "Hello."}}
        )

        self.assertEqual(dataset.spec.load_options["config_name"], "zh-en")
        self.assertEqual(
            sample[Role.SOURCE, Modality.TEXT].meta[TextMeta.LANG],
            Lang.ZH,
        )
        self.assertEqual(
            sample[Role.TARGET, Modality.TEXT].meta[TextMeta.LANG],
            Lang.EN,
        )

    def test_wmt19_spec_owns_language_pair_options(self):
        spec = Preset.WMT19.spec(
            source_lang=Lang.DE,
            target_lang="en",
            download_mode="reuse_dataset_if_exists",
        )

        self.assertEqual(spec.load_options["config_name"], "de-en")
        self.assertEqual(
            spec.load_options["download_mode"],
            "reuse_dataset_if_exists",
        )
        self.assertNotIn("source_lang", spec.load_options)
        self.assertNotIn("target_lang", spec.load_options)
        extended = Preset.WMT19.spec(source_lang="GU", target_lang="EN")
        self.assertEqual(extended.load_options["config_name"], "gu-en")

    def test_wmt19_rejects_invalid_language_options(self):
        cases: tuple[tuple[dict[str, Any], type[Exception], str], ...] = (
            ({"source_lang": ""}, ValueError, "source_lang"),
            ({"target_lang": " "}, ValueError, "target_lang"),
            ({"source_lang": 1}, TypeError, "source_lang"),
            ({"source_lang": "Lang.ZH"}, ValueError, "source_lang"),
            ({"source_lang": "zh-CN"}, ValueError, "source_lang"),
            ({"config_name": 1}, TypeError, "config_name"),
            ({"config_name": "Lang.ZH-en"}, ValueError, "config_name source"),
            ({"config_name": "zh-CN-en"}, ValueError, "<source>-<target>"),
        )

        for kwargs, error, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(error, message):
                    WMT19(**kwargs)

    def test_wmt19_preset_rejects_conflicting_config_and_languages(self):
        with self.assertRaises(ValueError):
            WMT19(config_name="cs-en", source_lang="de", target_lang="en")

    def test_resolve_sample_trims_to_schema(self):
        sample = {
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={
                    AudioView.WAVEFORM: ([0.0], 16000),
                    AudioView.FILE: "audio.wav",
                },
            )
        }
        schema = {
            (Role.DEFAULT, Modality.AUDIO): AudioReq(
                views=frozenset({AudioView.WAVEFORM}),
            )
        }

        resolved = AnyDataset.resolve_sample(sample, schema)

        audio = resolved[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(audio.views, {AudioView.WAVEFORM: ([0.0], 16000)})
        self.assertEqual(audio.meta, {})

    def test_resolve_sample_requires_selected_meta_fields(self):
        sample = {
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: ([0.0], 16000)},
            )
        }
        schema = {
            (Role.DEFAULT, Modality.AUDIO): AudioReq(
                views=frozenset({AudioView.WAVEFORM}),
                meta=frozenset({AudioMeta.LABEL}),
            )
        }

        with self.assertRaises(KeyError):
            AnyDataset.resolve_sample(sample, schema)

    def test_resolve_sample_rejects_mismatched_requirement_type(self):
        sample = {
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: ([0.0], 16000)},
            )
        }
        schema = {
            (Role.DEFAULT, Modality.AUDIO): TextReq(
                views=frozenset({TextView.TEXT}),
            )
        }

        with self.assertRaisesRegex(
            TypeError,
            "item and schema requirement types must match",
        ):
            AnyDataset.resolve_sample(sample, schema)

    def test_map_preset_accepts_transforms(self):
        ref = (Role.DEFAULT, Modality.IMAGE)
        dataset = AnyDataset.preset(
            "mnist",
            transforms={
                ref: lambda item: ImageItem(
                    views={ImageView.PIXEL: item.views[ImageView.PIXEL] + 1},
                    meta=item.meta,
                )
            }
        )
        dataset._dataset = [
            {
                "image": torch.tensor([[1, 2]]),
                "label": 0,
            }
        ]

        image = dataset[0][ref]

        self.assertTrue(
            torch.equal(image.views[ImageView.PIXEL], torch.tensor([[2, 3]]))
        )
        self.assertNotIn("transforms", dataset.spec.load_options)

    def test_presets_are_map_style_only(self):
        self.assertFalse(hasattr(IterableAnyDataset, "preset"))
        dataset = AnyDataset.preset("wmt19")
        self.assertIsInstance(dataset, AnyDataset)

    def test_map_dataset_applies_reference_transforms(self):
        ref = (Role.DEFAULT, Modality.IMAGE)
        dataset = AnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row,
            transforms={
                ref: lambda item: ImageItem(
                    views={ImageView.PIXEL: item.views[ImageView.PIXEL] + 1},
                    meta=item.meta,
                )
            },
        )
        dataset._dataset = [
            {
                ref: ImageItem(
                    views={ImageView.PIXEL: torch.tensor([[1, 2]])},
                )
            }
        ]

        image = dataset[0][ref]

        self.assertTrue(
            torch.equal(image.views[ImageView.PIXEL], torch.tensor([[2, 3]]))
        )

    def test_iterable_dataset_applies_reference_transforms(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        dataset = IterableAnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: {
                ref: AudioItem(
                    views={AudioView.WAVEFORM: (torch.tensor([row["value"]]), 4)},
                )
            },
            transforms={
                ref: lambda item: AudioItem(
                    views={
                        AudioView.WAVEFORM: (
                            item.views[AudioView.WAVEFORM][0] * 2,
                            item.views[AudioView.WAVEFORM][1],
                        )
                    },
                    meta=item.meta,
                )
            },
        )
        dataset._dataset = [{"value": 3}]

        sample = next(iter(dataset))
        waveform, sample_rate = sample[ref].views[AudioView.WAVEFORM]

        self.assertTrue(torch.equal(waveform, torch.tensor([6])))
        self.assertEqual(sample_rate, 4)

    def test_iterable_dataset_ignores_dataset_native_shard(self):
        # HF is a ShardingSource: rows are taken via source iter_shard
        # (len/getitem), not prepared Dataset.shard().
        dataset = IterableAnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row["value"],
        )
        dataset._dataset = _ShardableRows(
            [
                {"value": 0},
                {"value": 1},
                {"value": 2},
                {"value": 3},
            ]
        )

        values = list(dataset.iter_shard(2, 1))

        self.assertEqual(values, [(1, 1), (3, 3)])
        self.assertEqual(dataset.dataset.shard_calls, [])

    def test_iterable_dataset_falls_back_to_modulo_shard(self):
        dataset = IterableAnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row["value"],
        )
        dataset._source = _PlainSource()
        dataset._dataset = [
            {"path": "/tmp/a"},
            {"path": "/tmp/b"},
            {"path": "/tmp/c"},
            {"path": "/tmp/d"},
        ]
        dataset.iter_rows = lambda: (
            {"value": index} for index, _ in enumerate(dataset.dataset)
        )

        self.assertEqual(list(dataset.iter_shard(2, 1)), [(1, 1), (3, 3)])

    def test_iterable_dataset_uses_source_native_shard(self):
        dataset = IterableAnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row["value"],
        )
        source = _ShardedSource()
        dataset._source = source
        dataset._dataset = _NoScanRows(
            [{"value": index} for index in range(5)]
        )

        values = list(dataset.iter_shard(2, 1))

        self.assertEqual(values, [(1, 1), (3, 3)])
        self.assertEqual(source.calls, [(2, 1)])

    def test_map_dataset_uses_source_native_shard(self):
        dataset = AnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row["value"],
        )
        source = _ShardedSource()
        dataset._source = source
        dataset._dataset = _NoScanRows(
            [{"value": index} for index in range(5)]
        )

        values = list(dataset.iter_shard(2, 1))

        self.assertEqual(values, [(1, 1), (3, 3)])
        self.assertEqual(source.calls, [(2, 1)])

    def test_iterable_shard_requires_source_opt_in(self):
        dataset = IterableAnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row["value"],
        )
        dataset._source = _PlainSource()
        rows = _RawShardedRows([{"value": index} for index in range(4)])
        dataset._dataset = rows

        values = list(dataset.iter_shard(2, 1))

        self.assertEqual(values, [(1, 1), (3, 3)])
        self.assertEqual(rows.shard_calls, [])
        self.assertEqual(rows.iterations, 1)

    def test_map_shard_requires_source_opt_in(self):
        dataset = AnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row["value"],
        )
        dataset._source = _PlainSource()
        rows = _RawShardedRows([{"value": index} for index in range(4)])
        dataset._dataset = rows

        values = list(dataset.iter_shard(2, 1))

        self.assertEqual(values, [(1, 1), (3, 3)])
        self.assertEqual(rows.shard_calls, [])

    def test_map_range_uses_validated_prepared_range(self):
        dataset = AnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row,
        )
        entries = [(0, "zero"), (1, "one"), (2, "two")]
        rows = _RowsWithIndexedRange(["zero", "one", "two"], entries)
        dataset._dataset = rows

        values = list(dataset.iter_indexed_range(0, len(dataset)))

        self.assertEqual(values, entries)
        self.assertEqual(rows.range_calls, [(0, 3)])

    def test_map_range_validates_prepared_range_indexes(self):
        cases: tuple[tuple[Any, type[Exception], str], ...] = (
            (None, TypeError, "return an iterable"),
            ([[0, "zero"]], TypeError, "tuples"),
            ([(True, "zero")], TypeError, "integers"),
            ([(1, "one")], ValueError, "expected 0, got 1"),
            ([(0, "zero"), (0, "duplicate")], ValueError, "expected 1, got 0"),
            ([(0, "zero"), (1, "one")], ValueError, r"cover \[0, 3\)"),
            (
                [(0, "zero"), (1, "one"), (2, "two"), (3, "extra")],
                ValueError,
                "extra index 3",
            ),
        )

        for entries, error, message in cases:
            with self.subTest(entries=entries):
                dataset = AnyDataset(
                    spec=Spec(source=Source.HF, path="/tmp/missing"),
                    parse_fn=lambda row: row,
                )
                dataset._dataset = _RowsWithIndexedRange(
                    ["zero", "one", "two"],
                    entries,
                )

                with self.assertRaisesRegex(error, message):
                    list(dataset.iter_indexed_range(0, len(dataset)))

    def test_iterable_native_shard_validates_global_indexes(self):
        cases = (
            (None, TypeError, "return an iterable"),
            ([([1, {"value": 1}])], TypeError, "tuples"),
            ([(True, {"value": 1})], TypeError, "integers"),
            ([(3, {"value": 3})], ValueError, "expected 1, got 3"),
            (
                [(1, {"value": 1}), (5, {"value": 5})],
                ValueError,
                "expected 3, got 5",
            ),
        )

        for entries, error, message in cases:
            with self.subTest(entries=entries):
                dataset = IterableAnyDataset(
                    spec=Spec(source=Source.HF, path="/tmp/missing"),
                    parse_fn=lambda row: row["value"],
                )
                dataset._source = _FixedShardedSource(entries)
                dataset._dataset = object()

                with self.assertRaisesRegex(error, message):
                    list(dataset.iter_shard(2, 1))

    def test_map_native_shard_validates_global_indexes(self):
        cases = (
            (None, TypeError, "return an iterable"),
            ([([1, {"value": 1}])], TypeError, "tuples"),
            ([(True, {"value": 1})], TypeError, "integers"),
            ([(3, {"value": 3})], ValueError, "expected 1, got 3"),
            (
                [(1, {"value": 1}), (5, {"value": 5})],
                ValueError,
                "expected 3, got 5",
            ),
            (
                [(1, {"value": 1})],
                ValueError,
                "stopped before index 3",
            ),
            (
                [
                    (1, {"value": 1}),
                    (3, {"value": 3}),
                    (5, {"value": 5}),
                ],
                ValueError,
                "extra index 5",
            ),
        )

        for entries, error, message in cases:
            with self.subTest(entries=entries):
                dataset = AnyDataset(
                    spec=Spec(source=Source.HF, path="/tmp/missing"),
                    parse_fn=lambda row: row["value"],
                )
                dataset._source = _FixedShardedSource(entries)
                dataset._dataset = [
                    {"value": index}
                    for index in range(4)
                ]

                with self.assertRaisesRegex(error, message):
                    list(dataset.iter_shard(2, 1))

    def test_iterable_native_shard_does_not_require_a_known_tail(self):
        dataset = IterableAnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row["value"],
        )
        dataset._source = _FixedShardedSource([(1, {"value": 1})])
        dataset._dataset = object()

        self.assertEqual(list(dataset.iter_shard(2, 1)), [(1, 1)])

    def test_iterable_dataset_ignores_non_callable_shard_attribute(self):
        dataset = IterableAnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row["value"],
        )
        dataset._dataset = _RowsWithShardAttribute(
            [{"value": index} for index in range(4)]
        )

        self.assertEqual(list(dataset.iter_shard(2, 1)), [(1, 1), (3, 3)])

    def test_iterable_dataset_merges_rank_and_worker_shards(self):
        # Same contract as ignores_dataset_native_shard: rank/worker flattening
        # goes through the source indexed path, not Dataset.shard().
        dataset = IterableAnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row["value"],
        )
        dataset._dataset = _ShardableRows([{"value": index} for index in range(24)])
        worker = _WorkerInfo(num_workers=4, id=2)

        with (
            mock.patch("anydataset._runtime.sharding.dist.is_available", return_value=True),
            mock.patch("anydataset._runtime.sharding.dist.is_initialized", return_value=True),
            mock.patch("anydataset._runtime.sharding.dist.get_world_size", return_value=3),
            mock.patch("anydataset._runtime.sharding.dist.get_rank", return_value=1),
            mock.patch("anydataset._runtime.sharding.get_worker_info", return_value=worker),
        ):
            values = list(dataset)

        self.assertEqual(values, [7, 19])
        self.assertEqual(dataset.dataset.shard_calls, [])

    def test_map_dataset_drops_tail_by_rank_before_worker_shard(self):
        values_by_rank: list[list[int]] = []

        for rank in range(4):
            dataset = _map_dataset(range(7))
            worker = _WorkerInfo(num_workers=8, id=0)
            with (
                mock.patch("anydataset._runtime.sharding.dist.is_available", return_value=True),
                mock.patch("anydataset._runtime.sharding.dist.is_initialized", return_value=True),
                mock.patch("anydataset._runtime.sharding.dist.get_world_size", return_value=4),
                mock.patch("anydataset._runtime.sharding.dist.get_rank", return_value=rank),
                mock.patch("anydataset._runtime.sharding.get_worker_info", return_value=worker),
            ):
                values_by_rank.append(list(dataset))

        self.assertEqual(values_by_rank, [[0], [1], [2], [3]])

    def test_map_dataset_iter_shard_uses_rank_environment(self):
        dataset = _map_dataset(range(8))

        with mock.patch.dict("os.environ", {"WORLD_SIZE": "2", "RANK": "1"}):
            shard = runtime_shard()
            values = list(dataset.iter_shard(shard.flat_count, shard.flat_index))

        self.assertEqual(values, [(1, 1), (3, 3), (5, 5), (7, 7)])

    def test_collate_fn_pads_tensor_last_dim_and_returns_masks(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        samples = [
            {
                ref: AudioItem(
                    views={AudioView.WAVEFORM: (torch.tensor([1.0, 2.0]), 16000)},
                )
            },
            {
                ref: AudioItem(
                    views={AudioView.WAVEFORM: (torch.tensor([3.0]), 22050)},
                )
            },
        ]

        batch = collate_fn(_audio_codec_schema())(samples)

        audio = batch.sample[ref]
        waveform, sample_rates = audio.views[AudioView.WAVEFORM]
        self.assertTrue(
            torch.equal(
                waveform,
                torch.tensor([[1.0, 2.0], [3.0, 0.0]]),
            )
        )
        self.assertTrue(torch.equal(sample_rates, torch.tensor([16000, 22050])))
        self.assertTrue(
            torch.equal(
                batch.masks[FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM)],
                torch.tensor([[True, True], [True, False]]),
            )
        )

    def test_collate_fn_is_picklable(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        collator = pickle.loads(pickle.dumps(collate_fn(_audio_codec_schema())))

        batch = collator(
            [
                {
                    ref: AudioItem(
                        views={AudioView.WAVEFORM: (torch.tensor([1.0]), 16000)},
                    )
                }
            ]
        )

        waveform, sample_rates = batch.sample[ref].views[AudioView.WAVEFORM]
        self.assertTrue(torch.equal(waveform, torch.tensor([[1.0]])))
        self.assertTrue(torch.equal(sample_rates, torch.tensor([16000])))

    def test_collate_fn_batches_numpy_waveforms(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        samples = [
            {
                ref: AudioItem(
                    views={
                        AudioView.WAVEFORM: (
                            np.array([[1.0, 2.0, 3.0]]),
                            16000,
                        )
                    },
                )
            },
            {
                ref: AudioItem(
                    views={AudioView.WAVEFORM: (np.array([[4.0]]), 16000)},
                )
            },
        ]

        batch = collate_fn({ref: AudioReq(views=frozenset({AudioView.WAVEFORM}))})(
            samples
        )

        waveform, sample_rates = batch.sample[ref].views[AudioView.WAVEFORM]
        self.assertTrue(
            torch.equal(
                waveform,
                torch.tensor(
                    [[[1.0, 2.0, 3.0]], [[4.0, 0.0, 0.0]]],
                    dtype=torch.float64,
                ),
            )
        )
        self.assertTrue(torch.equal(sample_rates, torch.tensor([16000, 16000])))
        self.assertTrue(
            torch.equal(
                batch.masks[FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM)],
                torch.tensor([[[True, True, True]], [[True, False, False]]]),
            )
        )

    def test_collate_fn_batches_codec_views_by_frame(self):
        for view in (
            AudioView.LONGCAT,
            AudioView.DAC,
            AudioView.STABLE,
            AudioView.UNICODEC,
        ):
            with self.subTest(view=view):
                ref = (Role.DEFAULT, Modality.AUDIO)
                schema = {ref: AudioReq(views=frozenset({view}))}
                samples = [
                    {
                        ref: AudioItem(
                            views={
                                view: torch.tensor(
                                    [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
                                )
                            },
                        )
                    },
                    {
                        ref: AudioItem(
                            views={view: torch.tensor([[10, 12, 14], [11, 13, 15]])},
                        )
                    },
                ]

                batch = collate_fn(schema)(samples)

                codes = batch.sample[ref].views[view]
                self.assertTrue(
                    torch.equal(
                        codes,
                        torch.tensor(
                            [
                                [[1, 4, 7], [2, 5, 8], [3, 6, 9]],
                                [[10, 12, 14], [11, 13, 15], [0, 0, 0]],
                            ]
                        ),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        batch.masks[FieldRef(ref, FieldGroup.VIEWS, view)],
                        torch.tensor([[True, True, True], [True, True, False]]),
                    )
                )

    def test_collate_fn_rejects_legacy_codec_mapping(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        schema = {
            ref: AudioReq(
                views=frozenset({AudioView.LONGCAT}),
            )
        }
        samples = [
            {
                ref: AudioItem(
                    views={
                        AudioView.LONGCAT: {
                            "semantic_codes": torch.tensor([1, 2, 3]),
                            "acoustic_codes": torch.tensor([[4, 5], [6, 7]]),
                        }
                    },
                )
            }
        ]

        with self.assertRaisesRegex(TypeError, "Codec view values must be tensors"):
            collate_fn(schema)(samples)

    def test_collate_fn_batches_bicodec_semantic_and_global_units(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        schema = {ref: AudioReq(views=frozenset({AudioView.BICODEC}))}
        samples = [
            {
                ref: AudioItem(
                    views={
                        AudioView.BICODEC: {
                            "semantic": torch.tensor([[1], [2]]),
                            "global": torch.tensor([[10], [11]]),
                        }
                    }
                )
            },
            {
                ref: AudioItem(
                    views={
                        AudioView.BICODEC: {
                            "semantic": torch.tensor([[3]]),
                            "global": torch.tensor([[12], [13]]),
                        }
                    }
                )
            },
        ]

        batch = collate_fn(schema)(samples)
        values = batch.sample[ref].views[AudioView.BICODEC]

        self.assertTrue(
            torch.equal(values["semantic"], torch.tensor([[[1], [2]], [[3], [0]]]))
        )
        self.assertTrue(
            torch.equal(
                values["global"],
                torch.tensor([[[10], [11]], [[12], [13]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.masks[
                    FieldRef(ref, FieldGroup.VIEWS, AudioView.BICODEC)
                ],
                torch.tensor([[True, True], [True, False]]),
            )
        )

    def test_collate_fn_rejects_legacy_bicodec_acoustic_key(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        schema = {ref: AudioReq(views=frozenset({AudioView.BICODEC}))}
        samples = [
            {
                ref: AudioItem(
                    views={
                        AudioView.BICODEC: {
                            "semantic": torch.tensor([[1]]),
                            "acoustic": torch.tensor([[2]]),
                        }
                    }
                )
            }
        ]

        with self.assertRaisesRegex(ValueError, "semantic and global"):
            collate_fn(schema)(samples)

    def test_collate_fn_rejects_non_integer_bicodec_global_units(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        schema = {ref: AudioReq(views=frozenset({AudioView.BICODEC}))}
        samples = [
            {
                ref: AudioItem(
                    views={
                        AudioView.BICODEC: {
                            "semantic": torch.tensor([[1]]),
                            "global": torch.tensor([[2.0]]),
                        }
                    }
                )
            }
        ]

        with self.assertRaisesRegex(TypeError, "global values must contain integer ids"):
            collate_fn(schema)(samples)

    def test_collate_fn_rejects_mixed_codec_dtypes(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        schema = {ref: AudioReq(views=frozenset({AudioView.LONGCAT}))}
        samples = [
            {
                ref: AudioItem(
                    views={AudioView.LONGCAT: torch.tensor([[1]], dtype=torch.int8)},
                )
            },
            {
                ref: AudioItem(
                    views={AudioView.LONGCAT: torch.tensor([[300]], dtype=torch.int64)},
                )
            },
        ]

        with self.assertRaisesRegex(TypeError, "share one dtype"):
            collate_fn(schema)(samples)

    def test_collate_fn_rejects_mixed_codec_devices(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        schema = {ref: AudioReq(views=frozenset({AudioView.LONGCAT}))}
        samples = [
            {ref: AudioItem(views={AudioView.LONGCAT: torch.tensor([[1]])})},
            {
                ref: AudioItem(
                    views={
                        AudioView.LONGCAT: torch.empty(
                            (1, 1),
                            dtype=torch.int64,
                            device="meta",
                        )
                    },
                )
            },
        ]

        with self.assertRaisesRegex(ValueError, "share one device"):
            collate_fn(schema)(samples)

    def test_collate_fn_keeps_non_tensor_meta_as_values(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        schema = {
            ref: AudioReq(
                meta=frozenset({AudioMeta.LABEL}),
            )
        }
        samples = [
            {ref: AudioItem(meta={AudioMeta.LABEL: 1})},
            {ref: AudioItem(meta={AudioMeta.LABEL: 2})},
        ]

        batch = collate_fn(schema)(samples)

        audio = batch.sample[ref]
        self.assertEqual(audio.meta[AudioMeta.LABEL], [1, 2])
        self.assertNotIn(FieldRef(ref, FieldGroup.META, AudioMeta.LABEL), batch.masks)

    def test_collate_fn_keeps_text_language_meta_as_values(self):
        ref = (Role.DEFAULT, Modality.TEXT)
        schema = {
            ref: TextReq(
                views=frozenset({TextView.TEXT}),
                meta=frozenset({TextMeta.LANG}),
            )
        }
        samples = [
            {
                ref: TextItem(
                    views={TextView.TEXT: "hello"},
                    meta={TextMeta.LANG: Lang.EN},
                )
            },
            {
                ref: TextItem(
                    views={TextView.TEXT: "hallo"},
                    meta={TextMeta.LANG: Lang.DE},
                )
            },
        ]

        batch = collate_fn(schema)(samples)

        text = batch.sample[ref]
        self.assertEqual(text.views[TextView.TEXT], ["hello", "hallo"])
        self.assertEqual(text.meta[TextMeta.LANG], [Lang.EN, Lang.DE])
        self.assertNotIn(FieldRef(ref, FieldGroup.META, TextMeta.LANG), batch.masks)

    def test_collate_fn_keeps_mapping_meta_as_values(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        schema = {
            ref: AudioReq(
                meta=frozenset({AudioMeta.LABEL}),
            )
        }
        samples = [
            {ref: AudioItem(meta={AudioMeta.LABEL: {"score": torch.tensor([1])}})},
            {ref: AudioItem(meta={AudioMeta.LABEL: {"score": torch.tensor([2])}})},
        ]

        batch = collate_fn(schema)(samples)

        labels = batch.sample[ref].meta[AudioMeta.LABEL]
        self.assertEqual(len(labels), 2)
        self.assertTrue(torch.equal(labels[0]["score"], torch.tensor([1])))
        self.assertTrue(torch.equal(labels[1]["score"], torch.tensor([2])))
        self.assertNotIn(FieldRef(ref, FieldGroup.META, AudioMeta.LABEL), batch.masks)

    def test_collate_fn_requires_declared_meta_fields(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        schema = {
            ref: AudioReq(
                meta=frozenset({AudioMeta.LABEL}),
            )
        }
        samples = [
            {ref: AudioItem(meta={AudioMeta.LABEL: "speech"})},
            {ref: AudioItem()},
        ]

        with self.assertRaises(KeyError):
            collate_fn(schema)(samples)


class _PlainSource:
    def prepare(self, spec, cache_path):
        raise AssertionError("prepared dataset was injected")


class _ShardableRows:
    """Map-style stand-in for HF rows; ``shard()`` must stay unused."""

    def __init__(self, rows):
        self.rows = rows
        self.shard_calls = []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self):
        yield from self.rows

    def shard(self, *, num_shards: int, index: int):
        self.shard_calls.append((num_shards, index))
        return (
            row
            for row_index, row in enumerate(self.rows)
            if row_index % num_shards == index
        )


class _RowsWithShardAttribute:
    shard = "metadata"

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self):
        yield from self.rows


class _ShardedSource:
    def __init__(self):
        self.calls = []

    def prepare(self, spec, cache_path):
        raise AssertionError("prepared dataset was injected")

    def iter_shard(self, dataset, *, num_shards: int, shard_id: int):
        self.calls.append((num_shards, shard_id))
        return (
            (index, dataset.rows[index])
            for index in range(shard_id, len(dataset.rows), num_shards)
        )


class _FixedShardedSource:
    def __init__(self, entries):
        self.entries = entries

    def prepare(self, spec, cache_path):
        raise AssertionError("prepared dataset was injected")

    def iter_shard(self, dataset, *, num_shards: int, shard_id: int):
        return self.entries


class _NoScanRows:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self):
        raise AssertionError("native sharding must not scan all rows")


class _RawShardedRows:
    def __init__(self, rows):
        self.rows = rows
        self.shard_calls = []
        self.iterations = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self):
        self.iterations += 1
        yield from self.rows

    def iter_shard(self, num_shards: int, shard_id: int):
        self.shard_calls.append((num_shards, shard_id))
        raise AssertionError("raw sharding requires source opt-in")


class _RowsWithIndexedRange:
    def __init__(self, rows, entries):
        self.rows = rows
        self.entries = entries
        self.range_calls = []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def iter_indexed_range(self, start: int, stop: int):
        self.range_calls.append((start, stop))
        return self.entries


class _FalseyParser:
    def __bool__(self):
        return False

    def __call__(self, row):
        return {"parsed": row["value"]}


class _StatefulAnyDataset(AnyDataset):
    pass


class _EmptyMapDataset(MapStyleABC):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


def _map_dataset(rows):
    dataset = AnyDataset(
        spec=Spec(source=Source.HF, path="/tmp/missing"),
        parse_fn=lambda row: row,
    )
    dataset._dataset = list(rows)
    return dataset


class _WorkerInfo:
    def __init__(self, *, num_workers: int, id: int) -> None:
        self.num_workers = num_workers
        self.id = id


if __name__ == "__main__":
    unittest.main()
