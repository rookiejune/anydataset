import unittest
from unittest import mock

import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioKey,
    AudioOptKey,
    AudioReq,
    AudioView,
    FieldGroup,
    FieldRef,
    IterableAnyDataset,
    Modality,
    MultipleAnyDataset,
    Preset,
    Role,
    Source,
    Spec,
    Task,
    TextView,
    collate_fn,
    resolve_dataset,
)
from anydataset.presets import WMT19


class CanonicalDatasetTest(unittest.TestCase):
    def test_resolves_preset_to_spec(self):
        spec = resolve_dataset("fleurs:validation")

        self.assertEqual(spec.source, Source.HF)
        self.assertEqual(spec.path, "google/fleurs")
        self.assertEqual(spec.split, "validation")
        self.assertEqual(spec.load_options["streaming"], True)
        self.assertFalse(hasattr(spec, "name"))
        self.assertFalse(hasattr(spec, "key"))

    def test_spec_id_is_stable_physical_identity(self):
        spec = Preset.FSD50K.spec(split="dev")
        same = Spec(source=Source.HF, path="Fhrozen/FSD50k", split="dev")
        different = Spec(source=Source.HF, path="Fhrozen/FSD50k", split="train")

        self.assertEqual(spec.id, same.id)
        self.assertNotEqual(spec.id, different.id)
        self.assertEqual(spec.to_dict()["id"], spec.id)

    def test_spec_load_options_are_frozen(self):
        spec = Spec(
            source=Source.HF,
            path="org/data",
            load_options={"streaming": True},
        )

        with self.assertRaises(TypeError):
            spec.load_options["streaming"] = False

    def test_task_schema_uses_role_modality_keys(self):
        schema = Task.AUDIO_CODEC.schema()

        req = schema[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(req.views, frozenset({AudioView.WAVEFORM}))
        self.assertEqual(req.required, frozenset({AudioKey.SAMPLE_RATE}))

    def test_machine_translation_schema_uses_source_target_text_roles(self):
        schema = Task.MACHINE_TRANSLATION.schema()

        source = schema[Role.SOURCE, Modality.TEXT]
        target = schema[Role.TARGET, Modality.TEXT]
        self.assertEqual(source.views, frozenset({TextView.TEXT}))
        self.assertEqual(target.views, frozenset({TextView.TEXT}))

    def test_wmt19_preset_resolves_to_streaming_hf_spec(self):
        spec = resolve_dataset("wmt19:validation")

        self.assertEqual(spec.source, Source.HF)
        self.assertEqual(spec.path, "wmt/wmt19")
        self.assertEqual(spec.split, "validation")
        self.assertEqual(spec.load_options["config_name"], "cs-en")
        self.assertEqual(spec.load_options["streaming"], True)

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
        sample = WMT19(config_name="de-en").parse_fn(
            {
                "translation": {
                    "de": "Der Tee ist heiss.",
                    "en": "The tea is hot.",
                }
            }
        )

        source = sample[Role.SOURCE, Modality.TEXT]
        target = sample[Role.TARGET, Modality.TEXT]
        self.assertEqual(source.views[TextView.TEXT], "Der Tee ist heiss.")
        self.assertEqual(target.views[TextView.TEXT], "The tea is hot.")

    def test_wmt19_preset_accepts_explicit_language_pair(self):
        dataset = WMT19(source_lang="de", target_lang="en")

        self.assertEqual(dataset.spec.load_options["config_name"], "de-en")

    def test_wmt19_preset_rejects_conflicting_config_and_languages(self):
        with self.assertRaises(ValueError):
            WMT19(config_name="cs-en", source_lang="de", target_lang="en")

    def test_resolve_sample_trims_to_schema(self):
        sample = {
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={
                    AudioView.WAVEFORM: [0.0],
                    AudioView.FILE: "audio.wav",
                },
                required={AudioKey.SAMPLE_RATE: 16000},
            )
        }
        schema = {
            (Role.DEFAULT, Modality.AUDIO): AudioReq(
                views=frozenset({AudioView.WAVEFORM}),
                required=frozenset({AudioKey.SAMPLE_RATE}),
            )
        }

        resolved = AnyDataset.resolve_sample(sample, schema)

        audio = resolved[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(audio.views, {AudioView.WAVEFORM: [0.0]})
        self.assertEqual(audio.required, {AudioKey.SAMPLE_RATE: 16000})

    def test_resolve_sample_requires_selected_optional_fields(self):
        sample = {
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: [0.0]},
                required={AudioKey.SAMPLE_RATE: 16000},
            )
        }
        schema = {
            (Role.DEFAULT, Modality.AUDIO): AudioReq(
                views=frozenset({AudioView.WAVEFORM}),
                required=frozenset({AudioKey.SAMPLE_RATE}),
                optional=frozenset({AudioOptKey.LABEL}),
            )
        }

        with self.assertRaises(KeyError):
            AnyDataset.resolve_sample(sample, schema)

    def test_iterable_dataset_uses_source_native_shard(self):
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

        self.assertEqual(values, [1, 3])
        self.assertEqual(dataset.dataset.shard_calls, [(2, 1)])

    def test_iterable_dataset_falls_back_to_modulo_shard(self):
        dataset = IterableAnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row["value"],
        )
        dataset._dataset = [
            {"path": "/tmp/a"},
            {"path": "/tmp/b"},
            {"path": "/tmp/c"},
            {"path": "/tmp/d"},
        ]
        dataset.iter_rows = lambda: (
            {"value": index} for index, _ in enumerate(dataset.dataset)
        )

        self.assertEqual(list(dataset.iter_shard(2, 1)), [1, 3])

    def test_iterable_dataset_merges_rank_and_worker_shards(self):
        dataset = IterableAnyDataset(
            spec=Spec(source=Source.HF, path="/tmp/missing"),
            parse_fn=lambda row: row["value"],
            num_shards=2,
            shard_id=1,
        )
        dataset._dataset = _ShardableRows([{"value": index} for index in range(12)])
        worker = _WorkerInfo(num_workers=3, id=2)

        with mock.patch("anydataset.dataset.abc.get_worker_info", return_value=worker):
            values = list(dataset)

        self.assertEqual(values, [5, 11])
        self.assertEqual(dataset.dataset.shard_calls, [(6, 5)])

    def test_multiple_dataset_splits_pytorch_workers(self):
        dataset = MultipleAnyDataset([range(6)])
        worker = _WorkerInfo(num_workers=2, id=1)

        with mock.patch("anydataset.dataset.abc.get_worker_info", return_value=worker):
            values = list(dataset)

        self.assertEqual(values, [1, 3, 5])

    def test_multiple_dataset_merges_rank_and_worker_shards(self):
        dataset = MultipleAnyDataset([range(12)], num_shards=2, shard_id=1)
        worker = _WorkerInfo(num_workers=3, id=2)

        with mock.patch("anydataset.dataset.abc.get_worker_info", return_value=worker):
            values = list(dataset)

        self.assertEqual(values, [5, 11])

    def test_collate_fn_pads_tensor_last_dim_and_returns_masks(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        samples = [
            {
                ref: AudioItem(
                    views={AudioView.WAVEFORM: torch.tensor([1.0, 2.0])},
                    required={AudioKey.SAMPLE_RATE: 16000},
                )
            },
            {
                ref: AudioItem(
                    views={AudioView.WAVEFORM: torch.tensor([3.0])},
                    required={AudioKey.SAMPLE_RATE: 22050},
                )
            },
        ]

        batch = Task.AUDIO_CODEC.collate_fn()(samples)

        audio = batch.sample[ref]
        self.assertTrue(
            torch.equal(
                audio.views[AudioView.WAVEFORM],
                torch.tensor([[1.0, 2.0], [3.0, 0.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.masks[FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM)],
                torch.tensor([[True, True], [True, False]]),
            )
        )
        self.assertTrue(
            torch.equal(
                audio.required[AudioKey.SAMPLE_RATE],
                torch.tensor([16000, 22050]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.masks[FieldRef(ref, FieldGroup.REQUIRED, AudioKey.SAMPLE_RATE)],
                torch.tensor([True, True]),
            )
        )

    def test_collate_fn_masks_partially_missing_optional_tensor(self):
        ref = (Role.DEFAULT, Modality.AUDIO)
        schema = {
            ref: AudioReq(
                optional=frozenset({AudioOptKey.DURATION}),
            )
        }
        samples = [
            {ref: AudioItem(optional={AudioOptKey.DURATION: 1.5})},
            {ref: AudioItem()},
        ]

        batch = collate_fn(schema)(samples)

        audio = batch.sample[ref]
        self.assertTrue(
            torch.equal(
                audio.optional[AudioOptKey.DURATION],
                torch.tensor([1.5, 0.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.masks[FieldRef(ref, FieldGroup.OPTIONAL, AudioOptKey.DURATION)],
                torch.tensor([True, False]),
            )
        )


class _ShardableRows:
    def __init__(self, rows):
        self.rows = rows
        self.shard_calls = []

    def __iter__(self):
        yield from self.rows

    def shard(self, *, num_shards: int, index: int):
        self.shard_calls.append((num_shards, index))
        return (
            row
            for row_index, row in enumerate(self.rows)
            if row_index % num_shards == index
        )


class _WorkerInfo:
    def __init__(self, *, num_workers: int, id: int) -> None:
        self.num_workers = num_workers
        self.id = id


if __name__ == "__main__":
    unittest.main()
