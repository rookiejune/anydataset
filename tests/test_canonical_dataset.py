import unittest
from unittest import mock

import numpy as np
import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioMeta,
    AudioReq,
    AudioView,
    FieldGroup,
    FieldRef,
    ImageItem,
    ImageView,
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
        self.assertEqual(req.meta, frozenset())

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

    def test_preset_create_accepts_transforms(self):
        ref = (Role.DEFAULT, Modality.IMAGE)
        dataset = Preset.MNIST.create(
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
        )
        dataset._dataset = _ShardableRows([{"value": index} for index in range(24)])
        worker = _WorkerInfo(num_workers=4, id=2)

        with (
            mock.patch("anydataset._sharding.dist.is_available", return_value=True),
            mock.patch("anydataset._sharding.dist.is_initialized", return_value=True),
            mock.patch("anydataset._sharding.dist.get_world_size", return_value=3),
            mock.patch("anydataset._sharding.dist.get_rank", return_value=1),
            mock.patch("anydataset._sharding.get_worker_info", return_value=worker),
        ):
            values = list(dataset)

        self.assertEqual(values, [7, 19])
        self.assertEqual(dataset.dataset.shard_calls, [(12, 7)])

    def test_map_dataset_drops_tail_by_rank_before_worker_shard(self):
        values_by_rank: list[list[int]] = []

        for rank in range(4):
            dataset = _map_dataset(range(7))
            worker = _WorkerInfo(num_workers=8, id=0)
            with (
                mock.patch("anydataset._sharding.dist.is_available", return_value=True),
                mock.patch("anydataset._sharding.dist.is_initialized", return_value=True),
                mock.patch("anydataset._sharding.dist.get_world_size", return_value=4),
                mock.patch("anydataset._sharding.dist.get_rank", return_value=rank),
                mock.patch("anydataset._sharding.get_worker_info", return_value=worker),
            ):
                values_by_rank.append(list(dataset))

        self.assertEqual(values_by_rank, [[0], [1], [2], [3]])

    def test_multiple_dataset_splits_pytorch_workers(self):
        dataset = MultipleAnyDataset([_map_dataset(range(6))])
        worker = _WorkerInfo(num_workers=2, id=1)

        with mock.patch("anydataset._sharding.get_worker_info", return_value=worker):
            values = list(dataset)

        self.assertEqual(values, [1, 3, 5])

    def test_multiple_dataset_merges_rank_and_worker_shards(self):
        dataset = MultipleAnyDataset([_map_dataset(range(24))])
        worker = _WorkerInfo(num_workers=4, id=2)

        with (
            mock.patch("anydataset._sharding.dist.is_available", return_value=True),
            mock.patch("anydataset._sharding.dist.is_initialized", return_value=True),
            mock.patch("anydataset._sharding.dist.get_world_size", return_value=3),
            mock.patch("anydataset._sharding.dist.get_rank", return_value=1),
            mock.patch("anydataset._sharding.get_worker_info", return_value=worker),
        ):
            values = list(dataset)

        self.assertEqual(values, [7, 19])

    def test_multiple_dataset_uses_child_runtime_rank_shards(self):
        values_by_rank: list[list[int]] = []

        for rank in range(4):
            dataset = MultipleAnyDataset([_map_dataset(range(7))])
            worker = _WorkerInfo(num_workers=8, id=0)
            with (
                mock.patch("anydataset._sharding.dist.is_available", return_value=True),
                mock.patch("anydataset._sharding.dist.is_initialized", return_value=True),
                mock.patch("anydataset._sharding.dist.get_world_size", return_value=4),
                mock.patch("anydataset._sharding.dist.get_rank", return_value=rank),
                mock.patch("anydataset._sharding.get_worker_info", return_value=worker),
            ):
                values_by_rank.append(list(dataset))

        self.assertEqual(values_by_rank, [[0], [1], [2], [3]])

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

        batch = Task.AUDIO_CODEC.collate_fn()(samples)

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

    def test_collate_fn_batches_tensor_mapping_views_by_key(self):
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
                            "acoustic_codes": torch.tensor(
                                [[4, 5, 6], [7, 8, 9]]
                            ),
                        }
                    },
                )
            },
            {
                ref: AudioItem(
                    views={
                        AudioView.LONGCAT: {
                            "semantic_codes": torch.tensor([10, 11]),
                            "acoustic_codes": torch.tensor([[12, 13], [14, 15]]),
                        }
                    },
                )
            },
        ]

        batch = collate_fn(schema)(samples)

        longcat = batch.sample[ref].views[AudioView.LONGCAT]
        self.assertTrue(
            torch.equal(
                longcat["semantic_codes"],
                torch.tensor([[1, 2, 3], [10, 11, 0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                longcat["acoustic_codes"],
                torch.tensor(
                    [
                        [[4, 5, 6], [7, 8, 9]],
                        [[12, 13, 0], [14, 15, 0]],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.masks[FieldRef(ref, FieldGroup.VIEWS, AudioView.LONGCAT)],
                torch.tensor([[True, True, True], [True, True, False]]),
            )
        )

    def test_collate_fn_requires_mapping_tensors_in_sample_share_length(self):
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

        with self.assertRaisesRegex(ValueError, "must share the same last dimension"):
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
