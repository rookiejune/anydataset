import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioView,
    FieldGroup,
    FieldRef,
    ImageItem,
    ImageView,
    Modality,
    ModalityMaterializer,
    Role,
    Source,
    Spec,
    TextItem,
    TextMeta,
    TextView,
)
from anydataset.store import DatasetWriter, ViewMaterializer
from anydataset.store.jsonio import read_json
from anydataset.store.manifestio import read_samples_manifest, read_view_manifest
from anydataset.store.paths import view_dir
from anydataset.store.reader import read_store_dataset
from anydataset._parallel import iter_indexed_shard


class ViewMaterializerTest(unittest.TestCase):
    def test_materializer_uses_output_dir_name_as_dataset_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "longcat-delta"
            sample = _audio_sample(torch.tensor([[1.0, 2.0]]))

            ViewMaterializer(target, split="train").write(
                dataset_factory=_DatasetFactory((sample,)),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )

            self.assertEqual(
                read_json(target / "dataset.json")["dataset_id"],
                "longcat-delta",
            )

    def test_materializer_writes_only_provider_output_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            dataset = _source_dataset(source, root, [_audio_sample(waveform)])

            ViewMaterializer(target, split="train").write(
                dataset_factory=_DatasetFactory(dataset),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            sample = _read_sample(target, root)

            self.assertEqual(
                set(stored.views),
                {(Role.DEFAULT, Modality.AUDIO, AudioView.LONGCAT)},
            )
            self.assertFalse(
                view_dir(
                    target,
                    (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),
                ).exists()
            )
            self.assertEqual(set(sample[Role.DEFAULT, Modality.AUDIO].views), {AudioView.LONGCAT})
            self.assertTrue(
                torch.equal(
                    sample[Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[11, 12, 13]]),
                )
            )

    def test_materializer_uses_batch_provider_when_batch_size_is_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            dataset = _source_dataset(
                source,
                root,
                [
                    _audio_sample(torch.tensor([[1.0, 2.0, 3.0]])),
                    _audio_sample(torch.tensor([[4.0]])),
                ],
            )
            provider = _BatchProvider()

            ViewMaterializer(target, batch_size=2).write(
                dataset_factory=_DatasetFactory(dataset),
                provider_factory=_StaticProviderFactory(provider),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            self.assertEqual(provider.batch_shapes, [(2, 1, 3)])
            self.assertEqual(provider.single_calls, 0)
            self.assertTrue(
                torch.equal(
                    stored[0][Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[1, 2, 3]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    stored[1][Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[4]]),
                )
            )

    def test_materializer_collates_multiple_roles_for_one_batch_provider_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            dataset = _source_dataset(
                source,
                root,
                [
                    _role_audio_sample(
                        source_waveform=torch.tensor([[1.0, 2.0, 3.0]]),
                        target_waveform=torch.tensor([[4.0]]),
                    ),
                    _role_audio_sample(
                        source_waveform=torch.tensor([[5.0]]),
                        target_waveform=torch.tensor([[6.0, 7.0]]),
                    ),
                ],
            )
            provider = _MultiRoleBatchProvider()

            ViewMaterializer(target, batch_size=2).write(
                dataset_factory=_DatasetFactory(dataset),
                provider_factory=_StaticProviderFactory(provider),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            self.assertEqual(
                provider.batch_refs,
                [
                    (
                        (Role.SOURCE, Modality.AUDIO),
                        (Role.TARGET, Modality.AUDIO),
                    )
                ],
            )
            self.assertTrue(
                torch.equal(
                    stored[0][Role.SOURCE, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[1, 2, 3]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    stored[0][Role.TARGET, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[4]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    stored[1][Role.SOURCE, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[5]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    stored[1][Role.TARGET, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[6, 7]]),
                )
            )

    def test_materializer_rejects_wrong_batch_provider_output_count(self):
        provider = _BadBatchProvider()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "1 outputs for 2 samples"):
                ViewMaterializer(root / "target", batch_size=2).write(
                    dataset_factory=_DatasetFactory(
                        (
                            _audio_sample(torch.tensor([[1.0]])),
                            _audio_sample(torch.tensor([[2.0]])),
                        )
                    ),
                    provider_factory=_StaticProviderFactory(provider),
                    devices="cpu",
                )

    def test_materializer_splits_oom_batch_and_recaptures_padding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            dataset = _source_dataset(
                source,
                root,
                [
                    _audio_sample(torch.tensor([[1.0, 2.0, 3.0, 4.0]])),
                    _audio_sample(torch.tensor([[5.0, 6.0, 7.0]])),
                    _audio_sample(torch.tensor([[8.0, 9.0]])),
                    _audio_sample(torch.tensor([[10.0]])),
                ],
            )
            provider = _SplitOnOomBatchProvider()

            ViewMaterializer(target, batch_size=4).write(
                dataset_factory=_DatasetFactory(dataset),
                provider_factory=_StaticProviderFactory(provider),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            codes = [
                stored[index][Role.DEFAULT, Modality.AUDIO]
                .views[AudioView.LONGCAT]["semantic_codes"]
                for index in range(4)
            ]

        self.assertEqual(
            provider.batch_shapes,
            [
                (4, 1, 4),
                (2, 1, 4),
                (1, 1, 4),
                (1, 1, 3),
                (2, 1, 2),
                (1, 1, 2),
                (1, 1, 1),
            ],
        )
        self.assertTrue(torch.equal(codes[0], torch.tensor([[1, 2, 3, 4]])))
        self.assertTrue(torch.equal(codes[1], torch.tensor([[5, 6, 7]])))
        self.assertTrue(torch.equal(codes[2], torch.tensor([[8, 9]])))
        self.assertTrue(torch.equal(codes[3], torch.tensor([[10]])))

    def test_materializer_does_not_split_non_oom_batch_errors(self):
        provider = _NonOomBatchProvider()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(RuntimeError, "bad batch"):
                ViewMaterializer(root / "target", batch_size=2).write(
                    dataset_factory=_DatasetFactory(
                        (
                            _audio_sample(torch.tensor([[1.0]])),
                            _audio_sample(torch.tensor([[2.0]])),
                        )
                    ),
                    provider_factory=_StaticProviderFactory(provider),
                    devices="cpu",
                )

        self.assertEqual(provider.batch_shapes, [(2, 1, 1)])

    def test_materializer_processes_multiple_roles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            dataset = _source_dataset(
                source,
                root,
                [
                    {
                        (Role.SOURCE, Modality.AUDIO): AudioItem(
                            views={AudioView.WAVEFORM: (torch.tensor([[1.0, 2.0]]), 4)}
                        ),
                        (Role.TARGET, Modality.AUDIO): AudioItem(
                            views={AudioView.WAVEFORM: (torch.tensor([[3.0, 4.0]]), 4)}
                        ),
                    }
                ],
            )

            ViewMaterializer(target).write(
                dataset_factory=_DatasetFactory(dataset),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            sample = _read_sample(target, root)

            self.assertEqual(
                set(stored.views),
                {
                    (Role.SOURCE, Modality.AUDIO, AudioView.LONGCAT),
                    (Role.TARGET, Modality.AUDIO, AudioView.LONGCAT),
                },
            )
            self.assertTrue(
                torch.equal(
                    sample[Role.SOURCE, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[11, 12]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    sample[Role.TARGET, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[13, 14]]),
                )
            )

    def test_materializer_skips_items_from_other_modalities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            dataset = _source_dataset(
                source,
                root,
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={AudioView.WAVEFORM: (torch.tensor([[1.0, 2.0]]), 4)}
                        ),
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello"},
                            meta={TextMeta.LANG: "en_us"},
                        ),
                    }
                ],
            )

            ViewMaterializer(target).write(
                dataset_factory=_DatasetFactory(dataset),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            sample = _read_sample(target, root)

            self.assertEqual(
                set(stored.views),
                {(Role.DEFAULT, Modality.AUDIO, AudioView.LONGCAT)},
            )
            self.assertEqual(set(sample), {(Role.DEFAULT, Modality.AUDIO)})
            self.assertTrue(
                torch.equal(
                    sample[Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[11, 12]]),
                )
            )

    def test_modality_materializer_adds_missing_modality_with_empty_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            DatasetWriter(source, dataset_id="toy-text", split="train").write(
                [
                    {
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello"},
                            meta={TextMeta.LANG: "en_us"},
                        )
                    }
                ]
            )

            ModalityMaterializer(target, split="train").write(
                dataset_factory=_StoreDatasetFactory(source, root),
                provider_factory=_TTSProviderFactory(),
                devices="cpu",
            )
            stored = read_store_dataset(target)
            delta = _read_sample(target, root)
            merged = _store_dataset(source, root).merge(_store_dataset(target, root))[0]

            self.assertEqual(
                set(stored.views),
                {(Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)},
            )
            self.assertEqual(set(delta), {(Role.DEFAULT, Modality.AUDIO)})
            waveform, sample_rate = delta[Role.DEFAULT, Modality.AUDIO].views[
                AudioView.WAVEFORM
            ]
            self.assertTrue(torch.equal(waveform, torch.tensor([[5.0]])))
            self.assertEqual(sample_rate, 16000)
            self.assertEqual(delta[Role.DEFAULT, Modality.AUDIO].meta, {})
            self.assertEqual(
                merged[Role.DEFAULT, Modality.TEXT].meta[TextMeta.LANG],
                "en_us",
            )
            self.assertEqual(merged[Role.DEFAULT, Modality.AUDIO].meta, {})

    def test_modality_materializer_collates_multiple_roles_for_one_batch_provider_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            DatasetWriter(source, dataset_id="toy-text", split="train").write(
                [
                    _role_text_sample(source_text="hello", target_text="hi"),
                    _role_text_sample(source_text="world", target_text="ok"),
                ]
            )
            provider = _MultiRoleTTSProvider()

            ModalityMaterializer(target, split="train", batch_size=2).write(
                dataset_factory=_StoreDatasetFactory(source, root),
                provider_factory=_StaticProviderFactory(provider),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            self.assertEqual(
                provider.batch_refs,
                [
                    (
                        (Role.SOURCE, Modality.TEXT),
                        (Role.TARGET, Modality.TEXT),
                    )
                ],
            )
            source_waveform, _ = stored[0][Role.SOURCE, Modality.AUDIO].views[
                AudioView.WAVEFORM
            ]
            target_waveform, _ = stored[0][Role.TARGET, Modality.AUDIO].views[
                AudioView.WAVEFORM
            ]
            self.assertTrue(torch.equal(source_waveform, torch.tensor([[5.0]])))
            self.assertTrue(torch.equal(target_waveform, torch.tensor([[2.0]])))

    def test_modality_materializer_passes_reference_role_output(self):
        sample = {
            (Role.SOURCE, Modality.TEXT): TextItem(views={TextView.TEXT: "source"}),
            (Role.SOURCE, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: (torch.tensor([[1.0, 2.0]]), 16000)}
            ),
            (Role.TARGET, Modality.TEXT): TextItem(views={TextView.TEXT: "target"}),
        }
        provider = _ReferenceTTSProvider(reference_role=Role.SOURCE)
        with tempfile.TemporaryDirectory() as tmpdir:
            ModalityMaterializer(Path(tmpdir) / "target").write(
                dataset_factory=_DatasetFactory((sample,)),
                provider_factory=_StaticProviderFactory(provider),
                devices="cpu",
            )

        self.assertEqual(provider.calls, [("target", 3.0)])

    def test_modality_materializer_requires_reference_output_first(self):
        sample = {
            (Role.SOURCE, Modality.TEXT): TextItem(views={TextView.TEXT: "source"}),
            (Role.TARGET, Modality.TEXT): TextItem(views={TextView.TEXT: "target"}),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "Reference role"):
                ModalityMaterializer(Path(tmpdir) / "target").write(
                    dataset_factory=_DatasetFactory((sample,)),
                    provider_factory=_StaticProviderFactory(
                        _ReferenceTTSProvider(reference_role=Role.SOURCE)
                    ),
                    devices="cpu",
                )

    def test_modality_materializer_reports_modality_batch_reference_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            DatasetWriter(source, dataset_id="toy-text", split="train").write(
                [
                    _role_text_sample(source_text="hello", target_text="hi"),
                    _role_text_sample(source_text="world", target_text="ok"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "Batch modality provider"):
                ModalityMaterializer(
                    root / "target",
                    split="train",
                    batch_size=2,
                ).write(
                    dataset_factory=_StoreDatasetFactory(source, root),
                    provider_factory=_StaticProviderFactory(_BadMultiRoleTTSProvider()),
                    devices="cpu",
                )

    def test_modality_materializer_can_add_text_from_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [_audio_sample(torch.tensor([[1.0, 2.0, 3.0]]))]
            )

            ModalityMaterializer(target, split="train").write(
                dataset_factory=_StoreDatasetFactory(source, root),
                provider_factory=_ASRProviderFactory(),
                devices="cpu",
            )
            sample = _read_sample(target, root)

            self.assertEqual(set(sample), {(Role.DEFAULT, Modality.TEXT)})
            self.assertEqual(
                sample[Role.DEFAULT, Modality.TEXT].views[TextView.TEXT],
                "sum=6",
            )
            self.assertEqual(sample[Role.DEFAULT, Modality.TEXT].meta, {})

    def test_modality_materializer_rejects_existing_output_modality(self):
        sample = {
            (Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "hello"}),
            (Role.DEFAULT, Modality.AUDIO): AudioItem(
                views={AudioView.WAVEFORM: (torch.tensor([[1.0]]), 4)}
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "already has output modality"):
                ModalityMaterializer(Path(tmpdir) / "target").write(
                    dataset_factory=_DatasetFactory((sample,)),
                    provider_factory=_TTSProviderFactory(),
                    devices="cpu",
                )

    def test_modality_materializer_rejects_ambiguous_input_modality(self):
        sample = {
            (Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "hello"}),
            (Role.DEFAULT, Modality.IMAGE): ImageItem(
                views={ImageView.PIXEL: [[1, 2], [3, 4]]}
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "exactly one input modality"):
                ModalityMaterializer(Path(tmpdir) / "target").write(
                    dataset_factory=_DatasetFactory((sample,)),
                    provider_factory=_TTSProviderFactory(),
                    devices="cpu",
                )

    def test_materialized_delta_merges_into_base_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            waveform = torch.tensor([[1.0, 2.0]])
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={AudioView.WAVEFORM: (waveform, 4)}
                        ),
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={TextView.TEXT: "hello"},
                            meta={TextMeta.LANG: "en_us"},
                        ),
                    }
                ]
            )

            ViewMaterializer(target, split="train").write(
                dataset_factory=_StoreDatasetFactory(source, root),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )
            merged = _store_dataset(source, root).merge(_store_dataset(target, root))
            sample = merged[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        text = sample[Role.DEFAULT, Modality.TEXT]
        self.assertEqual(set(audio.views), {AudioView.WAVEFORM, AudioView.LONGCAT})
        self.assertTrue(torch.equal(audio.views[AudioView.WAVEFORM][0], waveform))
        self.assertTrue(
            torch.equal(
                audio.views[AudioView.LONGCAT]["semantic_codes"],
                torch.tensor([[11, 12]]),
            )
        )
        self.assertEqual(text.views[TextView.TEXT], "hello")
        self.assertEqual(text.meta[TextMeta.LANG], "en_us")

    def test_iter_indexed_shard_uses_map_style_indexes(self):
        dataset = [_audio_sample(torch.tensor([[float(index)]])) for index in range(5)]

        self.assertEqual(
            [index for index, _ in iter_indexed_shard(dataset, 2, 1)],
            [1, 3],
        )

    def test_materializer_parts_commit_readable_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            parts = root / "parts"
            samples = [
                _audio_sample(torch.tensor([[float(index)]]))
                for index in range(4)
            ]
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(samples)
            dataset = _store_dataset(source, root)
            materializer = ViewMaterializer(
                target,
                split="train",
            )

            materializer._write_part(
                dataset,
                _Provider(offset=10),
                parts_dir=parts,
                num_shards=2,
                shard_id=0,
            )
            dataset = _store_dataset(source, root)
            materializer._write_part(
                dataset,
                _Provider(offset=10),
                parts_dir=parts,
                num_shards=2,
                shard_id=1,
            )
            materializer._commit_parts(parts)

            stored = read_store_dataset(target)
            indexes = [entry.sample_index for entry in read_samples_manifest(target)]
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.LONGCAT)
            shards = {entry.shard for entry in read_view_manifest(target, view)}

            self.assertEqual(len(stored), 4)
            self.assertEqual(indexes, [0, 1, 2, 3])
            self.assertEqual(shards, {"part-00000-000000.tar", "part-00001-000000.tar"})
            for index in range(4):
                sample = stored[index]
                self.assertTrue(
                    torch.equal(
                        sample[Role.DEFAULT, Modality.AUDIO]
                        .views[AudioView.LONGCAT]["semantic_codes"],
                        torch.tensor([[index + 10]]),
                    )
                )

    def test_materializer_parallel_write_uses_devices_and_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            target = root / "target"
            samples = tuple(
                _audio_sample(torch.tensor([[float(index)]]))
                for index in range(4)
            )

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                ViewMaterializer(target, split="train").write(
                    dataset_factory=_DatasetFactory(samples),
                    provider_factory=_ParallelProviderFactory(),
                    devices=("cpu:0", "cpu:1"),
                )

            stored = read_store_dataset(target)
            logs = _materializer_logs(home)
            self.assertEqual(len(stored), 4)
            self.assertEqual([path.name for path in logs], ["part-00000.log", "part-00001.log"])
            self.assertIn("cpu:0", logs[0].read_text(encoding="utf-8"))
            self.assertIn("cpu:1", logs[1].read_text(encoding="utf-8"))
            self.assertTrue(
                torch.equal(
                    stored[0][Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[0]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    stored[1][Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.LONGCAT]["semantic_codes"],
                    torch.tensor([[101]]),
                )
            )

    def test_materializer_single_device_loader_workers_cover_all_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "target"
            samples = tuple(
                _audio_sample(torch.tensor([[float(index)]]))
                for index in range(6)
            )

            ViewMaterializer(target, split="train", num_workers=2).write(
                dataset_factory=_DatasetFactory(samples),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            self.assertEqual(len(stored), 6)
            for index in range(6):
                self.assertTrue(
                    torch.equal(
                        stored[index][Role.DEFAULT, Modality.AUDIO]
                        .views[AudioView.LONGCAT]["semantic_codes"],
                        torch.tensor([[index + 10]]),
                    )
                )

    def test_materializer_loader_workers_use_dataset_factory_not_dataset_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "target"

            ViewMaterializer(target, split="train", num_workers=2).write(
                dataset_factory=_UnpicklableDatasetFactory(6),
                provider_factory=_ProviderFactory(offset=10),
                devices="cpu",
            )

            stored = read_store_dataset(target)
            self.assertEqual(len(stored), 6)
            for index in range(6):
                self.assertTrue(
                    torch.equal(
                        stored[index][Role.DEFAULT, Modality.AUDIO]
                        .views[AudioView.LONGCAT]["semantic_codes"],
                        torch.tensor([[index + 10]]),
                    )
                )

    def test_materializer_parallel_loader_workers_cover_all_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            target = root / "target"
            samples = tuple(
                _audio_sample(torch.tensor([[float(index)]]))
                for index in range(8)
            )

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                ViewMaterializer(target, split="train", num_workers=2).write(
                    dataset_factory=_DatasetFactory(samples),
                    provider_factory=_ParallelProviderFactory(),
                    devices=("cpu:0", "cpu:1"),
                )

            stored = read_store_dataset(target)
            logs = _materializer_logs(home)
            self.assertEqual(len(stored), 8)
            self.assertEqual([path.name for path in logs], ["part-00000.log", "part-00001.log"])
            for index in range(8):
                expected = index + (100 if index % 2 else 0)
                self.assertTrue(
                    torch.equal(
                        stored[index][Role.DEFAULT, Modality.AUDIO]
                        .views[AudioView.LONGCAT]["semantic_codes"],
                        torch.tensor([[expected]]),
                    )
                )

    def test_modality_materializer_parallel_write_uses_modality_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            target = root / "target"
            samples = tuple(
                {
                    (Role.DEFAULT, Modality.TEXT): TextItem(
                        views={TextView.TEXT: "x" * (index + 1)}
                    )
                }
                for index in range(4)
            )

            with mock.patch.dict(os.environ, {"ANYDATASET_HOME": str(home)}):
                ModalityMaterializer(target, split="train").write(
                    dataset_factory=_DatasetFactory(samples),
                    provider_factory=_ParallelTTSProviderFactory(),
                    devices=("cpu:0", "cpu:1"),
                )

            stored = read_store_dataset(target)
            logs = _materializer_logs(home)
            self.assertEqual(len(stored), 4)
            self.assertEqual([path.name for path in logs], ["part-00000.log", "part-00001.log"])
            self.assertTrue(
                torch.equal(
                    stored[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.WAVEFORM][0],
                    torch.tensor([[1.0]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    stored[1][Role.DEFAULT, Modality.AUDIO].views[AudioView.WAVEFORM][0],
                    torch.tensor([[102.0]]),
                )
            )

    def test_resolve_devices_auto_falls_back_to_cpu(self):
        from anydataset.store.materializer import resolve_devices

        self.assertTrue(resolve_devices("auto"))

    def test_materializer_rejects_unpicklable_parallel_factory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with self.assertRaises(TypeError):
                ViewMaterializer(root / "target").write(
                    dataset_factory=lambda: (),
                    provider_factory=_ParallelProviderFactory(),
                    devices=("cpu:0", "cpu:1"),
                )

    def test_materializer_resume_continues_from_completed_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "target"
            calls = root / "calls.txt"
            samples = tuple(
                _audio_sample(torch.tensor([[float(index)]]))
                for index in range(4)
            )

            with self.assertRaisesRegex(RuntimeError, "stop after first batch"):
                ViewMaterializer(target, split="train", batch_size=2).write(
                    dataset_factory=_DatasetFactory(samples),
                    provider_factory=_FailOnceBatchProviderFactory(calls),
                    devices="cpu",
                    resume=True,
                )

            self.assertFalse(target.exists())
            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["0,1", "2,3"],
            )

            ViewMaterializer(target, split="train", batch_size=2).write(
                dataset_factory=_DatasetFactory(samples),
                provider_factory=_FailOnceBatchProviderFactory(calls),
                devices="cpu",
                resume=True,
            )

            stored = read_store_dataset(target)
            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["0,1", "2,3", "2,3"],
            )
            self.assertFalse((root / ".target.resume").exists())
            for index in range(4):
                self.assertTrue(
                    torch.equal(
                        stored[index][Role.DEFAULT, Modality.AUDIO]
                        .views[AudioView.LONGCAT]["semantic_codes"],
                        torch.tensor([[index]]),
                    )
                )

    def test_modality_materializer_resume_continues_from_completed_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "target"
            calls = root / "tts-calls.txt"
            samples = tuple(
                {
                    (Role.DEFAULT, Modality.TEXT): TextItem(
                        views={TextView.TEXT: "x" * (index + 1)}
                    )
                }
                for index in range(3)
            )

            with self.assertRaisesRegex(RuntimeError, "stop after first batch"):
                ModalityMaterializer(target, split="train", batch_size=2).write(
                    dataset_factory=_DatasetFactory(samples),
                    provider_factory=_FailOnceTTSBatchProviderFactory(calls),
                    devices="cpu",
                    resume=True,
                )

            ModalityMaterializer(target, split="train", batch_size=2).write(
                dataset_factory=_DatasetFactory(samples),
                provider_factory=_FailOnceTTSBatchProviderFactory(calls),
                devices="cpu",
                resume=True,
            )

            stored = read_store_dataset(target)
            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["1,2", "3", "3"],
            )
            for index in range(3):
                waveform, sample_rate = stored[index][
                    Role.DEFAULT, Modality.AUDIO
                ].views[AudioView.WAVEFORM]
                self.assertTrue(torch.equal(waveform, torch.tensor([[float(index + 1)]])))
                self.assertEqual(sample_rate, 16000)


def _source_dataset(path: Path, root: Path, samples):
    DatasetWriter(path, dataset_id="toy-audio", split="train").write(samples)
    return _store_dataset(path, root)


def _store_dataset(path: Path, root: Path):
    return AnyDataset(
        Spec(source=Source.STORE, path=str(path), split="train"),
    )


def _read_sample(path: Path, root: Path):
    dataset = AnyDataset(
        Spec(source=Source.STORE, path=str(path), split="train"),
    )
    return dataset[0]


def _audio_sample(waveform: torch.Tensor):
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (waveform, 4)},
        )
    }


def _role_audio_sample(
    *,
    source_waveform: torch.Tensor,
    target_waveform: torch.Tensor,
):
    return {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (source_waveform, 4)},
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (target_waveform, 4)},
        ),
    }


def _role_text_sample(
    *,
    source_text: str,
    target_text: str,
):
    return {
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: source_text},
        ),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: target_text},
        ),
    }


class _Provider:
    output = AudioView.LONGCAT

    def __init__(self, *, offset=0):
        self.offset = offset

    def __call__(self, views):
        waveform, _ = views[AudioView.WAVEFORM]
        return {"semantic_codes": waveform.to(torch.int64) + int(self.offset)}


class _BatchProvider(_Provider):
    def __init__(self):
        super().__init__()
        self.batch_shapes: list[tuple[int, ...]] = []
        self.single_calls = 0

    def __call__(self, views):
        self.single_calls += 1
        return super().__call__(views)

    def call_batch(self, batch):
        ref = (Role.DEFAULT, Modality.AUDIO)
        waveform, _ = batch.sample[ref].views[AudioView.WAVEFORM]
        self.batch_shapes.append(tuple(waveform.shape))
        lengths = batch.lengths(
            FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM)
        )
        return [
            {"semantic_codes": waveform[index, :, : int(length.item())].to(torch.int64)}
            for index, length in enumerate(lengths)
        ]


class _MultiRoleBatchProvider(_Provider):
    def __init__(self):
        super().__init__()
        self.batch_refs: list[tuple[tuple[Role, Modality], ...]] = []

    def call_batch(self, batch):
        refs = tuple(batch.sample)
        self.batch_refs.append(refs)
        return {ref: _batch_codes(batch, ref) for ref in refs}


class _BadBatchProvider(_Provider):
    def call_batch(self, batch):
        return [{"semantic_codes": torch.tensor([[1]])}]


class _SplitOnOomBatchProvider(_BatchProvider):
    def call_batch(self, batch):
        ref = (Role.DEFAULT, Modality.AUDIO)
        waveform, _ = batch.sample[ref].views[AudioView.WAVEFORM]
        self.batch_shapes.append(tuple(waveform.shape))
        if waveform.shape[0] > 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        lengths = batch.lengths(
            FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM)
        )
        return [
            {"semantic_codes": waveform[index, :, : int(length.item())].to(torch.int64)}
            for index, length in enumerate(lengths)
        ]


class _NonOomBatchProvider(_BatchProvider):
    def call_batch(self, batch):
        ref = (Role.DEFAULT, Modality.AUDIO)
        waveform, _ = batch.sample[ref].views[AudioView.WAVEFORM]
        self.batch_shapes.append(tuple(waveform.shape))
        raise RuntimeError("bad batch")


def _batch_codes(batch, ref):
    waveform, _ = batch.sample[ref].views[AudioView.WAVEFORM]
    lengths = batch.lengths(FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM))
    return [
        {"semantic_codes": waveform[index, :, : int(length.item())].to(torch.int64)}
        for index, length in enumerate(lengths)
    ]


class _TTSProvider:
    output = AudioView.WAVEFORM

    def __init__(self, *, offset=0):
        self.offset = offset

    def __call__(self, views):
        return torch.tensor([[float(len(views[TextView.TEXT]) + self.offset)]]), 16000


class _MultiRoleTTSProvider(_TTSProvider):
    def __init__(self):
        super().__init__()
        self.batch_refs: list[tuple[tuple[Role, Modality], ...]] = []

    def call_batch(self, batch):
        refs = tuple(batch.sample)
        self.batch_refs.append(refs)
        return {
            ref: [
                (torch.tensor([[float(len(text))]]), 16000)
                for text in batch.sample[ref].views[TextView.TEXT]
            ]
            for ref in refs
        }


class _ReferenceTTSProvider(_TTSProvider):
    def __init__(self, *, reference_role):
        super().__init__()
        self.reference_role = reference_role
        self.calls = []

    def __call__(self, views):
        waveform, _ = views[AudioView.WAVEFORM]
        self.calls.append((views[TextView.TEXT], float(waveform.sum().item())))
        return super().__call__(views)


class _BadMultiRoleTTSProvider(_TTSProvider):
    def call_batch(self, batch):
        return {
            (Role.DEFAULT, Modality.TEXT): [
                (torch.tensor([[0.0]]), 16000)
                for _ in batch.sample[Role.SOURCE, Modality.TEXT].views[TextView.TEXT]
            ]
        }


class _ASRProvider:
    output = TextView.TEXT

    def __call__(self, views):
        waveform, _ = views[AudioView.WAVEFORM]
        return f"sum={int(waveform.sum().item())}"


@dataclass(frozen=True)
class _DatasetFactory:
    dataset: object

    def __call__(self):
        return self.dataset


@dataclass(frozen=True)
class _StoreDatasetFactory:
    path: Path
    root: Path

    def __call__(self):
        return _store_dataset(self.path, self.root)


@dataclass(frozen=True)
class _ProviderFactory:
    offset: int = 0

    def __call__(self, device: str):
        return _Provider(offset=self.offset)


class _UnpicklableDataset:
    def __init__(self, count: int):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index: int):
        return _audio_sample(torch.tensor([[float(index)]]))

    def __getstate__(self):
        raise TypeError("dataset instance must not be pickled")


@dataclass(frozen=True)
class _UnpicklableDatasetFactory:
    count: int

    def __call__(self):
        return _UnpicklableDataset(self.count)


@dataclass(frozen=True)
class _StaticProviderFactory:
    provider: object

    def __call__(self, device: str):
        return self.provider


@dataclass(frozen=True)
class _TTSProviderFactory:
    def __call__(self, device: str):
        return _TTSProvider()


@dataclass(frozen=True)
class _ASRProviderFactory:
    def __call__(self, device: str):
        return _ASRProvider()


@dataclass(frozen=True)
class _ParallelProviderFactory:
    def __call__(self, device: str):
        return _Provider(offset=100 if device.endswith(":1") else 0)


@dataclass(frozen=True)
class _ParallelTTSProviderFactory:
    def __call__(self, device: str):
        return _TTSProvider(offset=100 if device.endswith(":1") else 0)


class _FailOnceBatchProvider(_BatchProvider):
    def __init__(self, calls: Path):
        super().__init__()
        self.calls = calls

    def call_batch(self, batch):
        ref = (Role.DEFAULT, Modality.AUDIO)
        waveform, _ = batch.sample[ref].views[AudioView.WAVEFORM]
        indexes = [str(int(value.item())) for value in waveform[:, 0, 0]]
        _append_call(self.calls, ",".join(indexes))
        if len(self.calls.read_text(encoding="utf-8").splitlines()) == 2:
            raise RuntimeError("stop after first batch")
        return super().call_batch(batch)


@dataclass(frozen=True)
class _FailOnceBatchProviderFactory:
    calls: Path

    def __call__(self, device: str):
        return _FailOnceBatchProvider(self.calls)


class _FailOnceTTSBatchProvider(_TTSProvider):
    def __init__(self, calls: Path):
        super().__init__()
        self.calls = calls

    def call_batch(self, batch):
        ref = (Role.DEFAULT, Modality.TEXT)
        values = [len(text) for text in batch.sample[ref].views[TextView.TEXT]]
        _append_call(self.calls, ",".join(str(value) for value in values))
        outputs = [(torch.tensor([[float(value)]]), 16000) for value in values]
        if len(self.calls.read_text(encoding="utf-8").splitlines()) == 2:
            raise RuntimeError("stop after first batch")
        return outputs


@dataclass(frozen=True)
class _FailOnceTTSBatchProviderFactory:
    calls: Path

    def __call__(self, device: str):
        return _FailOnceTTSBatchProvider(self.calls)


def _append_call(path: Path, text: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + text + "\n", encoding="utf-8")


def _materializer_logs(home: Path) -> list[Path]:
    logs = sorted((home / "logs").glob("*/materializer/part-*.log"))
    if len(logs) != 2:
        raise AssertionError(f"expected two materializer logs, found: {logs}")
    return logs


if __name__ == "__main__":
    unittest.main()
