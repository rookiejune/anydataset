from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest import mock

import torch

from anydataset import AnyDataset, Source, Spec
from anydataset.dataset.write import DatasetStoreWriter
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioReq,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextView,
)
from anydataset.store import DatasetWriter, StoreLocalBatchSampler, store_local_loader
from anydataset.store.jsonio import read_json, write_json
from anydataset.store.manifest import (
    DatasetManifest,
    SampleManifestEntry,
    STORE_SCHEMA_VERSION,
)
from anydataset.store.manifestio import (
    read_sample_manifest_index,
    read_view_manifest_indexes,
    write_samples_manifest,
)
from anydataset.store.paths import (
    dataset_ready_path,
    samples_parquet_path,
    view_dir,
    view_manifest_parquet_path,
)
from anydataset.store.reader import read_store_dataset, read_store_manifest


class StoreSourceTest(unittest.TestCase):
    def test_dataset_store_writer_validates_views_at_construction(self):
        with self.assertRaisesRegex(TypeError, "views must be a tuple"):
            DatasetStoreWriter("unused", views=[])

    def test_anydataset_reads_dataset_written_by_writer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(
                [
                    _audio_sample(
                        waveform=waveform,
                        label="speech",
                        text="hello",
                    )
                ]
            )

            dataset = AnyDataset(
                Spec(source=Source.STORE, path=str(output), split="train"),
            )
            sample = dataset[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        text = sample[Role.DEFAULT, Modality.TEXT]
        loaded_waveform, sample_rate = audio.views[AudioView.WAVEFORM]
        self.assertTrue(torch.equal(loaded_waveform, waveform))
        self.assertEqual(sample_rate, 4)
        self.assertEqual(audio.meta[AudioMeta.LABEL], "speech")
        self.assertEqual(text.views[TextView.TEXT], "hello")

    def test_read_store_manifest_reads_dataset_json_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(
                [_audio_sample(waveform=waveform)]
            )
            view_manifest_parquet_path(
                output,
                (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),
            ).unlink()

            manifest = read_store_manifest(output)

            self.assertEqual(manifest.dataset_id, "toy-audio")
            self.assertEqual(manifest.sample_count, 1)

    def test_read_store_manifest_rejects_missing_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            manifest = read_json(output / "dataset.json")
            del manifest["schema_version"]
            write_json(output / "dataset.json", manifest)

            with self.assertRaisesRegex(
                ValueError,
                "Unsupported store schema_version: None; expected 2",
            ):
                read_store_manifest(output)

    def test_read_store_manifest_rejects_non_integer_sample_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            manifest = read_json(output / "dataset.json")
            manifest["sample_count"] = 1.0
            write_json(output / "dataset.json", manifest)

            with self.assertRaisesRegex(
                ValueError,
                "sample_count must be a non-negative integer",
            ):
                read_store_manifest(output)

    def test_anydataset_reads_store_shorthand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(
                [_audio_sample(waveform=waveform)]
            )

            dataset = AnyDataset(
                spec=f"store://{output}:train",
            )
            sample = dataset[0]

        loaded_waveform, sample_rate = sample[Role.DEFAULT, Modality.AUDIO].views[
            AudioView.WAVEFORM
        ]
        self.assertTrue(torch.equal(loaded_waveform, waveform))
        self.assertEqual(sample_rate, 4)

    def test_store_source_rejects_requested_split_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = AnyDataset(
                Spec(source=Source.STORE, path=str(output), split="validation"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "split 'train' does not match requested split 'validation'",
            ):
                dataset.prepare()

    def test_store_source_rejects_requested_split_when_manifest_has_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = AnyDataset(
                Spec(source=Source.STORE, path=str(output), split="train"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "split None does not match requested split 'train'",
            ):
                dataset.prepare()

    def test_file_view_is_extracted_to_cache_and_reused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )
            dataset = AnyDataset(
                Spec(source=Source.STORE, path=str(output)),
            )
            dataset.prepare()

            with mock.patch(
                "anydataset.store.reader.os.replace",
                wraps=os.replace,
            ) as replace:
                file_view = Path(
                    dataset[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE]
                )
            cached_dataset = AnyDataset(
                Spec(source=Source.STORE, path=str(output)),
            )
            with mock.patch(
                "anydataset.store.reader.read_payload_bytes",
                side_effect=AssertionError("cache miss"),
            ):
                cached = Path(
                    cached_dataset[0][Role.DEFAULT, Modality.AUDIO].views[
                        AudioView.FILE
                    ]
                )

            self.assertEqual(replace.call_count, 1)
            self.assertTrue(file_view.is_file())
            self.assertTrue(
                file_view.is_relative_to(Path(os.environ["ANYDATASET_HOME"]))
            )
            self.assertFalse((output / ".cache").exists())
            self.assertEqual(file_view.read_bytes(), b"RIFF-data")
            self.assertEqual(cached, file_view)

    def test_file_view_cache_recovers_after_cached_file_is_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )
            dataset = read_store_dataset(output)
            ref = (Role.DEFAULT, Modality.AUDIO)
            cached = Path(dataset[0][ref].views[AudioView.FILE])
            cached.unlink()

            restored = Path(dataset[0][ref].views[AudioView.FILE])

            self.assertEqual(restored, cached)
            self.assertEqual(restored.read_bytes(), b"RIFF-data")

    def test_file_view_cache_separates_roles_for_read_only_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            target = root / "target.wav"
            source.write_bytes(b"SOURCE")
            target.write_bytes(b"TARGET")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="paired-files").write(
                [
                    {
                        (Role.SOURCE, Modality.AUDIO): AudioItem(
                            views={AudioView.FILE: str(source)}
                        ),
                        (Role.TARGET, Modality.AUDIO): AudioItem(
                            views={AudioView.FILE: str(target)}
                        ),
                    }
                ]
            )

            output.chmod(0o555)
            try:
                sample = read_store_dataset(output)[0]
            finally:
                output.chmod(0o755)

            cached_source = Path(
                sample[Role.SOURCE, Modality.AUDIO].views[AudioView.FILE]
            )
            cached_target = Path(
                sample[Role.TARGET, Modality.AUDIO].views[AudioView.FILE]
            )
            self.assertNotEqual(cached_source, cached_target)
            self.assertEqual(cached_source.read_bytes(), b"SOURCE")
            self.assertEqual(cached_target.read_bytes(), b"TARGET")
            self.assertFalse((output / ".cache").exists())

    def test_file_view_cache_changes_when_store_is_rebuilt_at_same_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            output = root / "dataset"
            source.write_bytes(b"OLD")
            DatasetWriter(output, dataset_id="rebuilt-file").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )
            first = Path(
                read_store_dataset(output)[0][Role.DEFAULT, Modality.AUDIO].views[
                    AudioView.FILE
                ]
            )

            shutil.rmtree(output)
            source.write_bytes(b"NEW-CONTENT")
            DatasetWriter(output, dataset_id="rebuilt-file").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )
            second = Path(
                read_store_dataset(output)[0][Role.DEFAULT, Modality.AUDIO].views[
                    AudioView.FILE
                ]
            )

            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b"OLD")
            self.assertEqual(second.read_bytes(), b"NEW-CONTENT")

    def test_file_view_cache_does_not_reuse_stale_open_shard_after_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            output = root / "dataset"
            source.write_bytes(b"OLD")
            DatasetWriter(output, dataset_id="rebuilt-file").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )
            old_dataset = read_store_dataset(output)
            ref = (Role.DEFAULT, Modality.AUDIO)
            old_cache = Path(old_dataset[0][ref].views[AudioView.FILE])

            shutil.rmtree(output)
            source.write_bytes(b"NEW-CONTENT")
            DatasetWriter(output, dataset_id="rebuilt-file").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )
            old_cache.unlink()
            refreshed = Path(old_dataset[0][ref].views[AudioView.FILE])
            new_dataset = read_store_dataset(output)
            reused = Path(new_dataset[0][ref].views[AudioView.FILE])

            self.assertEqual(refreshed.read_bytes(), b"NEW-CONTENT")
            self.assertEqual(reused, refreshed)
            self.assertEqual(reused.read_bytes(), b"NEW-CONTENT")

    def test_reader_reuses_open_payload_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [
                    _audio_sample(waveform=torch.tensor([[1.0]])),
                    _audio_sample(waveform=torch.tensor([[2.0]])),
                ]
            )
            dataset = read_store_dataset(output)

            with mock.patch(
                "anydataset.store.payload.tarfile.open",
                wraps=__import__("tarfile").open,
            ) as open_tar:
                dataset[1]
                dataset[0]

            self.assertEqual(open_tar.call_count, 1)

    def test_reader_evicts_old_payload_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=1,
            ).write(
                [
                    _audio_sample(waveform=torch.tensor([[1.0]])),
                    _audio_sample(waveform=torch.tensor([[2.0]])),
                ]
            )
            dataset = read_store_dataset(output)
            dataset._payloads.max_open_shards = 1

            with mock.patch(
                "anydataset.store.payload.tarfile.open",
                wraps=__import__("tarfile").open,
            ) as open_tar:
                dataset[0]
                dataset[1]
                dataset[0]

            self.assertEqual(open_tar.call_count, 3)

    def test_store_local_batch_sampler_keeps_batches_inside_payload_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=2,
            ).write(
                [
                    _audio_sample(waveform=torch.tensor([[float(index)]]))
                    for index in range(5)
                ]
            )
            dataset = read_store_dataset(output)

            sampler = StoreLocalBatchSampler(dataset, batch_size=3)

            self.assertEqual(list(sampler), [[0, 1], [2, 3], [4]])
            self.assertEqual(len(sampler), 3)

    def test_store_local_batch_sampler_shuffle_preserves_payload_locality(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=2,
            ).write(
                [
                    _audio_sample(waveform=torch.tensor([[float(index)]]))
                    for index in range(6)
                ]
            )
            dataset = read_store_dataset(output)
            sampler = StoreLocalBatchSampler(
                dataset,
                batch_size=2,
                shuffle=True,
                seed=13,
            )
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)

            batches = list(sampler)

            self.assertEqual(
                sorted(index for batch in batches for index in batch),
                list(range(6)),
            )
            for batch in batches:
                shards = {
                    dataset.views[view].entries_by_index[index].shard
                    for index in batch
                }
                self.assertEqual(len(shards), 1)

    def test_store_local_batch_sampler_splits_distributed_ranks_by_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=2,
            ).write(
                [
                    _audio_sample(waveform=torch.tensor([[float(index)]]))
                    for index in range(6)
                ]
            )
            dataset = read_store_dataset(output)

            sampler = StoreLocalBatchSampler(
                dataset,
                batch_size=3,
                num_replicas=2,
                rank=1,
            )

            self.assertEqual(list(sampler), [[2, 3]])
            self.assertEqual(len(sampler), 1)

    def test_store_local_loader_owns_sampler_kwargs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = read_store_dataset(output)

            with self.assertRaisesRegex(ValueError, "owns loader kwargs"):
                store_local_loader(dataset, batch_size=1, shuffle=True, sampler=[])

    def test_store_local_batch_sampler_rejects_string_view_refs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = read_store_dataset(output)

            with self.assertRaisesRegex(TypeError, "role must be a Role"):
                StoreLocalBatchSampler(
                    dataset,
                    batch_size=1,
                    views=(("default", Modality.AUDIO, AudioView.WAVEFORM),),
                )

    def test_reader_discovers_all_views_without_preloading_indexes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="multi-view").write(
                [
                    _audio_sample(
                        waveform=torch.tensor([[1.0, 2.0]]),
                        file=str(source),
                        sample_rate=16000,
                    )
                ]
            )

            with mock.patch(
                "anydataset.store.reader.read_view_manifest_indexes",
                side_effect=AssertionError("view index loaded"),
            ):
                dataset = read_store_dataset(output)

        self.assertEqual(
            set(dataset.views),
            {
                (Role.DEFAULT, Modality.AUDIO, AudioView.FILE),
                (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),
            },
        )
        self.assertEqual(len(dataset.samples), 1)

    def test_reader_does_not_load_sample_rows_when_opening_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )

            with mock.patch(
                "anydataset.store.reader.read_samples_manifest_row_group",
                side_effect=AssertionError("sample rows loaded"),
            ):
                dataset = read_store_dataset(output)

        self.assertEqual(len(dataset), 1)

    def test_reader_loads_only_requested_sample_row_group(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [
                    _audio_sample(waveform=torch.tensor([[1.0]])),
                    _audio_sample(waveform=torch.tensor([[2.0]])),
                ]
            )
            _rewrite_sample_manifest_one_row_per_group(output)
            dataset = read_store_dataset(output)

            with mock.patch(
                "anydataset.store.reader.read_samples_manifest_row_group",
                wraps=__import__(
                    "anydataset.store.reader",
                    fromlist=["read_samples_manifest_row_group"],
                ).read_samples_manifest_row_group,
            ) as read_group:
                dataset.samples[1]

            read_group.assert_called_once()
            self.assertEqual(read_group.call_args.args[1], 1)

    def test_reader_can_preload_all_view_indexes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="multi-view").write(
                [
                    _audio_sample(
                        waveform=torch.tensor([[1.0, 2.0]]),
                        file=str(source),
                        sample_rate=16000,
                    )
                ]
            )

            dataset = read_store_dataset(output, preload=True)

        self.assertEqual(
            set(dataset.views),
            {
                (Role.DEFAULT, Modality.AUDIO, AudioView.FILE),
                (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),
            },
        )
        self.assertEqual(len(dataset.views._cache), 2)

    def test_reader_selects_requested_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            waveform = torch.tensor([[1.0, 2.0]])
            DatasetWriter(output, dataset_id="multi-view").write(
                [
                    _audio_sample(
                        waveform=waveform,
                        file=str(source),
                        sample_rate=16000,
                    )
                ]
            )
            file_view = (Role.DEFAULT, Modality.AUDIO, AudioView.FILE)
            waveform_view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            view_manifest_parquet_path(output, file_view).write_bytes(b"not parquet")

            dataset = read_store_dataset(output, views=(waveform_view,))
            sample = dataset[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(set(audio.views), {AudioView.WAVEFORM})
        self.assertTrue(torch.equal(audio.views[AudioView.WAVEFORM][0], waveform))

    def test_store_dataset_merge_adds_overlay_views_logically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(
                [
                    _audio_sample(
                        waveform=waveform,
                        label="speech",
                        text="hello",
                    )
                ]
            )
            store = read_store_dataset(output)
            overlay = [
                {
                    (Role.DEFAULT, Modality.AUDIO): AudioItem(
                        views={
                            AudioView.LONGCAT: {
                                "semantic_codes": torch.tensor([[1, 2, 3]])
                            }
                        },
                        meta={AudioMeta.LABEL: "speech"},
                    )
                }
            ]

            merged = store.merge(overlay)
            sample = merged[0]
            stored = read_store_dataset(output)

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        text = sample[Role.DEFAULT, Modality.TEXT]
        self.assertEqual(set(audio.views), {AudioView.WAVEFORM, AudioView.LONGCAT})
        self.assertTrue(torch.equal(audio.views[AudioView.WAVEFORM][0], waveform))
        self.assertTrue(
            torch.equal(
                audio.views[AudioView.LONGCAT]["semantic_codes"],
                torch.tensor([[1, 2, 3]]),
            )
        )
        self.assertEqual(audio.meta[AudioMeta.LABEL], "speech")
        self.assertEqual(text.views[TextView.TEXT], "hello")
        self.assertEqual(
            set(stored.views),
            {
                (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),
                (Role.DEFAULT, Modality.TEXT, TextView.TEXT),
            },
        )

    def test_anydataset_merge_returns_logical_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={
                                AudioView.LONGCAT: {
                                    "semantic_codes": torch.tensor([[1, 2, 3]])
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

            dataset = AnyDataset(
                f"store://{output}:train",
            ).merge(source)
            sample = dataset[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(set(audio.views), {AudioView.WAVEFORM, AudioView.LONGCAT})
        self.assertTrue(torch.equal(audio.views[AudioView.WAVEFORM][0], waveform))
        self.assertEqual(sample[Role.DEFAULT, Modality.TEXT].views[TextView.TEXT], "hello")

    def test_merge_requires_map_style_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = read_store_dataset(output)

            with self.assertRaises(TypeError):
                dataset.merge(iter([]))

    def test_store_dataset_merge_rejects_view_conflicts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(output, dataset_id="toy-audio", split="train").write(
                [_audio_sample(waveform=waveform)]
            )
            store = read_store_dataset(output)
            overlay = [_audio_sample(waveform=torch.tensor([[4.0, 5.0, 6.0]]))]

            merged = store.merge(overlay)

            with self.assertRaises(ValueError):
                merged[0]
            stored = read_store_dataset(output)

        self.assertEqual(
            set(stored.views),
            {(Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)},
        )

    def test_merged_dataset_write_materializes_full_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            delta = root / "delta"
            output = root / "merged"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            DatasetWriter(source, dataset_id="source", split="train").write(
                [
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
            )
            DatasetWriter(delta, dataset_id="delta", split="train").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={
                                AudioView.LONGCAT: {
                                    "semantic_codes": torch.tensor([[1, 2, 3]])
                                }
                            },
                            meta={AudioMeta.LABEL: "speech"},
                        )
                    }
                ]
            )

            read_store_dataset(source).merge(read_store_dataset(delta)).write(
                output,
                dataset_id="merged",
                split="train",
            )
            sample = read_store_dataset(output)[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(set(audio.views), {AudioView.WAVEFORM, AudioView.LONGCAT})
        self.assertTrue(torch.equal(audio.views[AudioView.WAVEFORM][0], waveform))
        self.assertEqual(sample[Role.DEFAULT, Modality.TEXT].views[TextView.TEXT], "hello")

    def test_dataset_write_supports_parallel_parts_and_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "parallel"
            DatasetStoreWriter(
                output,
                dataset_id="parallel",
                split="train",
                num_shards=2,
                num_workers=2,
            ).write(
                dataset_factory=_RangeAudioFactory(5),
            )
            dataset = read_store_dataset(output)
            self.assertEqual(len(dataset), 5)
            values = [
                float(
                    dataset[index][Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.WAVEFORM][0][0, 0]
                )
                for index in range(len(dataset))
            ]
            self.assertEqual(values, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_dataset_write_prepares_sharded_csv_before_loader_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "csv" / "shard_0"
            source.mkdir(parents=True)
            (source / "0.csv").write_text("value\n0\n", encoding="utf-8")
            (source / "1.csv").write_text("value\n1\n", encoding="utf-8")
            output = root / "parallel-csv"

            DatasetStoreWriter(
                output,
                dataset_id="parallel-csv",
                num_workers=1,
            ).write(dataset_factory=_CsvAudioFactory(root / "csv"))

            dataset = read_store_dataset(output)
            values = [
                float(
                    dataset[index][Role.DEFAULT, Modality.AUDIO]
                    .views[AudioView.WAVEFORM][0][0, 0]
                )
                for index in range(len(dataset))
            ]
            self.assertEqual(values, [0.0, 1.0])

    def test_parallel_write_cleans_workers_after_partial_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            context = mock.Mock()
            first = mock.Mock()
            first.is_alive.return_value = True
            second = mock.Mock()
            second.start.side_effect = RuntimeError("start failed")
            context.Process.side_effect = (first, second)
            writer = DatasetStoreWriter(
                Path(tmpdir) / "output",
                num_shards=2,
            )

            with (
                mock.patch(
                    "anydataset.dataset.write.multiprocessing_context",
                    return_value=context,
                ),
                mock.patch("anydataset.dataset.write.free_port", return_value="1234"),
            ):
                with self.assertRaisesRegex(RuntimeError, "start failed"):
                    writer._run_parts(_RangeAudioFactory(1), Path(tmpdir) / "parts")

            first.terminate.assert_called_once_with()
            first.join.assert_called_once_with()
            second.join.assert_not_called()

    def test_schema_selects_requested_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )
            dataset = AnyDataset(
                Spec(source=Source.STORE, path=str(output)),
            )
            schema = {
                (Role.DEFAULT, Modality.AUDIO): AudioReq(
                    views=frozenset({AudioView.FILE}),
                )
            }

            resolved = AnyDataset.resolve_sample(dataset[0], schema)

        self.assertEqual(set(resolved[Role.DEFAULT, Modality.AUDIO].views), {AudioView.FILE})

    def test_reader_rejects_incomplete_view_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            _write_empty_dataset(output)
            view_path = view_dir(output, (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM))
            view_path.mkdir(parents=True)
            (view_path / ".ready").touch()

            with self.assertRaises(FileNotFoundError):
                read_store_dataset(output)

    def test_reader_requires_dataset_ready_marker_to_be_a_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            _write_empty_dataset(output)
            dataset_ready_path(output).unlink()
            dataset_ready_path(output).mkdir()

            with self.assertRaisesRegex(ValueError, "dataset is not ready"):
                read_store_dataset(output)

    def test_reader_rejects_invalid_view_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            _write_empty_dataset(output)
            view_path = output / "default" / "audio" / "not-a-view"
            view_path.mkdir(parents=True)
            (view_path / "manifest.parquet").write_bytes(b"not-parquet")
            (view_path / ".ready").touch()

            with self.assertRaises(ValueError):
                read_store_dataset(output, preload=True)

    def test_reader_rejects_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            output.mkdir()
            write_json(
                output / "dataset.json",
                asdict(
                    DatasetManifest(
                        dataset_id="toy-audio",
                        schema_version=STORE_SCHEMA_VERSION,
                        sample_count=2,
                    )
                ),
            )
            write_samples_manifest(
                output,
                [
                    SampleManifestEntry(sample_id="same", sample_index=0),
                    SampleManifestEntry(sample_id="same", sample_index=1),
                ],
            )
            dataset_ready_path(output).touch()

            with self.assertRaises(ValueError):
                read_store_dataset(output)

    def test_reader_reuses_sample_index_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )

            with mock.patch(
                "anydataset.store.reader.read_sample_manifest_index",
                wraps=read_sample_manifest_index,
            ) as read_index:
                for _ in range(3):
                    read_store_dataset(output)

            self.assertEqual(read_index.call_count, 1)

    def test_reader_revalidates_rewritten_sample_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [
                    _audio_sample(waveform=torch.tensor([[1.0]])),
                    _audio_sample(waveform=torch.tensor([[2.0]])),
                ]
            )
            read_store_dataset(output)

            write_samples_manifest(
                output,
                [
                    SampleManifestEntry(sample_id="same", sample_index=0),
                    SampleManifestEntry(sample_id="same", sample_index=1),
                ],
            )

            with self.assertRaisesRegex(ValueError, "Duplicate sample_id 'same'"):
                read_store_dataset(output)

    def test_reader_rejects_sample_manifest_changed_during_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )

            def changing_index(root):
                yield from read_sample_manifest_index(root)
                path = samples_parquet_path(root)
                path.write_bytes(path.read_bytes() + b"changed")

            with mock.patch(
                "anydataset.store.reader.read_sample_manifest_index",
                side_effect=changing_index,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Sample manifest changed while validating index",
                ):
                    read_store_dataset(output)

    def test_reader_rejects_view_manifest_with_missing_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [
                    _audio_sample(waveform=torch.tensor([[1.0]])),
                    _audio_sample(waveform=torch.tensor([[2.0]])),
                ]
            )
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            _drop_last_parquet_row(view_manifest_parquet_path(output, view))

            with self.assertRaises(ValueError):
                read_store_dataset(output, preload=True)

    def test_reader_rejects_view_manifest_changed_while_loading_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            path = view_manifest_parquet_path(output, view)
            dataset = read_store_dataset(output)

            def changing_indexes(root, selected_view):
                yield from read_view_manifest_indexes(root, selected_view)
                path.write_bytes(path.read_bytes() + b"changed")

            with mock.patch(
                "anydataset.store.reader.read_view_manifest_indexes",
                side_effect=changing_indexes,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "View manifest changed while loading index",
                ):
                    dataset.views[view]

    def test_reader_rejects_legacy_view_manifest_without_sample_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            _rewrite_view_manifest_as_legacy(output, view)
            dataset = read_store_dataset(output)

            with self.assertRaisesRegex(
                ValueError,
                "Store schema 2 view manifest schema does not match expected fields",
            ):
                dataset[0]

    def test_reader_rejects_sample_manifest_with_wrong_column_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            _rewrite_sample_indexes_as_float(output)

            with self.assertRaisesRegex(
                ValueError,
                "Store schema 2 sample manifest schema does not match expected fields",
            ):
                read_store_dataset(output)


def _audio_sample(
    waveform=None,
    *,
    file=None,
    sample_rate: int = 4,
    label=None,
    text: str | None = None,
):
    views = {}
    if waveform is not None:
        views[AudioView.WAVEFORM] = (waveform, sample_rate)
    if file is not None:
        views[AudioView.FILE] = file
    meta = {}
    if label is not None:
        meta[AudioMeta.LABEL] = label
    sample = {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views=views,
            meta=meta,
        )
    }
    if text is not None:
        sample[(Role.DEFAULT, Modality.TEXT)] = TextItem(
            views={TextView.TEXT: text}
        )
    return sample


def _write_empty_dataset(path: Path) -> None:
    path.mkdir()
    write_json(
        path / "dataset.json",
        asdict(
            DatasetManifest(
                dataset_id="toy-audio",
                schema_version=STORE_SCHEMA_VERSION,
                sample_count=0,
            )
        ),
    )
    write_samples_manifest(path, [])
    dataset_ready_path(path).touch()


class _RangeAudioDataset:
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        return _audio_sample(waveform=[[float(index)]])


class _RangeAudioFactory:
    def __init__(self, count: int) -> None:
        self.count = count

    def __call__(self):
        return _RangeAudioDataset(self.count)


@dataclass(frozen=True)
class _CsvAudioFactory:
    root: Path

    def __call__(self):
        return AnyDataset(
            Spec(source="sharded_csv", path=str(self.root)),
            parse_fn=_csv_audio_sample,
        )


def _csv_audio_sample(row):
    return _audio_sample(waveform=[[float(row["value"])]])


def _drop_last_parquet_row(path: Path) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    pq.write_table(table.slice(0, table.num_rows - 1), path)


def _rewrite_sample_manifest_one_row_per_group(root: Path) -> None:
    import pyarrow.parquet as pq

    path = root / "samples.parquet"
    table = pq.read_table(path)
    writer = pq.ParquetWriter(path.with_suffix(".tmp"), table.schema)
    try:
        for index in range(table.num_rows):
            writer.write_table(table.slice(index, 1))
    finally:
        writer.close()
    path.with_suffix(".tmp").replace(path)


def _rewrite_view_manifest_as_legacy(root: Path, view) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = view_manifest_parquet_path(root, view)
    table = pq.read_table(path).drop(["sample_index"])
    sample_ids = [sample_id for _index, sample_id in read_sample_manifest_index(root)]
    table = table.append_column("sample_id", pa.array(sample_ids))
    pq.write_table(table, path)


def _rewrite_sample_indexes_as_float(root: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = samples_parquet_path(root)
    table = pq.read_table(path)
    column = table.schema.get_field_index("sample_index")
    indexes = pa.array(table["sample_index"].to_pylist(), type=pa.float64())
    pq.write_table(table.set_column(column, "sample_index", indexes), path)


if __name__ == "__main__":
    unittest.main()
