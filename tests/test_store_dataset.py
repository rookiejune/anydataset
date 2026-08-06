from __future__ import annotations

import os
import pickle
import shutil
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

import anydataset.store
import anydataset.store.reader as store_reader
import anydataset.store.payload.groups as payload_groups_module
import torch

from anydataset import AnyDataset, Source, Spec
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioReq,
    AudioView,
    FileBytes,
    ImageItem,
    ImageView,
    Modality,
    Role,
    TextItem,
    TextView,
)
from anydataset.store import DatasetWriter
from anydataset.store.jsonio import read_json, write_json
from anydataset.store.manifest.schema import (
    DatasetManifest,
    SampleManifestEntry,
    STORE_SCHEMA_VERSION,
    dataset_manifest_dict,
)
from anydataset.store.manifest.io import (
    read_sample_manifest_index,
    read_view_manifest_indexes,
    write_samples_manifest,
)
from anydataset.store.payload.groups import PayloadGroupCache, read_payload_groups
from anydataset.store.payload.files import files_dir
from anydataset.store.paths import (
    dataset_ready_path,
    payload_groups_path,
    samples_parquet_path,
    view_dir,
    view_manifest_parquet_path,
)
from anydataset.store.reader import (
    SampleManifestSequence,
    StoreDataset,
    StoreView,
    StoreViews,
    ViewEntryIndex,
    read_store_dataset,
    read_store_manifest,
)


@dataclass(frozen=True)
class _CustomPayload:
    value: str


class StoreSourceTest(unittest.TestCase):
    def test_dataset_store_writer_validates_views_at_construction(self):
        with self.assertRaisesRegex(TypeError, "views must be a tuple"):
            DatasetWriter("unused", views=[])

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

    def test_store_reader_rejects_custom_payload_objects_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="custom-payload").write(
                [
                    {
                        (Role.DEFAULT, Modality.IMAGE): ImageItem(
                            views={ImageView.PIXEL: _CustomPayload("unsafe")}
                        )
                    }
                ]
            )

            dataset = read_store_dataset(output)

            with self.assertRaisesRegex(ValueError, "safe tensor-only"):
                dataset[0]

    def test_store_reader_allows_custom_payload_objects_when_trusted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="custom-payload").write(
                [
                    {
                        (Role.DEFAULT, Modality.IMAGE): ImageItem(
                            views={ImageView.PIXEL: _CustomPayload("trusted")}
                        )
                    }
                ]
            )

            dataset = read_store_dataset(output, unsafe_pickle_payloads=True)
            sample = dataset[0]

        self.assertEqual(
            sample[Role.DEFAULT, Modality.IMAGE].views[ImageView.PIXEL],
            _CustomPayload("trusted"),
        )

    def test_store_source_accepts_explicit_unsafe_pickle_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="custom-payload").write(
                [
                    {
                        (Role.DEFAULT, Modality.IMAGE): ImageItem(
                            views={ImageView.PIXEL: _CustomPayload("source")}
                        )
                    }
                ]
            )
            dataset = AnyDataset(
                Spec(
                    source=Source.STORE,
                    path=str(output),
                    load_options={"unsafe_pickle_payloads": True},
                )
            )

            sample = dataset[0]

        self.assertEqual(
            sample[Role.DEFAULT, Modality.IMAGE].views[ImageView.PIXEL],
            _CustomPayload("source"),
        )

    def test_store_source_rejects_non_bool_unsafe_pickle_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = AnyDataset(
                Spec(
                    source=Source.STORE,
                    path=str(output),
                    load_options={"unsafe_pickle_payloads": "true"},
                )
            )

            with self.assertRaisesRegex(TypeError, "must be a boolean"):
                dataset.prepare()

    def test_anydataset_from_store_accepts_explicit_unsafe_pickle_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="custom-payload").write(
                [
                    {
                        (Role.DEFAULT, Modality.IMAGE): ImageItem(
                            views={ImageView.PIXEL: _CustomPayload("from-store")}
                        )
                    }
                ]
            )
            dataset = AnyDataset.from_store(output, unsafe_pickle_payloads=True)

            sample = dataset[0]

        self.assertEqual(
            sample[Role.DEFAULT, Modality.IMAGE].views[ImageView.PIXEL],
            _CustomPayload("from-store"),
        )

    def test_anydataset_from_store_rejects_non_bool_unsafe_pickle_payloads(self):
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            AnyDataset.from_store("unused", unsafe_pickle_payloads="true")  # type: ignore[arg-type]

    def test_anydataset_from_store_validates_file_mode(self):
        with self.assertRaisesRegex(TypeError, "file_mode must be a string"):
            AnyDataset.from_store("unused", file_mode=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "'path' or 'bytes'"):
            AnyDataset.from_store("unused", file_mode="stream")  # type: ignore[arg-type]

    def test_store_file_mode_is_operational_spec_state(self):
        base = Spec(source=Source.STORE, path="/data/store")
        in_memory = Spec(
            source=Source.STORE,
            path="/data/store",
            load_options={"file_mode": "bytes"},
        )

        self.assertEqual(base.id, in_memory.id)
        self.assertNotIn("file_mode", in_memory.to_dict()["load_options"])

    def test_store_dataset_cost_row_is_manifest_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]), text="hello")]
            )
            dataset = read_store_dataset(output)

            row = dataset.cost_row(0)

        self.assertIsInstance(row, SampleManifestEntry)
        self.assertEqual(row.sample_index, 0)
        self.assertTrue(row.sample_id)

    def test_store_dataset_getitems_sorts_physical_reads_and_restores_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="batch-read").write(
                [
                    _audio_sample(waveform=torch.tensor([[float(index)]]))
                    for index in range(4)
                ]
            )
            dataset = read_store_dataset(output)

            with mock.patch.object(
                store_reader,
                "_sample_for_entry",
                wraps=store_reader._sample_for_entry,
            ) as read:
                batch = dataset.__getitems__([2, 0, 2, -1, 1])

        self.assertEqual(
            [call.args[1] for call in read.call_args_list],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [
                sample[Role.DEFAULT, Modality.AUDIO]
                .views[AudioView.WAVEFORM][0]
                .item()
                for sample in batch
            ],
            [2.0, 0.0, 2.0, 3.0, 1.0],
        )

    def test_store_dataset_getitems_validates_indexes_before_reading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="batch-read-errors").write(
                [_audio_sample(waveform=torch.tensor([[0.0]]))]
            )
            dataset = read_store_dataset(output)

            with mock.patch.object(
                store_reader,
                "_sample_for_entry",
                wraps=store_reader._sample_for_entry,
            ) as read:
                with self.assertRaisesRegex(TypeError, "must be an integer"):
                    dataset.__getitems__([0, True])
                self.assertEqual(read.call_count, 0)

                with self.assertRaisesRegex(IndexError, "out of range"):
                    dataset.__getitems__([0, 1])
                self.assertEqual(read.call_count, 0)

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

    def test_reader_rejects_schema_v2_store_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            manifest = read_json(output / "dataset.json")
            manifest["schema_version"] = 2
            del manifest["provenance"]
            write_json(output / "dataset.json", manifest)

            with self.assertRaisesRegex(ValueError, "schema_version 2 is legacy"):
                read_store_dataset(output)

    def test_reader_warns_for_schema_v2_store_with_explicit_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            manifest = read_json(output / "dataset.json")
            manifest["schema_version"] = 2
            del manifest["provenance"]
            write_json(output / "dataset.json", manifest)

            with self.assertWarnsRegex(RuntimeWarning, "schema_version 2"):
                dataset = read_store_dataset(output, legacy_policy="warn")

            self.assertEqual(dataset.manifest.schema_version, 2)
            self.assertEqual(dataset.manifest.provenance, {})

    def test_reader_allows_schema_v2_store_with_explicit_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            manifest = read_json(output / "dataset.json")
            manifest["schema_version"] = 2
            del manifest["provenance"]
            write_json(output / "dataset.json", manifest)

            with mock.patch("warnings.warn") as warn:
                dataset = read_store_dataset(output, legacy_policy="allow")

            warn.assert_not_called()
            self.assertEqual(dataset.manifest.schema_version, 2)
            self.assertEqual(dataset.manifest.provenance, {})

    def test_store_source_allows_schema_v2_with_explicit_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            manifest = read_json(output / "dataset.json")
            manifest["schema_version"] = 2
            del manifest["provenance"]
            write_json(output / "dataset.json", manifest)

            dataset = AnyDataset(
                Spec(
                    source=Source.STORE,
                    path=str(output),
                    load_options={"legacy_policy": "allow"},
                )
            )
            sample = dataset[0]

        self.assertTrue(
            torch.equal(
                sample[Role.DEFAULT, Modality.AUDIO].views[AudioView.WAVEFORM][0],
                torch.tensor([[1.0]]),
            )
        )

    def test_anydataset_from_store_allows_schema_v2_with_explicit_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            manifest = read_json(output / "dataset.json")
            manifest["schema_version"] = 2
            del manifest["provenance"]
            write_json(output / "dataset.json", manifest)

            dataset = AnyDataset.from_store(output, legacy_policy="allow")
            sample = dataset[0]

        self.assertTrue(
            torch.equal(
                sample[Role.DEFAULT, Modality.AUDIO].views[AudioView.WAVEFORM][0],
                torch.tensor([[1.0]]),
            )
        )

    def test_reader_rejects_schema_v2_store_with_explicit_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            manifest = read_json(output / "dataset.json")
            manifest["schema_version"] = 2
            del manifest["provenance"]
            write_json(output / "dataset.json", manifest)

            with self.assertRaisesRegex(ValueError, "schema_version 2 is legacy"):
                read_store_dataset(output, legacy_policy="reject")

    def test_reader_rejects_invalid_legacy_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            manifest = read_json(output / "dataset.json")
            manifest["schema_version"] = 2
            del manifest["provenance"]
            write_json(output / "dataset.json", manifest)

            with self.assertRaisesRegex(ValueError, "legacy_policy"):
                read_store_manifest(output, legacy_policy="silent")

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
                "Unsupported store schema_version: None; expected 2 or 3",
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
            opened = []

            def read_tracked_store(*args, **kwargs):
                store = read_store_dataset(*args, **kwargs)
                opened.append(store)
                return store

            with mock.patch(
                "anydataset.dataset.source.store.read_store_dataset",
                side_effect=read_tracked_store,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "split 'train' does not match requested split 'validation'",
                ):
                    dataset.prepare()

            self.assertEqual(len(opened), 1)
            self.assertTrue(opened[0].closed)

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
                "anydataset._io.files.os.replace",
                wraps=os.replace,
            ) as replace, mock.patch.object(
                store_reader,
                "atomic_write_bytes",
                wraps=store_reader.atomic_write_bytes,
            ) as atomic_write:
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
            atomic_write.assert_called_once()
            self.assertFalse(atomic_write.call_args.kwargs["durable"])
            self.assertTrue(file_view.is_file())
            self.assertTrue(
                file_view.is_relative_to(Path(os.environ["ANYDATASET_HOME"]))
            )
            self.assertFalse((output / ".cache").exists())
            self.assertEqual(file_view.read_bytes(), b"RIFF-data")
            self.assertEqual(cached, file_view)

    def test_file_view_bytes_mode_avoids_cache_and_lease(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.flac"
            source.write_bytes(b"fLaC-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )

            with mock.patch.object(
                store_reader,
                "lease_store_files",
                side_effect=AssertionError("bytes mode acquired a file lease"),
            ):
                dataset = read_store_dataset(output, file_mode="bytes")
                value = dataset[0][Role.DEFAULT, Modality.AUDIO].views[
                    AudioView.FILE
                ]

            self.assertEqual(value, FileBytes(b"fLaC-data", ".flac"))
            self.assertIsNone(dataset._file_lease)
            self.assertFalse(files_dir(output).exists())
            dataset.close()

    def test_store_source_and_from_store_support_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.ogg"
            source.write_bytes(b"OggS-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )

            direct = AnyDataset(
                Spec(
                    source=Source.STORE,
                    path=str(output),
                    load_options={"file_mode": "bytes"},
                )
            )
            from_store = AnyDataset.from_store(output, file_mode="bytes")

            self.assertEqual(
                direct[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE],
                FileBytes(b"OggS-data", ".ogg"),
            )
            self.assertEqual(
                from_store[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE],
                FileBytes(b"OggS-data", ".ogg"),
            )
            self.assertEqual(from_store.spec.load_options["file_mode"], "bytes")

    def test_read_store_dataset_validates_file_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            _write_empty_dataset(output)

            with self.assertRaisesRegex(TypeError, "file_mode must be a string"):
                read_store_dataset(output, file_mode=True)  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "'path' or 'bytes'"):
                read_store_dataset(output, file_mode="stream")  # type: ignore[arg-type]

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
                "anydataset.store.payload.archive.tarfile.open",
                wraps=__import__("tarfile").open,
            ) as open_tar:
                dataset[1]
                dataset[0]

            self.assertEqual(open_tar.call_count, 1)

    def test_store_dataset_close_releases_reader_resources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = read_store_dataset(output)
            dataset[0]

            self.assertFalse(dataset.closed)
            self.assertTrue(dataset._payloads._archives)
            self.assertTrue(dataset._manifest_cache._files)

            dataset.close()
            dataset.close()

            self.assertTrue(dataset.closed)
            self.assertFalse(dataset._payloads._archives)
            self.assertFalse(dataset._manifest_cache._files)
            with self.assertRaisesRegex(RuntimeError, "StoreDataset is closed"):
                dataset[0]

    def test_store_dataset_context_manager_closes_resources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )

            with read_store_dataset(output) as dataset:
                dataset[0]
                self.assertFalse(dataset.closed)

            self.assertTrue(dataset.closed)

    def test_store_dataset_restores_pre_manifest_cache_pickle_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = read_store_dataset(output)
            view_ref = next(iter(dataset.views))
            store_view = dataset.views[view_ref]

            samples_state = dataset.samples.__getstate__()
            samples_state.pop("manifest_cache")
            samples = SampleManifestSequence.__new__(SampleManifestSequence)
            samples.__setstate__(samples_state)

            index_state = store_view.entries_by_index.__getstate__()
            index_state.pop("manifest_cache")
            index = ViewEntryIndex.__new__(ViewEntryIndex)
            index.__setstate__(index_state)

            views_state = dict(dataset.views.__dict__)
            views_state.pop("_manifest_cache")
            views_state["samples"] = samples
            views_state["_cache"] = {view_ref: StoreView(view_ref, index)}
            views = StoreViews.__new__(StoreViews)
            views.__setstate__(views_state)

            dataset_state = dataset.__getstate__()
            dataset_state.pop("pickle_schema_version")
            dataset_state.pop("_manifest_cache")
            dataset_state.pop("_resource_state")
            dataset_state.pop("_unsafe_pickle_payloads")
            dataset_state.pop("_file_mode")
            dataset_state["samples"] = samples
            dataset_state["views"] = views
            restored = StoreDataset.__new__(StoreDataset)
            restored.__setstate__(dataset_state)

            waveform, sample_rate = restored[0][
                Role.DEFAULT, Modality.AUDIO
            ].views[AudioView.WAVEFORM]

            self.assertTrue(torch.equal(waveform, torch.tensor([[1.0]])))
            self.assertEqual(sample_rate, 4)
            self.assertIs(restored.samples._manifest_cache, restored._manifest_cache)
            self.assertIs(restored.views._manifest_cache, restored._manifest_cache)
            self.assertIs(
                index._manifest_cache,
                restored._manifest_cache,
            )
            restored.close()

    def test_store_dataset_pickle_state_is_explicit_and_versioned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = read_store_dataset(output)

            state = dataset.__getstate__()

            self.assertEqual(state["pickle_schema_version"], 2)
            self.assertEqual(
                set(state),
                {
                    "pickle_schema_version",
                    "root",
                    "manifest",
                    "samples",
                    "views",
                    "_file_lease",
                    "_payloads",
                    "_unsafe_pickle_payloads",
                    "_file_mode",
                    "_payload_group_cache",
                    "_manifest_cache",
                    "_resource_state",
                },
            )
            dataset.close()

    def test_store_dataset_pickle_preserves_bytes_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.flac"
            source.write_bytes(b"fLaC-data")
            output = root / "dataset"
            DatasetWriter(output, dataset_id="file-audio").write(
                [_audio_sample(file=str(source), sample_rate=16000)]
            )
            dataset = read_store_dataset(output, file_mode="bytes")

            restored = pickle.loads(pickle.dumps(dataset))
            try:
                self.assertEqual(restored._file_mode, "bytes")
                self.assertIsNone(restored._file_lease)
                self.assertEqual(
                    restored[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE],
                    FileBytes(b"fLaC-data", ".flac"),
                )
            finally:
                dataset.close()
                restored.close()

    def test_store_dataset_migrates_v1_pickle_to_path_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = read_store_dataset(output)
            state = dataset.__getstate__()
            state["pickle_schema_version"] = 1
            state.pop("_file_mode")
            restored = StoreDataset.__new__(StoreDataset)

            restored.__setstate__(state)

            self.assertEqual(restored._file_mode, "path")
            dataset.close()
            restored.close()

    def test_store_dataset_direct_pickle_preserves_selected_views(self):
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
            waveform_view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            dataset = read_store_dataset(output, views=(waveform_view,))

            restored = pickle.loads(pickle.dumps(dataset))
            try:
                audio = restored[0][Role.DEFAULT, Modality.AUDIO]
                self.assertEqual(set(audio.views), {AudioView.WAVEFORM})
                self.assertTrue(
                    torch.equal(audio.views[AudioView.WAVEFORM][0], waveform)
                )
            finally:
                dataset.close()
                restored.close()

    def test_store_dataset_rejects_unknown_pickle_schema_and_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = read_store_dataset(output)
            state = dataset.__getstate__()
            restored = StoreDataset.__new__(StoreDataset)

            for version in (0, 3):
                with self.subTest(version=version):
                    state["pickle_schema_version"] = version
                    with self.assertRaisesRegex(
                        ValueError,
                        f"Unsupported StoreDataset pickle_schema_version {version}",
                    ):
                        restored.__setstate__(state)

            for version in (True, "1"):
                with self.subTest(version=version):
                    state["pickle_schema_version"] = version
                    with self.assertRaisesRegex(
                        TypeError,
                        "pickle_schema_version must be an integer",
                    ):
                        restored.__setstate__(state)

            state = dataset.__getstate__()
            state["root"] = str(output)
            with self.assertRaisesRegex(TypeError, "field 'root' must be a Path"):
                restored.__setstate__(state)

            state = dataset.__getstate__()
            state["unexpected"] = True
            with self.assertRaisesRegex(ValueError, "unsupported field 'unexpected'"):
                restored.__setstate__(state)

            state = dataset.__getstate__()
            state.pop("views")
            with self.assertRaisesRegex(ValueError, "missing required field 'views'"):
                restored.__setstate__(state)
            dataset.close()

    def test_store_dataset_restores_legacy_payload_group_cache_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = read_store_dataset(output)
            legacy_cache_type = type(
                "_PayloadGroupCache",
                (),
                {
                    "__module__": store_reader.__name__,
                    "__getstate__": _empty_pickle_state,
                },
            )
            with mock.patch.object(
                store_reader,
                "_PayloadGroupCache",
                legacy_cache_type,
            ), mock.patch.object(
                StoreDataset,
                "__getstate__",
                _legacy_store_dataset_pickle_state,
            ):
                legacy = replace(
                    dataset,
                    _payload_group_cache=legacy_cache_type(),
                )
                payload = pickle.dumps(legacy)
            legacy.close()

            restored = pickle.loads(payload)
            try:
                self.assertIsInstance(
                    restored._payload_group_cache,
                    PayloadGroupCache,
                )
                self.assertIsNone(restored._payload_group_cache.fingerprint)
                self.assertEqual(
                    [
                        list(group)
                        for group in restored._shuffle(
                            shuffle=True,
                            seed=0,
                            epoch=0,
                            num_replicas=1,
                            rank=0,
                        )
                    ],
                    [[0]],
                )
            finally:
                restored.close()

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
                "anydataset.store.payload.archive.tarfile.open",
                wraps=__import__("tarfile").open,
            ) as open_tar:
                dataset[0]
                dataset[1]
                dataset[0]

            self.assertEqual(open_tar.call_count, 3)

    def test_store_dataloader_keeps_batches_inside_payload_shards(self):
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
            dataset = AnyDataset(Spec(source=Source.STORE, path=str(output)))

            loader = dataset.dataloader(
                costs=None,
                max_batch_memory=3,
                max_batch_samples=3,
                collate_fn=_sample_indexes,
            )

            self.assertEqual(list(loader), [[0, 1], [2, 3], [4]])

    def test_store_dataloader_shuffle_preserves_payload_locality(self):
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
            dataset = AnyDataset(Spec(source=Source.STORE, path=str(output)))
            loader = dataset.dataloader(
                costs=None,
                max_batch_memory=2,
                max_batch_samples=2,
                shuffle=True,
                seed=13,
                collate_fn=_sample_indexes,
            )
            store = dataset.dataset
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)

            batches = list(loader)

            self.assertEqual(
                sorted(index for batch in batches for index in batch),
                list(range(6)),
            )
            for batch in batches:
                shards = {
                    store.views[view].entries_by_index[index].shard
                    for index in batch
                }
                self.assertEqual(len(shards), 1)

    def test_store_shuffle_splits_each_payload_group_across_ranks(self):
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

            rank0 = list(
                dataset._shuffle(
                    shuffle=True,
                    seed=0,
                    epoch=0,
                    num_replicas=2,
                    rank=0,
                )
            )
            rank1 = list(
                dataset._shuffle(
                    shuffle=True,
                    seed=0,
                    epoch=0,
                    num_replicas=2,
                    rank=1,
                )
            )

            rank0_indexes = [index for group in rank0 for index in group]
            rank1_indexes = [index for group in rank1 for index in group]
            self.assertEqual(sorted((*rank0_indexes, *rank1_indexes)), list(range(6)))
            self.assertEqual(len(rank0_indexes), 3)
            self.assertEqual(len(rank1_indexes), 3)
            self.assertTrue(all(len(group) == 1 for group in (*rank0, *rank1)))

    def test_store_shuffle_keeps_ranks_nonempty_when_groups_are_fewer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=100,
            ).write(
                [
                    _audio_sample(waveform=torch.tensor([[float(index)]]))
                    for index in range(6)
                ]
            )
            dataset = read_store_dataset(output)

            ranks = [
                [
                    index
                    for group in dataset._shuffle(
                        shuffle=False,
                        seed=0,
                        epoch=0,
                        num_replicas=4,
                        rank=rank,
                    )
                    for index in group
                ]
                for rank in range(4)
            ]

            self.assertEqual(ranks, [[0, 4], [1, 5], [2], [3]])

    def test_empty_store_has_an_empty_read_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="empty").write([])
            dataset = read_store_dataset(output)

            for shuffle in (False, True):
                self.assertEqual(
                    list(
                        dataset._shuffle(
                            shuffle=shuffle,
                            seed=0,
                            epoch=0,
                            num_replicas=1,
                            rank=0,
                        )
                    ),
                    [],
                )

    def test_store_shuffle_caches_payload_groups_until_manifest_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=2,
            ).write(
                [
                    _audio_sample(waveform=torch.tensor([[float(index)]]))
                    for index in range(4)
                ]
            )
            dataset = read_store_dataset(output)
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)

            with mock.patch(
                "anydataset.store.payload.groups.scan_payload_groups",
                wraps=__import__(
                    "anydataset.store.payload.groups",
                    fromlist=["scan_payload_groups"],
                ).scan_payload_groups,
            ) as scan:
                list(
                    dataset._shuffle(
                        shuffle=False,
                        seed=0,
                        epoch=0,
                        num_replicas=1,
                        rank=0,
                    )
                )
                list(
                    dataset._shuffle(
                        shuffle=True,
                        seed=0,
                        epoch=1,
                        num_replicas=1,
                        rank=0,
                    )
                )
                manifest_path = view_manifest_parquet_path(output, view)
                manifest_stat = manifest_path.stat()
                os.utime(
                    manifest_path,
                    ns=(manifest_stat.st_atime_ns, manifest_stat.st_mtime_ns + 1),
                )
                list(
                    dataset._shuffle(
                        shuffle=True,
                        seed=0,
                        epoch=2,
                        num_replicas=1,
                        rank=0,
                    )
                )

            self.assertEqual(scan.call_count, 1)

    def test_store_shuffle_uses_persisted_payload_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=2,
            ).write(
                [
                    _audio_sample(waveform=torch.tensor([[float(index)]]))
                    for index in range(4)
                ]
            )
            dataset = read_store_dataset(output)

            with mock.patch(
                "anydataset.store.payload.groups.scan_payload_groups",
                side_effect=AssertionError("payload group sidecar was ignored"),
            ):
                groups = list(
                    dataset._shuffle(
                        shuffle=True,
                        seed=0,
                        epoch=0,
                        num_replicas=1,
                        rank=0,
                    )
                )

            self.assertEqual(
                sorted(sorted(group) for group in groups),
                [[0, 1], [2, 3]],
            )

    def test_payload_group_sidecar_survives_store_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            copied = root / "copied"
            DatasetWriter(
                source,
                dataset_id="toy-audio",
                max_shard_samples=2,
            ).write(
                [
                    _audio_sample(waveform=torch.tensor([[float(index)]]))
                    for index in range(4)
                ]
            )
            shutil.copytree(source, copied)
            dataset = read_store_dataset(copied)

            with mock.patch(
                "anydataset.store.payload.groups.scan_payload_groups",
                side_effect=AssertionError("copied payload sidecar was ignored"),
            ):
                groups = list(
                    dataset._shuffle(
                        shuffle=True,
                        seed=0,
                        epoch=0,
                        num_replicas=1,
                        rank=0,
                    )
                )

            self.assertEqual(
                sorted(sorted(group) for group in groups),
                [[0, 1], [2, 3]],
            )

    def test_payload_groups_rejects_non_integer_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            path = payload_groups_path(output)
            sidecar = read_json(path)
            sidecar["version"] = 2.0
            write_json(path, sidecar)

            self.assertIsNone(read_payload_groups(output))

    def test_store_shuffle_scans_when_payload_group_checksum_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=2,
            ).write(
                [
                    _audio_sample(waveform=torch.tensor([[float(index)]]))
                    for index in range(4)
                ]
            )
            sidecar_path = payload_groups_path(output)
            sidecar = read_json(sidecar_path)
            sidecar["groups_sha256"] = "0" * 64
            write_json(sidecar_path, sidecar)
            dataset = read_store_dataset(output)

            with mock.patch(
                "anydataset.store.payload.groups.scan_payload_groups",
                wraps=__import__(
                    "anydataset.store.payload.groups",
                    fromlist=["scan_payload_groups"],
                ).scan_payload_groups,
            ) as scan:
                groups = list(
                    dataset._shuffle(
                        shuffle=True,
                        seed=0,
                        epoch=0,
                        num_replicas=1,
                        rank=0,
                    )
                )

            scan.assert_called_once()
            self.assertEqual(
                sorted(sorted(group) for group in groups),
                [[0, 1], [2, 3]],
            )

    def test_store_shuffle_scans_when_payload_groups_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [
                    _audio_sample(waveform=torch.tensor([[float(index)]]))
                    for index in range(4)
                ]
            )
            sidecar_path = payload_groups_path(output)
            sidecar = read_json(sidecar_path)
            sidecar["groups"] = [[[0, 1, 2]], [[0, 3, 2]]]
            sidecar["groups_sha256"] = payload_groups_module._groups_checksum(
                sidecar["groups"]
            )
            write_json(sidecar_path, sidecar)
            dataset = read_store_dataset(output)

            with mock.patch(
                "anydataset.store.payload.groups.scan_payload_groups",
                wraps=payload_groups_module.scan_payload_groups,
            ) as scan:
                groups = list(
                    dataset._shuffle(
                        shuffle=True,
                        seed=0,
                        epoch=0,
                        num_replicas=1,
                        rank=0,
                    )
                )

            scan.assert_called_once()
            self.assertEqual(
                sorted(sorted(group) for group in groups),
                [[0, 1, 2, 3]],
            )

    def test_store_shuffle_uses_sidecar_for_selected_view_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = root / "dataset"
            DatasetWriter(
                output,
                dataset_id="multi-view",
                max_shard_samples=2,
            ).write(
                [
                    _audio_sample(
                        waveform=torch.tensor([[float(index)]]),
                        file=str(source),
                    )
                    for index in range(4)
                ]
            )
            file_view = (Role.DEFAULT, Modality.AUDIO, AudioView.FILE)
            waveform_view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            view_manifest_parquet_path(output, file_view).write_bytes(b"not parquet")
            dataset = read_store_dataset(output, views=(waveform_view,))

            with mock.patch(
                "anydataset.store.payload.groups.scan_payload_groups",
                side_effect=AssertionError("selected-view sidecar was ignored"),
            ):
                groups = list(
                    dataset._shuffle(
                        shuffle=True,
                        seed=0,
                        epoch=0,
                        num_replicas=1,
                        rank=0,
                    )
                )

            self.assertEqual(
                sorted(sorted(group) for group in groups),
                [[0, 1], [2, 3]],
            )

    def test_store_exposes_no_public_loader_or_sampler(self):
        self.assertTrue(
            all(
                "loader" not in name.lower() and "sampler" not in name.lower()
                for name in anydataset.store.__all__
            )
        )
        self.assertIn("validate_store_payloads", anydataset.store.__all__)
        self.assertIn("cleanup_store_files", anydataset.store.__all__)
        self.assertIn("migrate_store", anydataset.store.__all__)

    def test_store_dataloader_owns_loader_kwargs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            dataset = read_store_dataset(output)

            with self.assertRaisesRegex(ValueError, "owns loader kwargs"):
                dataset.dataloader(
                    costs=None,
                    max_batch_memory=1,
                    batch_size=1,
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
                "anydataset.store.manifest.index.read_view_manifest_indexes",
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
                "anydataset.store.manifest.index.read_samples_manifest_row_group",
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
                "anydataset.store.manifest.index.read_samples_manifest_row_group",
                wraps=__import__(
                    "anydataset.store.manifest.index",
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

    def test_anydataset_from_store_preserves_selected_views_when_pickled(self):
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
            waveform_view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            dataset = AnyDataset.from_store(output, views=(waveform_view,))

            restored = pickle.loads(pickle.dumps(dataset))
            sample = restored[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        self.assertEqual(set(audio.views), {AudioView.WAVEFORM})
        self.assertTrue(torch.equal(audio.views[AudioView.WAVEFORM][0], waveform))

    def test_dataset_write_supports_parallel_parts_and_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "parallel"
            provenance = {"input_id": "parallel-source-v1"}
            DatasetWriter(
                output,
                dataset_id="parallel",
                split="train",
                num_shards=2,
                num_workers=2,
                provenance=provenance,
            ).write(
                dataset_factory=_RangeAudioFactory(5),
            )
            dataset = read_store_dataset(output)
            self.assertEqual(len(dataset), 5)
            self.assertEqual(dict(dataset.manifest.provenance), provenance)
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

            DatasetWriter(
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
            writer = DatasetWriter(
                Path(tmpdir) / "output",
                num_shards=2,
            )

            with (
                mock.patch(
                    "anydataset.store.part.dispatch.multiprocessing_context",
                    return_value=context,
                ),
                mock.patch(
                    "anydataset.store.part.dispatch.free_port",
                    return_value="1234",
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "start failed"):
                    writer.write(dataset_factory=_RangeAudioFactory(1))

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
                dataset_manifest_dict(
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
                "anydataset.store.manifest.index.read_view_manifest_indexes",
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
                "Store schema 3 view manifest schema does not match expected fields",
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
                "Store schema 3 sample manifest schema does not match expected fields",
            ):
                read_store_dataset(output)

    def test_reader_rejects_invalid_sample_metadata_at_lazy_row_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            DatasetWriter(output, dataset_id="toy-audio").write(
                [_audio_sample(waveform=torch.tensor([[1.0]]))]
            )
            _rewrite_sample_items(
                output,
                '[[["default", "audio"], 42]]',
            )
            dataset = read_store_dataset(output)

            with self.assertRaisesRegex(
                ValueError,
                "Sample manifest item metadata must be a JSON object",
            ):
                dataset[0]


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


def _empty_pickle_state(_instance):
    return {}


def _legacy_store_dataset_pickle_state(instance):
    state = dict(instance.__dict__)
    state.pop("_file_mode")
    return state


def _sample_indexes(samples):
    return [_sample_index(sample) for sample in samples]


def _sample_index(sample) -> int:
    waveform, _sample_rate = sample[Role.DEFAULT, Modality.AUDIO].views[
        AudioView.WAVEFORM
    ]
    return int(waveform.flatten()[0].item())


def _write_empty_dataset(path: Path) -> None:
    path.mkdir()
    write_json(
        path / "dataset.json",
        dataset_manifest_dict(
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


def _rewrite_sample_items(root: Path, payload: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = samples_parquet_path(root)
    table = pq.read_table(path)
    column = table.schema.get_field_index("items")
    items = pa.array([payload], type=pa.string())
    pq.write_table(table.set_column(column, "items", items), path)


if __name__ == "__main__":
    unittest.main()
