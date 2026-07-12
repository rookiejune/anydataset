from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import torch

from anydataset import AnyDataset, Source, Spec
from anydataset.dataset import DatasetStoreWriter
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
from anydataset.store import DatasetWriter
from anydataset.store.jsonio import write_json
from anydataset.store.manifest import (
    DatasetManifest,
    SampleManifestEntry,
    STORE_SCHEMA_VERSION,
)
from anydataset.store.manifestio import write_samples_manifest
from anydataset.store.paths import dataset_ready_path, view_dir, view_manifest_parquet_path
from anydataset.store.reader import read_store_dataset, read_store_manifest


class StoreSourceTest(unittest.TestCase):
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

            file_view = Path(dataset[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE])
            with mock.patch(
                "anydataset.store.reader.read_payload_bytes",
                side_effect=AssertionError("cache miss"),
            ):
                cached = Path(dataset[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE])

            self.assertTrue(file_view.is_file())
            self.assertEqual(file_view.read_bytes(), b"RIFF-data")
            self.assertEqual(cached, file_view)

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


if __name__ == "__main__":
    unittest.main()
