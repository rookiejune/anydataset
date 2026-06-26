import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import torch

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioMeta,
    AudioReq,
    AudioView,
    Modality,
    Role,
    Source,
    Spec,
    TextItem,
    TextView,
)
from anydataset.store import DatasetWriter
from anydataset.store.jsonio import write_json
from anydataset.store.manifest import DatasetManifest, SampleManifestEntry
from anydataset.store.manifestio import write_samples_manifest
from anydataset.store.paths import dataset_ready_path, view_dir, view_manifest_parquet_path
from anydataset.store.reader import read_store_dataset


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
                cache_root=root / "cache",
            )
            sample = dataset[0]

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        text = sample[Role.DEFAULT, Modality.TEXT]
        loaded_waveform, sample_rate = audio.views[AudioView.WAVEFORM]
        self.assertTrue(torch.equal(loaded_waveform, waveform))
        self.assertEqual(sample_rate, 4)
        self.assertEqual(audio.meta[AudioMeta.LABEL], "speech")
        self.assertEqual(text.views[TextView.TEXT], "hello")

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
                cache_root=root / "cache",
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

    def test_reader_loads_all_view_indexes(self):
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

            dataset = read_store_dataset(output)

        self.assertEqual(
            set(dataset.views),
            {
                (Role.DEFAULT, Modality.AUDIO, AudioView.FILE),
                (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),
            },
        )
        self.assertEqual(len(dataset.samples), 1)

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
                cache_root=root / "cache",
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
            write_json(
                view_path / "view.json",
                {"role": "default", "modality": "audio", "view": "waveform"},
            )

            with self.assertRaises(ValueError):
                read_store_dataset(output)

    def test_reader_rejects_view_metadata_that_disagrees_with_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            _write_empty_dataset(output)
            view_path = view_dir(output, (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM))
            view_path.mkdir(parents=True)
            write_json(
                view_path / "view.json",
                {"role": "default", "modality": "audio", "view": "file"},
            )
            (view_path / "manifest.parquet").write_bytes(b"not-parquet")
            (view_path / ".ready").touch()

            with self.assertRaises(ValueError):
                read_store_dataset(output)

    def test_reader_rejects_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "dataset"
            output.mkdir()
            write_json(
                output / "dataset.json",
                asdict(DatasetManifest(dataset_id="toy-audio", sample_count=2)),
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
        asdict(DatasetManifest(dataset_id="toy-audio", sample_count=0)),
    )
    write_samples_manifest(path, [])
    dataset_ready_path(path).touch()


def _drop_last_parquet_row(path: Path) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    pq.write_table(table.slice(0, table.num_rows - 1), path)


if __name__ == "__main__":
    unittest.main()
