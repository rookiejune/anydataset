from __future__ import annotations

import os
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import torch

from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextView,
)
from anydataset.store import DatasetWriter
from anydataset.store.parts import (
    DatasetFragmentWriter,
    DatasetPartWriter,
    commit_store_parts,
    commit_store_fragments,
    completed_fragment_indexes,
)
from anydataset.store.writer import DEFAULT_MAX_SHARD_SAMPLES
from anydataset.store.jsonio import read_json
from anydataset.store.manifest import DatasetManifest, ViewManifestEntry
from anydataset.store.manifestio import (
    read_samples_manifest,
    read_view_manifest,
    write_view_manifest,
)
from anydataset.store.paths import (
    dataset_json_path,
    dataset_ready_path,
    samples_parquet_path,
    view_dir,
    view_manifest_parquet_path,
    view_ready_path,
    view_shard_path,
)
from anydataset.store import parts as store_parts


class DatasetWriterTest(unittest.TestCase):
    def test_writer_writes_waveform_view_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            waveform = torch.tensor([[1.0, 2.0, 3.0]])
            sample = audio_sample(
                waveform=waveform,
                sample_rate=4,
                label="speech",
                text="hello",
            )
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)

            written = DatasetWriter(
                output,
                dataset_id="toy-audio",
                split="train",
            ).write([sample])

            dataset_json = read_json(dataset_json_path(output))
            DatasetManifest(**dataset_json)
            sample_entry = next(read_samples_manifest(output))
            view_entry = next(read_view_manifest(output, view))

            self.assertEqual(written, output)
            self.assertEqual(
                set(dataset_json),
                {"dataset_id", "sample_count", "schema_version", "split"},
            )
            self.assertTrue(dataset_ready_path(output).exists())
            self.assertTrue(view_ready_path(output, view).exists())
            self.assertTrue(samples_parquet_path(output).is_file())
            self.assertTrue(view_manifest_parquet_path(output, view).is_file())
            self.assertFalse((view_dir(output, view) / "view.json").exists())
            self.assertEqual(sample_entry.sample_index, view_entry.sample_index)
            audio_entry = sample_entry.item((Role.DEFAULT, Modality.AUDIO))
            self.assertEqual(audio_entry[1]["label"], "speech")
            self.assertEqual(
                set(view_entry.__dict__),
                {
                    "role",
                    "modality",
                    "view",
                    "sample_index",
                    "shard",
                    "key",
                },
            )

            with tarfile.open(view_shard_path(output, view, view_entry.shard), "r") as tar:
                payload = tar.extractfile(view_entry.key).read()
            loaded_waveform, loaded_sample_rate = torch.load(BytesIO(payload))
            self.assertTrue(torch.equal(loaded_waveform, waveform))
            self.assertEqual(loaded_sample_rate, 4)

    def test_writer_copies_file_view_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.wav"
            source.write_bytes(b"RIFF-data")
            output = Path(tmpdir) / "dataset"
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.FILE)

            DatasetWriter(output, dataset_id="file-audio").write(
                [audio_sample(file=str(source), sample_rate=16000)]
            )
            view_entry = next(read_view_manifest(output, view))

            self.assertEqual(Path(view_entry.key).suffix, ".wav")
            with tarfile.open(view_shard_path(output, view, view_entry.shard), "r") as tar:
                self.assertEqual(tar.extractfile(view_entry.key).read(), b"RIFF-data")

    def test_writer_rolls_shards_by_sample_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            samples = [
                audio_sample(waveform=torch.tensor([[float(index)]]), sample_rate=4)
                for index in range(3)
            ]

            DatasetWriter(
                output,
                dataset_id="toy-audio",
                max_shard_samples=1,
            ).write(samples)

            entries = list(read_view_manifest(output, view))

            self.assertEqual(
                [entry.shard for entry in entries],
                ["000000.tar", "000001.tar", "000002.tar"],
            )

    def test_writer_defaults_to_100k_samples_per_shard(self):
        writer = DatasetWriter("unused", dataset_id="toy")

        self.assertEqual(writer.max_shard_samples, 100_000)
        self.assertEqual(DEFAULT_MAX_SHARD_SAMPLES, 100_000)

    def test_explicit_view_must_exist_on_each_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            missing = Path(tmpdir) / "missing.wav"
            writer = DatasetWriter(
                output,
                dataset_id="toy",
                views=((Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),),
            )

            with self.assertRaises(KeyError):
                writer.write([audio_sample(file=str(missing), sample_rate=16000)])

            self.assertFalse(output.exists())

    def test_writer_rejects_sample_without_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"

            with self.assertRaises(ValueError):
                DatasetWriter(output, dataset_id="toy").write(
                    [audio_sample(sample_rate=16000)]
                )

            self.assertFalse(output.exists())

    def test_writer_rejects_inconsistent_item_view_sets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dataset"
            source = Path(tmpdir) / "source.wav"
            source.write_bytes(b"RIFF-data")

            with self.assertRaises(ValueError):
                DatasetWriter(output, dataset_id="toy-audio").write(
                    [
                        audio_sample(
                            waveform=torch.tensor([[1.0]]),
                            file=str(source),
                            sample_rate=4,
                        ),
                        audio_sample(
                            waveform=torch.tensor([[2.0]]),
                            sample_rate=4,
                        ),
                    ]
                )

            self.assertFalse(output.exists())

    def test_fragment_writer_commits_batch_fragments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fragments = root / "fragments"
            output = root / "dataset"
            samples = [
                audio_sample(
                    waveform=torch.tensor([[float(index)]]),
                    sample_rate=4,
                )
                for index in range(3)
            ]

            DatasetFragmentWriter(
                fragments / "batch-000000000000-000000000001-a",
                dataset_id="toy-audio",
                split="train",
                fragment_id="batch-000000000000-000000000001-a",
            ).write([(0, samples[0]), (1, samples[1])])
            DatasetFragmentWriter(
                fragments / "batch-000000000002-000000000002-b",
                dataset_id="toy-audio",
                split="train",
                fragment_id="batch-000000000002-000000000002-b",
            ).write([(2, samples[2])])

            self.assertEqual(
                completed_fragment_indexes(
                    fragments,
                    dataset_id="toy-audio",
                    split="train",
                ),
                frozenset({0, 1, 2}),
            )

            commit_store_fragments(
                output,
                fragments,
                dataset_id="toy-audio",
                split="train",
                expected_sample_count=3,
            )

            indexes = [entry.sample_index for entry in read_samples_manifest(output)]
            self.assertEqual(indexes, [0, 1, 2])
            self.assertTrue(dataset_ready_path(output).exists())

    def test_commit_parts_allows_views_for_partial_item_coverage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parts = root / "parts"
            output = root / "dataset"
            audio_view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            text_view = (Role.DEFAULT, Modality.TEXT, TextView.TEXT)

            DatasetPartWriter(
                parts / "part-00000",
                dataset_id="toy-mixed",
                split="train",
                shard_id=0,
                num_shards=1,
            ).write(
                [
                    (
                        0,
                        audio_sample(
                            waveform=torch.tensor([[1.0]]),
                            sample_rate=4,
                            text="hello",
                        ),
                    ),
                    (
                        1,
                        {
                            (Role.DEFAULT, Modality.TEXT): TextItem(
                                views={TextView.TEXT: "text only"}
                            )
                        },
                    ),
                ]
            )

            commit_store_parts(
                output,
                parts,
                dataset_id="toy-mixed",
                split="train",
            )

            self.assertEqual(len(list(read_view_manifest(output, audio_view))), 1)
            self.assertEqual(len(list(read_view_manifest(output, text_view))), 2)

    def test_part_writer_requires_increasing_sample_indexes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            samples = [
                audio_sample(
                    waveform=torch.tensor([[float(index)]]),
                    sample_rate=4,
                )
                for index in range(2)
            ]

            with self.assertRaisesRegex(ValueError, "indexes must be increasing"):
                DatasetPartWriter(
                    root / "part-00000",
                    dataset_id="toy-audio",
                    split="train",
                    shard_id=0,
                    num_shards=1,
                ).write([(1, samples[1]), (0, samples[0])])

    def test_commit_fragments_links_view_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fragments = root / "fragments"
            output = root / "dataset"
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            sample = audio_sample(
                waveform=torch.tensor([[1.0]]),
                sample_rate=4,
            )

            DatasetFragmentWriter(
                fragments / "batch-000000000000-000000000000-a",
                dataset_id="toy-audio",
                split="train",
                fragment_id="batch-000000000000-000000000000-a",
            ).write([(0, sample)])

            fragment = next(fragments.iterdir())
            fragment_entry = next(read_view_manifest(fragment, view))
            fragment_shard = view_shard_path(fragment, view, fragment_entry.shard)
            commit_store_fragments(
                output,
                fragments,
                dataset_id="toy-audio",
                split="train",
                expected_sample_count=1,
            )
            output_entry = next(read_view_manifest(output, view))
            output_shard = view_shard_path(output, view, output_entry.shard)

            self.assertEqual(os.stat(fragment_shard).st_ino, os.stat(output_shard).st_ino)

    def test_commit_fragments_preserves_order_for_unsorted_fragment_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fragments = root / "fragments"
            output = root / "dataset"
            samples = [
                audio_sample(
                    waveform=torch.tensor([[float(index)]]),
                    sample_rate=4,
                )
                for index in range(3)
            ]

            DatasetFragmentWriter(
                fragments / "batch-000000000000-000000000002-a",
                dataset_id="toy-audio",
                split="train",
                fragment_id="batch-000000000000-000000000002-a",
            ).write([(2, samples[2]), (0, samples[0])])
            DatasetFragmentWriter(
                fragments / "batch-000000000001-000000000001-b",
                dataset_id="toy-audio",
                split="train",
                fragment_id="batch-000000000001-000000000001-b",
            ).write([(1, samples[1])])

            commit_store_fragments(
                output,
                fragments,
                dataset_id="toy-audio",
                split="train",
                expected_sample_count=3,
            )

            entries = list(read_samples_manifest(output))
            self.assertEqual([entry.sample_index for entry in entries], [0, 1, 2])
            self.assertEqual(
                [entry.sample_id for entry in entries],
                [
                    "000000000000-toy-audio",
                    "000000000001-toy-audio",
                    "000000000002-toy-audio",
                ],
            )

    def test_commit_parts_rejects_view_entry_for_wrong_sample_item(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parts = root / "parts"
            output = root / "dataset"
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)

            DatasetPartWriter(
                parts / "part-00000",
                dataset_id="toy-mixed",
                split="train",
                shard_id=0,
                num_shards=1,
            ).write(
                [
                    (
                        0,
                        audio_sample(
                            waveform=torch.tensor([[1.0]]),
                            sample_rate=4,
                        ),
                    ),
                    (
                        1,
                        {
                            (Role.DEFAULT, Modality.TEXT): TextItem(
                                views={TextView.TEXT: "text only"}
                            )
                        },
                    ),
                ]
            )
            entries = list(read_view_manifest(parts / "part-00000", view))
            write_view_manifest(
                parts / "part-00000",
                view,
                [
                    ViewManifestEntry(
                        role=entry.role,
                        modality=entry.modality,
                        view=entry.view,
                        sample_index=1,
                        shard=entry.shard,
                        key=entry.key,
                    )
                    for entry in entries
                ],
            )

            with self.assertRaisesRegex(ValueError, "sample_index"):
                commit_store_parts(
                    output,
                    parts,
                    dataset_id="toy-mixed",
                    split="train",
                )

    def test_link_or_copy_falls_back_to_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.tar"
            target = root / "target.tar"
            source.write_bytes(b"payload")

            with mock.patch.object(store_parts.os, "link", side_effect=OSError):
                store_parts._link_or_copy(source, target)

            self.assertEqual(target.read_bytes(), b"payload")


def audio_sample(
    *,
    waveform=None,
    file=None,
    longcat=None,
    sample_rate: int,
    label=None,
    text: str | None = None,
):
    views = {}
    if waveform is not None:
        views[AudioView.WAVEFORM] = (waveform, sample_rate)
    if file is not None:
        views[AudioView.FILE] = file
    if longcat is not None:
        views[AudioView.LONGCAT] = longcat
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


if __name__ == "__main__":
    unittest.main()
