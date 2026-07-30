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
from anydataset.store import _part_commit as store_parts
from anydataset.store._part_commit import (
    commit_fragment_part,
    commit_store_fragments,
    commit_store_parts,
    completed_fragment_indexes,
)
from anydataset.store._part_writer import DatasetFragmentWriter, DatasetPartWriter
from anydataset.store._config import DEFAULT_MAX_SHARD_SAMPLES
from anydataset.store.jsonio import read_json, write_json
from anydataset.store.manifest import (
    DatasetManifest,
    SampleManifestEntry,
    ViewManifestEntry,
)
from anydataset.store.manifestio import (
    read_samples_manifest,
    read_view_manifest,
    write_samples_manifest,
    write_view_manifest,
)
from anydataset.store.paths import (
    dataset_json_path,
    dataset_ready_path,
    payload_groups_path,
    samples_parquet_path,
    view_dir,
    view_manifest_parquet_path,
    view_ready_path,
    view_shard_path,
    view_shard_index_path,
)
from anydataset.store.reader import read_store_dataset


class DatasetWriterTest(unittest.TestCase):
    def test_fragment_id_rejects_platform_path_separators(self):
        for fragment_id in ("parent/child", "parent\\child"):
            with self.subTest(fragment_id=fragment_id):
                with self.assertRaisesRegex(ValueError, "path separators"):
                    DatasetFragmentWriter(
                        "unused",
                        dataset_id="toy-audio",
                        fragment_id=fragment_id,
                    )

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
                {
                    "dataset_id",
                    "sample_count",
                    "schema_version",
                    "split",
                    "provenance",
                },
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

    def test_explicit_views_are_strictly_validated(self):
        view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
        cases = (
            ([view], TypeError, "views must be a tuple"),
            ((view[:2],), TypeError, "views entries"),
            ((("default", Modality.AUDIO, AudioView.WAVEFORM),), TypeError, "role"),
            (((Role.DEFAULT, "audio", AudioView.WAVEFORM),), TypeError, "modality"),
            (
                ((Role.DEFAULT, Modality.AUDIO, TextView.TEXT),),
                TypeError,
                "AudioView",
            ),
        )

        for views, error, message in cases:
            with self.subTest(views=views):
                with self.assertRaisesRegex(error, message):
                    DatasetWriter("unused", dataset_id="toy", views=views)

    def test_writers_reject_duplicate_explicit_views(self):
        view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)

        with self.assertRaisesRegex(ValueError, "Duplicate store view"):
            DatasetWriter("unused", dataset_id="toy", views=(view, view))
        with self.assertRaisesRegex(ValueError, "Duplicate store view"):
            DatasetPartWriter(
                "unused",
                dataset_id="toy",
                shard_id=0,
                num_shards=1,
                views=(view, view),
            )

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
                    text=f"text-{index}",
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

            with mock.patch.object(
                store_parts,
                "_sample_indexes_for_ref",
                side_effect=AssertionError("sample manifest was scanned per view"),
            ):
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

    def test_fragment_metadata_rejects_boolean_integer_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fragments = root / "fragments"
            fragment = fragments / "batch-000000000000-000000000000-a"
            DatasetFragmentWriter(
                fragment,
                dataset_id="toy-audio",
                fragment_id=fragment.name,
            ).write(
                [
                    (
                        0,
                        audio_sample(
                            waveform=torch.tensor([[1.0]]),
                            sample_rate=4,
                        ),
                    )
                ]
            )
            metadata_path = fragment / "fragment.json"
            original = read_json(metadata_path)
            for field, value, message in (
                (
                    "sample_indexes",
                    [True],
                    "sample_indexes entries must be integers",
                ),
                ("sample_count", True, "sample_count must be an integer"),
                ("sample_count", 1.0, "sample_count must be an integer"),
            ):
                with self.subTest(field=field, value=value):
                    metadata = dict(original)
                    metadata[field] = value
                    write_json(metadata_path, metadata)

                    with self.assertRaisesRegex(ValueError, message):
                        completed_fragment_indexes(
                            fragments,
                            dataset_id="toy-audio",
                        )

    def test_fragment_manifest_merge_reads_one_row_per_store_lazily(self):
        stores = (Path("fragment-a"), Path("fragment-b"))
        consumed = {store: 0 for store in stores}

        def entries(store, indexes):
            for index in indexes:
                consumed[store] += 1
                yield SampleManifestEntry(
                    sample_id=f"sample-{index}",
                    sample_index=index,
                    items=(),
                )

        manifests = {
            stores[0]: entries(stores[0], (0, 2)),
            stores[1]: entries(stores[1], (1, 3)),
        }
        with mock.patch.object(
            store_parts,
            "read_samples_manifest",
            side_effect=lambda store, **_kwargs: manifests[store],
        ):
            merged = store_parts._merged_sample_entries(stores)

            self.assertEqual(next(merged).sample_index, 0)

        self.assertEqual(consumed, {stores[0]: 1, stores[1]: 1})

    def test_manifest_merge_keeps_every_active_parquet_handle_open(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            store_count = 17
            rows_per_store = 4097
            total = store_count * rows_per_store
            stores = tuple(root / f"store-{index:02d}" for index in range(store_count))

            for store_index, store in enumerate(stores):
                write_samples_manifest(
                    store,
                    (
                        SampleManifestEntry(
                            sample_id=f"sample-{index}",
                            sample_index=index,
                            items=(),
                        )
                        for index in range(store_index, total, store_count)
                    ),
                )
                write_view_manifest(
                    store,
                    view,
                    (
                        ViewManifestEntry(
                            role=Role.DEFAULT,
                            modality=Modality.AUDIO,
                            view=AudioView.WAVEFORM,
                            sample_index=index,
                            shard=f"{store_index:06d}.tar",
                            key=f"sample-{index}.pt",
                        )
                        for index in range(store_index, total, store_count)
                    ),
                )
                view_ready_path(store, view).touch()

            self.assertEqual(
                [
                    entry.sample_index
                    for entry in store_parts._merged_sample_entries(stores)
                ],
                list(range(total)),
            )
            self.assertEqual(
                [
                    entry.sample_index
                    for entry in store_parts._merged_view_entries(stores, view)
                ],
                list(range(total)),
            )

    def test_index_runs_preserve_irregular_distributions(self):
        indexes = (0, 3, 6, 8, 9, 10, 14, 20, 26, 27, 28)
        runs = store_parts._IndexRuns()

        for index in indexes:
            runs.add(index)

        self.assertEqual(tuple(runs), indexes)
        self.assertEqual(
            runs._runs,
            [
                [0, 3, 3],
                [8, 1, 3],
                [14, 6, 3],
                [27, 1, 2],
            ],
        )

    def test_fragment_commit_bounds_manifest_merge_fan_in(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fragments = root / "fragments"
            output = root / "dataset"
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            for index in range(5):
                fragment = fragments / f"batch-{index:012d}-{index:012d}-a"
                DatasetFragmentWriter(
                    fragment,
                    dataset_id="toy-audio",
                    fragment_id=fragment.name,
                ).write(
                    [
                        (
                            index,
                            audio_sample(
                                waveform=torch.tensor([[float(index)]]),
                                sample_rate=4,
                            ),
                        )
                    ]
                )

            with mock.patch.object(store_parts, "_MERGE_FAN_IN", 2):
                commit_store_fragments(
                    output,
                    fragments,
                    dataset_id="toy-audio",
                    expected_sample_count=5,
                )

            self.assertEqual(
                [entry.sample_index for entry in read_samples_manifest(output)],
                list(range(5)),
            )
            self.assertEqual(
                [entry.sample_index for entry in read_view_manifest(output, view)],
                list(range(5)),
            )
            self.assertFalse(any(root.glob(".dataset-merge-*")))

    def test_fragment_commits_preserve_views_from_all_fragments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fragments = root / "fragments"
            output = root / "dataset"
            part = root / "part"
            source_view = (Role.SOURCE, Modality.AUDIO, AudioView.WAVEFORM)
            target_view = (Role.TARGET, Modality.AUDIO, AudioView.WAVEFORM)
            source_fragment = fragments / "batch-000000000000-000000000000-a"
            target_fragment = fragments / "batch-000000000001-000000000001-b"

            DatasetFragmentWriter(
                source_fragment,
                dataset_id="paired-audio",
                fragment_id=source_fragment.name,
            ).write(
                [
                    (
                        0,
                        {
                            (Role.SOURCE, Modality.AUDIO): AudioItem(
                                views={AudioView.WAVEFORM: (torch.tensor([[1.0]]), 4)}
                            )
                        },
                    )
                ]
            )
            DatasetFragmentWriter(
                target_fragment,
                dataset_id="paired-audio",
                fragment_id=target_fragment.name,
            ).write(
                [
                    (
                        1,
                        {
                            (Role.TARGET, Modality.AUDIO): AudioItem(
                                views={AudioView.WAVEFORM: (torch.tensor([[2.0]]), 4)}
                            )
                        },
                    )
                ]
            )

            commit_store_fragments(
                output,
                fragments,
                dataset_id="paired-audio",
                expected_sample_count=2,
            )
            commit_fragment_part(
                part,
                (source_fragment, target_fragment),
                dataset_id="paired-audio",
                shard_id=0,
                num_shards=1,
            )

            for store in (output, part):
                self.assertEqual(
                    [
                        entry.sample_index
                        for entry in read_view_manifest(store, source_view)
                    ],
                    [0],
                )
                self.assertEqual(
                    [
                        entry.sample_index
                        for entry in read_view_manifest(store, target_view)
                    ],
                    [1],
                )

    def test_fragment_part_allows_sparse_indexes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fragments = root / "fragments"
            part = root / "part"
            sample = audio_sample(waveform=torch.tensor([[1.0]]), sample_rate=4)
            fragment = fragments / "batch-000000000001-000000000001-a"
            DatasetFragmentWriter(
                fragment,
                dataset_id="toy-audio",
                fragment_id=fragment.name,
            ).write([(1, sample)])

            commit_fragment_part(
                part,
                (fragment,),
                dataset_id="toy-audio",
                shard_id=1,
                num_shards=2,
            )

            self.assertEqual(
                [entry.sample_index for entry in read_samples_manifest(part)],
                [1],
            )

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

    def test_commit_parts_rejects_part_without_store_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parts = root / "parts"
            part = parts / "part-00000"
            DatasetPartWriter(
                part,
                dataset_id="toy-audio",
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
                    )
                ]
            )
            manifest = read_json(dataset_json_path(part))
            del manifest["schema_version"]
            write_json(dataset_json_path(part), manifest)

            with self.assertRaisesRegex(
                ValueError,
                "Unsupported store schema_version: None; expected 2 or 3",
            ):
                commit_store_parts(
                    root / "output",
                    parts,
                    dataset_id="toy-audio",
                )

    def test_commit_parts_rejects_non_integer_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parts, part, _view, _entry = _single_audio_part(root)
            metadata_path = part / "part.json"
            original = read_json(metadata_path)

            for field, value in (
                ("num_shards", "1"),
                ("num_shards", 1.0),
                ("num_shards", True),
                ("shard_id", "0"),
                ("shard_id", "invalid"),
                ("shard_id", 0.0),
                ("shard_id", False),
                ("sample_count", True),
                ("sample_count", 1.0),
            ):
                with self.subTest(field=field, value=value):
                    metadata = dict(original)
                    metadata[field] = value
                    write_json(metadata_path, metadata)

                    with self.assertRaisesRegex(
                        ValueError,
                        rf"Part {field} must be an integer",
                    ):
                        commit_store_parts(
                            root / "output",
                            parts,
                            dataset_id="toy-audio",
                        )

            metadata = dict(original)
            del metadata["shard_id"]
            write_json(metadata_path, metadata)
            with self.assertRaisesRegex(
                ValueError,
                "Part shard_id must be an integer",
            ):
                commit_store_parts(
                    root / "output",
                    parts,
                    dataset_id="toy-audio",
                )

    def test_commit_parts_rejects_declared_sample_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parts = root / "parts"
            part = parts / "part-00000"
            DatasetPartWriter(
                part,
                dataset_id="toy-audio",
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
                    )
                ]
            )
            for path in (dataset_json_path(part), part / "part.json"):
                data = read_json(path)
                data["sample_count"] = 2
                write_json(path, data)

            with self.assertRaisesRegex(ValueError, "sample manifest row count"):
                commit_store_parts(
                    root / "output",
                    parts,
                    dataset_id="toy-audio",
                )

    def test_commit_parts_rejects_missing_view_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parts, part, view, entry = _single_audio_part(root)
            view_shard_path(part, view, entry.shard).unlink()

            with self.assertRaisesRegex(FileNotFoundError, "missing referenced shard"):
                commit_store_parts(
                    root / "output",
                    parts,
                    dataset_id="toy-audio",
                )

            self.assertFalse((root / "output").exists())

    def test_commit_parts_rejects_missing_view_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parts, part, view, entry = _single_audio_part(root)
            with tarfile.open(view_shard_path(part, view, entry.shard), "w"):
                pass

            with self.assertRaisesRegex(ValueError, "missing payload"):
                commit_store_parts(
                    root / "output",
                    parts,
                    dataset_id="toy-audio",
                )

            self.assertFalse((root / "output").exists())

    def test_link_or_copy_falls_back_to_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.tar"
            target = root / "target.tar"
            source.write_bytes(b"payload")

            with mock.patch.object(store_parts.os, "link", side_effect=OSError):
                store_parts._link_or_copy(source, target)

            self.assertEqual(target.read_bytes(), b"payload")

    def test_commit_parts_rebuilds_payload_index_after_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parts, part, view, entry = _single_audio_part(root)
            view_shard_index_path(part, view, entry.shard).unlink()
            output = root / "output"

            with mock.patch.object(store_parts.os, "link", side_effect=OSError):
                commit_store_parts(output, parts, dataset_id="toy-audio")

            self.assertTrue(
                view_shard_index_path(output, view, entry.shard).is_file()
            )
            dataset = read_store_dataset(output)
            with mock.patch.object(
                tarfile.TarFile,
                "getmembers",
                side_effect=AssertionError("rebuilt payload index was ignored"),
            ):
                waveform, sample_rate = dataset[0][
                    Role.DEFAULT, Modality.AUDIO
                ].views[AudioView.WAVEFORM]

            self.assertTrue(torch.equal(waveform, torch.tensor([[1.0]])))
            self.assertEqual(sample_rate, 4)

    def test_commit_parts_compresses_interleaved_payload_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parts = root / "parts"
            sample_count = 40
            for shard_id in range(4):
                DatasetPartWriter(
                    parts / f"part-{shard_id:05d}",
                    dataset_id="toy-audio",
                    shard_id=shard_id,
                    num_shards=4,
                    max_shard_samples=sample_count,
                ).write(
                    (
                        (
                            index,
                            audio_sample(
                                waveform=torch.tensor([[float(index)]]),
                                sample_rate=4,
                            ),
                        )
                        for index in range(shard_id, sample_count, 4)
                    )
                )

            output = root / "output"
            commit_store_parts(output, parts, dataset_id="toy-audio")
            sidecar = read_json(payload_groups_path(output))

            self.assertEqual(sidecar["version"], 2)
            self.assertEqual(len(sidecar["groups"]), 4)
            self.assertEqual(
                sum(len(bucket) for bucket in sidecar["groups"]),
                4,
            )
            dataset = read_store_dataset(output)
            with mock.patch(
                "anydataset.store._payload_groups.scan_payload_groups",
                side_effect=AssertionError("compressed sidecar was ignored"),
            ):
                shuffled = list(
                    dataset._shuffle(
                        shuffle=True,
                        seed=7,
                        epoch=0,
                        num_replicas=1,
                        rank=0,
                    )
                )
                ordered = list(
                    dataset._shuffle(
                        shuffle=False,
                        seed=7,
                        epoch=0,
                        num_replicas=1,
                        rank=0,
                    )
                )

            self.assertEqual(
                sorted(index for group in shuffled for index in group),
                list(range(sample_count)),
            )
            self.assertEqual(
                [index for group in ordered for index in group],
                list(range(sample_count)),
            )

    def test_commit_parts_groups_interleaved_multi_view_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parts = root / "parts"
            for shard_id in range(2):
                DatasetPartWriter(
                    parts / f"part-{shard_id:05d}",
                    dataset_id="multi-view",
                    shard_id=shard_id,
                    num_shards=2,
                    max_shard_samples=8,
                ).write(
                    (
                        (
                            index,
                            audio_sample(
                                waveform=torch.tensor([[float(index)]]),
                                sample_rate=4,
                                text=(
                                    f"text-{index}"
                                    if index % 4 == shard_id
                                    else None
                                ),
                            ),
                        )
                        for index in range(shard_id, 8, 2)
                    )
                )

            output = root / "output"
            commit_store_parts(output, parts, dataset_id="multi-view")
            dataset = read_store_dataset(output)

            with mock.patch(
                "anydataset.store._payload_groups.scan_payload_groups",
                side_effect=AssertionError("multi-view sidecar was ignored"),
            ):
                groups = list(
                    dataset._shuffle(
                        shuffle=True,
                        seed=7,
                        epoch=0,
                        num_replicas=1,
                        rank=0,
                    )
                )

            self.assertEqual(
                sorted(sorted(group) for group in groups),
                [[0, 4], [1, 5], [2, 6], [3, 7]],
            )


def _single_audio_part(root: Path):
    parts = root / "parts"
    part = parts / "part-00000"
    view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
    DatasetPartWriter(
        part,
        dataset_id="toy-audio",
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
            )
        ]
    )
    return parts, part, view, next(read_view_manifest(part, view))


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
