import os
import pickle
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from unittest import mock

from anydataset.types import AudioView, Modality, Role
from anydataset.store._refs import sample_ref_path, validate_entry_ref, view_path
from anydataset.store.jsonio import read_json, write_json
from anydataset.store.manifest.schema import (
    DatasetManifest,
    SampleManifestEntry,
    STORE_SCHEMA_VERSION,
    ViewManifestEntry,
    dataset_manifest_dict,
    dataset_manifest_from_dict,
)
from anydataset.store.manifest.io import (
    ManifestParquetCache,
    read_samples_manifest,
    read_sample_manifest_index,
    read_view_manifest,
    read_view_manifest_indexes,
    sample_manifest_row_groups,
    samples_manifest_exists,
    view_manifest_row_count,
    view_manifest_row_groups,
    write_samples_manifest,
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
    view_shards_dir,
)


class StoreTest(unittest.TestCase):
    def test_store_ref_paths_use_manifest_order(self):
        view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)

        self.assertEqual(view_path(view), ("default", "audio", "waveform"))
        self.assertEqual(sample_ref_path(view[:2]), ("default", "audio"))

    def test_store_entry_ref_must_match_path(self):
        entry = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)

        validate_entry_ref(entry, entry)
        with self.assertRaisesRegex(
            ValueError,
            r"^View manifest entry ref must match its path\.$",
        ):
            validate_entry_ref(
                entry,
                (Role.DEFAULT, Modality.AUDIO, AudioView.FILE),
            )

    def test_view_paths_use_role_modality_view(self):
        root = Path("/tmp/dataset")
        view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)

        self.assertEqual(dataset_json_path(root), root / "dataset.json")
        self.assertEqual(samples_parquet_path(root), root / "samples.parquet")
        self.assertEqual(dataset_ready_path(root), root / ".ready")
        self.assertEqual(view_dir(root, view), root / "default" / "audio" / "waveform")
        self.assertEqual(
            view_manifest_parquet_path(root, view),
            root / "default" / "audio" / "waveform" / "manifest.parquet",
        )
        self.assertEqual(
            view_ready_path(root, view),
            root / "default" / "audio" / "waveform" / ".ready",
        )
        self.assertEqual(
            view_shards_dir(root, view),
            root / "default" / "audio" / "waveform" / "shards",
        )
        self.assertEqual(
            view_shard_path(root, view, "000000.tar"),
            root / "default" / "audio" / "waveform" / "shards" / "000000.tar",
        )

    def test_manifest_dataclasses_round_trip(self):
        provenance = {"input_id": "input-v1"}
        manifest = DatasetManifest(
            dataset_id="toy-audio",
            schema_version=STORE_SCHEMA_VERSION,
            split="train",
            sample_count=2,
            provenance=provenance,
        )
        provenance["input_id"] = "changed"
        sample = SampleManifestEntry(
            sample_id="toy-audio-000000",
            sample_index=3,
            items=(
                ((Role.DEFAULT, Modality.AUDIO), {"label": "speech"}),
            ),
        )
        payload = ViewManifestEntry(
            role=Role.DEFAULT,
            modality=Modality.AUDIO,
            view=AudioView.WAVEFORM,
            sample_index=3,
            shard="000000.tar",
            key="000000.pt",
        )

        self.assertEqual(
            DatasetManifest(**dataset_manifest_dict(manifest)),
            manifest,
        )
        self.assertEqual(manifest.provenance, {"input_id": "input-v1"})
        self.assertEqual(pickle.loads(pickle.dumps(manifest)), manifest)
        with self.assertRaises(FrozenInstanceError):
            manifest.dataset_id = "changed"
        with self.assertRaises(TypeError):
            manifest.provenance["input_id"] = "changed"
        self.assertEqual(ViewManifestEntry(**asdict(payload)), payload)
        self.assertEqual(asdict(sample)["items"][0][1], {"label": "speech"})
        self.assertEqual(
            set(asdict(payload)),
            {
                "role",
                "modality",
                "view",
                "sample_index",
                "shard",
                "key",
            },
        )

    def test_dataset_manifest_owns_its_schema_contract(self):
        cases = (
            (
                {"dataset_id": 1, "sample_count": 0},
                "dataset_id must be a string",
            ),
            (
                {"dataset_id": "toy", "sample_count": True},
                "sample_count must be a non-negative integer",
            ),
            (
                {"dataset_id": "toy", "sample_count": -1},
                "sample_count must be a non-negative integer",
            ),
            (
                {"dataset_id": "toy", "sample_count": 0, "split": 1},
                "split must be a string or None",
            ),
        )
        for values, error in cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, error):
                    DatasetManifest(
                        schema_version=STORE_SCHEMA_VERSION,
                        **values,
                    )

    def test_dataset_manifest_parser_owns_fields_and_version(self):
        base = {
            "dataset_id": "toy",
            "sample_count": 0,
            "schema_version": STORE_SCHEMA_VERSION,
            "split": None,
            "provenance": {},
        }
        with self.assertRaisesRegex(ValueError, "missing field 'split'"):
            dataset_manifest_from_dict(
                {key: value for key, value in base.items() if key != "split"}
            )
        with self.assertRaisesRegex(ValueError, "unsupported field 'extra'"):
            dataset_manifest_from_dict({**base, "extra": True})
        with self.assertRaisesRegex(ValueError, "Unsupported store schema_version"):
            dataset_manifest_from_dict({**base, "schema_version": 1})

    def test_json_and_manifest_helpers_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "nested" / "dataset.json"
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            sample = SampleManifestEntry(
                sample_id="sample-0",
                sample_index=0,
            )
            payload = ViewManifestEntry(
                role=Role.DEFAULT,
                modality=Modality.AUDIO,
                view=AudioView.WAVEFORM,
                sample_index=0,
                shard="000000.tar",
                key="sample-0.pt",
            )

            write_json(path, {"b": 2, "a": 1})
            write_json(path, {"a": 3})
            write_samples_manifest(root, [sample])
            write_view_manifest(root, view, [payload])

            self.assertEqual(read_json(path), {"a": 3})
            self.assertTrue(samples_manifest_exists(root))
            self.assertEqual(tuple(read_samples_manifest(root)), (sample,))
            self.assertEqual(tuple(read_view_manifest(root, view)), (payload,))
            self.assertEqual(sample_manifest_row_groups(root), (1,))
            self.assertEqual(view_manifest_row_count(root, view), 1)
            self.assertEqual(view_manifest_row_groups(root, view), (1,))

    def test_manifest_index_helpers_do_not_materialize_row_dicts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            sample = SampleManifestEntry(
                sample_id="sample-0",
                sample_index=0,
            )
            payload = ViewManifestEntry(
                role=Role.DEFAULT,
                modality=Modality.AUDIO,
                view=AudioView.WAVEFORM,
                sample_index=0,
                shard="000000.tar",
                key="sample-0.pt",
            )
            write_samples_manifest(root, [sample])
            write_view_manifest(root, view, [payload])

            with mock.patch(
                "anydataset.store.manifest.io._read_manifest_rows",
                side_effect=AssertionError("row dicts materialized"),
            ):
                sample_index = tuple(read_sample_manifest_index(root))
                view_indexes = tuple(read_view_manifest_indexes(root, view))

        self.assertEqual(sample_index, ((0, "sample-0"),))
        self.assertEqual(view_indexes, (0,))

    def test_manifest_cache_reuses_handles_until_fingerprint_changes(self):
        import pyarrow.parquet as parquet

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_samples_manifest(
                root,
                [SampleManifestEntry(sample_id="sample-0", sample_index=0)],
            )
            cache = ManifestParquetCache()

            with mock.patch(
                "pyarrow.parquet.ParquetFile",
                wraps=parquet.ParquetFile,
            ) as open_parquet:
                self.assertEqual(
                    tuple(read_samples_manifest(root, cache=cache))[0].sample_id,
                    "sample-0",
                )
                self.assertEqual(
                    tuple(read_samples_manifest(root, cache=cache))[0].sample_id,
                    "sample-0",
                )
                self.assertEqual(open_parquet.call_count, 1)

                path = samples_parquet_path(root)
                stat = path.stat()
                os.utime(
                    path,
                    ns=(stat.st_atime_ns, stat.st_mtime_ns + 1),
                )
                tuple(read_samples_manifest(root, cache=cache))
                self.assertEqual(open_parquet.call_count, 2)

            cache.close()
            self.assertFalse(cache._files)

    def test_manifest_cache_discards_handles_after_fork(self):
        import pyarrow.parquet as parquet

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_samples_manifest(
                root,
                [SampleManifestEntry(sample_id="sample-0", sample_index=0)],
            )
            cache = ManifestParquetCache()
            tuple(read_samples_manifest(root, cache=cache))
            self.assertEqual(len(cache._files), 1)
            cache._pid = -1

            with mock.patch(
                "pyarrow.parquet.ParquetFile",
                wraps=parquet.ParquetFile,
            ) as open_parquet:
                tuple(read_samples_manifest(root, cache=cache))

            self.assertEqual(open_parquet.call_count, 1)
            self.assertEqual(len(cache._files), 1)
            cache.close()


if __name__ == "__main__":
    unittest.main()
