import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from anydataset import AudioView, Modality, Role
from anydataset.store.jsonio import read_json, write_json
from anydataset.store.manifest import (
    DatasetManifest,
    SampleManifestEntry,
    ViewManifestEntry,
)
from anydataset.store.manifestio import (
    read_samples_manifest,
    read_view_manifest,
    samples_manifest_exists,
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
        view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
        manifest = DatasetManifest(
            dataset_id="toy-audio",
            split="train",
            sample_count=2,
        )
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
            sample_id="toy-audio-000000",
            shard="000000.tar",
            key="000000.pt",
        )

        self.assertEqual(DatasetManifest(**asdict(manifest)), manifest)
        self.assertEqual(ViewManifestEntry(**asdict(payload)), payload)
        self.assertEqual(asdict(sample)["items"][0][1], {"label": "speech"})
        self.assertEqual(
            set(asdict(payload)),
            {"role", "modality", "view", "sample_id", "shard", "key"},
        )

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
                sample_id="sample-0",
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


if __name__ == "__main__":
    unittest.main()
