import tempfile
import unittest
from pathlib import Path

from anydataset import AudioView, ModalityKey, ViewRef
from anydataset.store import (
    DatasetManifest,
    SampleManifestEntry,
    ViewManifestEntry,
    ViewSelection,
    dataset_json_path,
    dataset_ready_path,
    read_json,
    read_jsonl,
    samples_jsonl_path,
    view_dir,
    view_json_path,
    view_manifest_path,
    view_ready_path,
    view_ref_from_dict,
    view_ref_to_dict,
    view_shard_path,
    view_shards_dir,
    write_json,
    write_jsonl,
)


class StoreTest(unittest.TestCase):
    def test_view_paths_use_structured_view_ref(self):
        root = Path("/tmp/dataset")
        default = ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM)
        source = ViewRef(ModalityKey.AUDIO, AudioView.LONGCAT, role="source")

        self.assertEqual(dataset_json_path(root), root / "dataset.json")
        self.assertEqual(samples_jsonl_path(root), root / "samples.jsonl")
        self.assertEqual(dataset_ready_path(root), root / ".ready")
        self.assertEqual(
            view_dir(root, default, "rev1"),
            root / "audio" / "views" / "waveform" / "rev1",
        )
        self.assertEqual(
            view_dir(root, source, "rev2"),
            root / "audio" / "source" / "views" / "longcat" / "rev2",
        )
        self.assertEqual(
            view_json_path(root, default, "rev1"),
            root / "audio" / "views" / "waveform" / "rev1" / "view.json",
        )
        self.assertEqual(
            view_manifest_path(root, default, "rev1"),
            root / "audio" / "views" / "waveform" / "rev1" / "manifest.jsonl",
        )
        self.assertEqual(
            view_ready_path(root, default, "rev1"),
            root / "audio" / "views" / "waveform" / "rev1" / ".ready",
        )
        self.assertEqual(
            view_shards_dir(root, default, "rev1"),
            root / "audio" / "views" / "waveform" / "rev1" / "shards",
        )
        self.assertEqual(
            view_shard_path(root, default, "rev1", "000000.tar"),
            root / "audio" / "views" / "waveform" / "rev1" / "shards" / "000000.tar",
        )

    def test_view_paths_reject_bad_revision_and_shard(self):
        ref = ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM)

        with self.assertRaises(ValueError):
            view_dir("/tmp/dataset", ref, "bad/rev")

        with self.assertRaises(ValueError):
            view_shard_path("/tmp/dataset", ref, "rev1", "bad/shard.tar")

    def test_manifest_dataclasses_round_trip_to_json_objects(self):
        ref = ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM)
        manifest = DatasetManifest(
            dataset_id="toy-audio",
            split="train",
            sample_count=2,
            views=(ViewSelection(ref, "raw"),),
            config={"task": "audio_codec"},
            provenance={"input": "memory"},
        )

        loaded = DatasetManifest.from_dict(manifest.to_dict())

        self.assertEqual(loaded, manifest)
        self.assertEqual(
            manifest.to_dict()["views"],
            [{"modality": "audio", "role": None, "view_key": "waveform", "revision": "raw"}],
        )

    def test_sample_manifest_entry_round_trip(self):
        entry = SampleManifestEntry(
            sample_id="toy-audio-000000",
            dataset_name="toy:train",
            sample_index=3,
            source={"uri": "memory://toy/3"},
            modality=ModalityKey.AUDIO,
            duration=1.5,
            sample_rate=16000,
            label="speech",
            text="hello",
            metadata={"speaker": "alice"},
        )

        loaded = SampleManifestEntry.from_dict(entry.to_dict())

        self.assertEqual(loaded, entry)
        self.assertEqual(entry.to_dict()["role"], None)

    def test_view_manifest_entry_round_trip(self):
        ref = ViewRef(ModalityKey.AUDIO, AudioView.WAVEFORM)
        entry = ViewManifestEntry(
            ref=ref,
            revision="raw",
            sample_id="toy-audio-000000",
            shard="000000.tar",
            key="000000.wav",
            shape=(1, 16000),
            dtype="float32",
            checksum="sha256:abc",
            provenance={"provider": "writer"},
        )

        loaded = ViewManifestEntry.from_dict(entry.to_dict())

        self.assertEqual(loaded, entry)
        self.assertEqual(entry.to_dict()["shape"], [1, 16000])

    def test_view_ref_json_shape_is_explicit(self):
        ref = ViewRef(ModalityKey.AUDIO, AudioView.FILE, role="target")
        data = view_ref_to_dict(ref)

        self.assertEqual(
            data,
            {"modality": "audio", "role": "target", "view_key": "file"},
        )
        self.assertEqual(view_ref_from_dict(data), ref)

    def test_json_helpers_write_parent_dirs_and_replace_existing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dataset.json"

            write_json(path, {"b": 2, "a": 1})
            write_json(path, {"a": 3})

            self.assertEqual(read_json(path), {"a": 3})

    def test_jsonl_helpers_round_trip_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"

            write_jsonl(path, [{"sample_id": "a"}, {"sample_id": "b"}])

            self.assertEqual(
                list(read_jsonl(path)),
                [{"sample_id": "a"}, {"sample_id": "b"}],
            )


if __name__ == "__main__":
    unittest.main()
