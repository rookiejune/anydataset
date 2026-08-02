from __future__ import annotations

import tarfile
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import torch

from anydataset.store.reader import read_store_dataset
from anydataset.store.manifest.schema import ViewManifestEntry
from anydataset.store.paths import view_shard_index_path, view_shard_path
from anydataset.store.payload.archive import (
    Payload,
    PayloadCache,
    add_payload,
    payload_value,
    read_payload_bytes,
    write_payload_index,
)
from anydataset.store.writer import DatasetWriter
from anydataset.store.jsonio import read_json, write_json
from anydataset.types import AudioItem, AudioView, Modality, Role


class PayloadCacheTest(unittest.TestCase):
    def test_payload_value_uses_safe_weights_only_mode_by_default(self):
        view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
        with mock.patch(
            "anydataset.store.payload.archive.torch.load", return_value="loaded"
        ) as load:
            self.assertEqual(payload_value(view, b"payload"), "loaded")

        self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_payload_value_uses_unsafe_pickle_when_explicit(self):
        view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
        with mock.patch(
            "anydataset.store.payload.archive.torch.load", return_value="loaded"
        ) as load:
            self.assertEqual(
                payload_value(view, b"payload", unsafe_pickle=True),
                "loaded",
            )

        self.assertFalse(load.call_args.kwargs["weights_only"])

    def test_payload_cache_close_replaces_inherited_lock_after_fork(self):
        cache = PayloadCache()
        inherited_lock = cache._lock
        locked = threading.Event()
        release = threading.Event()

        def hold_lock():
            with inherited_lock:
                locked.set()
                release.wait()

        thread = threading.Thread(target=hold_lock)
        thread.start()
        self.assertTrue(locked.wait(timeout=1))
        try:
            cache._pid = -1
            cache.close()

            self.assertIsNot(cache._lock, inherited_lock)
            self.assertTrue(cache._lock.acquire(blocking=False))
            cache._lock.release()
        finally:
            release.set()
            thread.join(timeout=1)
        self.assertFalse(thread.is_alive())

    def test_payload_lookup_uses_cached_tarinfo_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "store"
            ref = (Role.DEFAULT, Modality.AUDIO)
            DatasetWriter(output, dataset_id="payload-index").write(
                [
                    {
                        ref: AudioItem(
                            views={
                                AudioView.WAVEFORM: (
                                    torch.tensor([[float(index)]]),
                                    16_000,
                                )
                            }
                        )
                    }
                    for index in range(2)
                ]
            )
            dataset = read_store_dataset(output)

            with mock.patch.object(
                tarfile.TarFile,
                "_getmember",
                side_effect=AssertionError("linear tar member lookup"),
            ):
                first = dataset[0][ref].views[AudioView.WAVEFORM][0]
                second = dataset[1][ref].views[AudioView.WAVEFORM][0]

        self.assertTrue(torch.equal(first, torch.tensor([[0.0]])))
        self.assertTrue(torch.equal(second, torch.tensor([[1.0]])))

    def test_payload_lookup_uses_index_for_each_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "store"
            ref = (Role.DEFAULT, Modality.AUDIO)
            DatasetWriter(
                output,
                dataset_id="payload-index-shards",
                max_shard_samples=1,
            ).write(
                [
                    {
                        ref: AudioItem(
                            views={
                                AudioView.WAVEFORM: (
                                    torch.tensor([[float(index)]]),
                                    16_000,
                                )
                            }
                        )
                    }
                    for index in range(2)
                ]
            )
            shards = output / "default" / "audio" / "waveform" / "shards"
            self.assertTrue((shards / "000000.tar.index.json").is_file())
            self.assertTrue((shards / "000001.tar.index.json").is_file())
            dataset = read_store_dataset(output)

            with mock.patch.object(
                tarfile.TarFile,
                "getmembers",
                side_effect=AssertionError("sidecar index was ignored"),
            ):
                first = dataset[0][ref].views[AudioView.WAVEFORM][0]
                second = dataset[1][ref].views[AudioView.WAVEFORM][0]

        self.assertTrue(torch.equal(first, torch.tensor([[0.0]])))
        self.assertTrue(torch.equal(second, torch.tensor([[1.0]])))

    def test_payload_lookup_falls_back_for_corrupt_offset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "store"
            ref = (Role.DEFAULT, Modality.AUDIO)
            DatasetWriter(output, dataset_id="payload-index-corrupt").write(
                [
                    {
                        ref: AudioItem(
                            views={
                                AudioView.WAVEFORM: (
                                    torch.tensor([[1.0]]),
                                    16_000,
                                )
                            }
                        )
                    }
                ]
            )
            sidecar = (
                output
                / "default"
                / "audio"
                / "waveform"
                / "shards"
                / "000000.tar.index.json"
            )
            data = read_json(sidecar)
            member = next(iter(data["members"].values()))
            member["offset"] += tarfile.BLOCKSIZE
            write_json(sidecar, data)
            dataset = read_store_dataset(output)

            with mock.patch.object(
                PayloadCache,
                "_load_members",
                wraps=PayloadCache._load_members,
            ) as load_members:
                waveform = dataset[0][ref].views[AudioView.WAVEFORM][0]

        self.assertTrue(torch.equal(waveform, torch.tensor([[1.0]])))
        self.assertEqual(load_members.call_count, 1)

    def test_payload_index_rejects_duplicate_member_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            shard = "000000.tar"
            path = view_shard_path(root, view, shard)
            path.parent.mkdir(parents=True)
            with tarfile.open(path, "w") as archive:
                add_payload(archive, Payload("duplicate.pt", b"first"))
                add_payload(archive, Payload("duplicate.pt", b"later"))

            with self.assertRaisesRegex(ValueError, "duplicate payload key"):
                write_payload_index(root, view, shard)

    def test_payload_keys_must_be_portable_path_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "payloads.tar"
            with tarfile.open(path, "w") as archive:
                for key in ("", ".", "..", "nested/value.pt", "nested\\value.pt"):
                    with self.subTest(key=key):
                        with self.assertRaises((TypeError, ValueError)):
                            add_payload(archive, Payload(key, b"payload"))

    def test_payload_index_rejects_non_portable_member_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            shard = "000000.tar"
            path = view_shard_path(root, view, shard)
            path.parent.mkdir(parents=True)
            with tarfile.open(path, "w") as archive:
                info = tarfile.TarInfo("nested\\payload.pt")
                info.size = len(b"payload")
                archive.addfile(info, BytesIO(b"payload"))

            with self.assertRaisesRegex(ValueError, "path separators"):
                write_payload_index(root, view, shard)

    def test_payload_lookup_rejects_duplicate_members_without_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            shard = "000000.tar"
            path = view_shard_path(root, view, shard)
            path.parent.mkdir(parents=True)
            with tarfile.open(path, "w") as archive:
                add_payload(archive, Payload("duplicate.pt", b"first"))
                add_payload(archive, Payload("duplicate.pt", b"later"))
            entry = ViewManifestEntry(*view, 0, shard, "duplicate.pt")

            with self.assertRaisesRegex(ValueError, "duplicate payload key"):
                PayloadCache().read(root, view, entry)

    def test_payload_lookup_ignores_non_file_member_with_same_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            shard = "000000.tar"
            path = view_shard_path(root, view, shard)
            path.parent.mkdir(parents=True)
            with tarfile.open(path, "w") as archive:
                add_payload(archive, Payload("duplicate.pt", b"payload"))
                directory = tarfile.TarInfo("duplicate.pt")
                directory.type = tarfile.DIRTYPE
                archive.addfile(directory)
            entry = ViewManifestEntry(*view, 0, shard, "duplicate.pt")

            write_payload_index(root, view, shard)
            for cache in (None, PayloadCache()):
                with self.subTest(cache=cache is not None):
                    self.assertEqual(
                        read_payload_bytes(root, view, entry, cache=cache),
                        b"payload",
                    )

    def test_payload_lookup_ignores_sidecar_with_non_portable_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            shard = "000000.tar"
            path = view_shard_path(root, view, shard)
            path.parent.mkdir(parents=True)
            with tarfile.open(path, "w") as archive:
                add_payload(archive, Payload("payload.pt", b"payload"))
            write_payload_index(root, view, shard)
            index_path = view_shard_index_path(root, view, shard)
            index = read_json(index_path)
            index["members"]["nested\\payload.pt"] = index["members"].pop(
                "payload.pt"
            )
            write_json(index_path, index)
            entry = ViewManifestEntry(*view, 0, shard, "payload.pt")
            cache = PayloadCache()

            with mock.patch.object(
                PayloadCache,
                "_load_members",
                wraps=PayloadCache._load_members,
            ) as load_members:
                data = cache.read(root, view, entry)
            cache.close()

        self.assertEqual(data, b"payload")
        self.assertEqual(load_members.call_count, 1)

    def test_payload_lookup_ignores_sidecar_with_boolean_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            shard = "000000.tar"
            path = view_shard_path(root, view, shard)
            path.parent.mkdir(parents=True)
            with tarfile.open(path, "w") as archive:
                add_payload(archive, Payload("payload.pt", b"payload"))
            write_payload_index(root, view, shard)
            index_path = view_shard_index_path(root, view, shard)
            index = read_json(index_path)
            index["version"] = True
            write_json(index_path, index)
            entry = ViewManifestEntry(*view, 0, shard, "payload.pt")
            cache = PayloadCache()

            with mock.patch.object(
                PayloadCache,
                "_load_members",
                wraps=PayloadCache._load_members,
            ) as load_members:
                data = cache.read(root, view, entry)
            cache.close()

        self.assertEqual(data, b"payload")
        self.assertEqual(load_members.call_count, 1)


if __name__ == "__main__":
    unittest.main()
