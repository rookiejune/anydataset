from __future__ import annotations

import gc
import multiprocessing
import os
import pickle
import subprocess
import sys
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import torch

from anydataset.store import (
    DatasetWriter,
)
from anydataset.store.manifest.migration import migrate_store
from anydataset.store.jsonio import read_json, write_json
from anydataset.store.manifest.schema import STORE_SCHEMA_VERSION, ViewManifestEntry
from anydataset.store.manifest.io import read_sample_manifest_index
from anydataset.store.paths import view_manifest_parquet_path, view_shard_path
from anydataset.store.payload.files import (
    StoreFilesInUseError,
    cleanup_store_files,
    lease_store_files,
)
from anydataset.store.payload.integrity import (
    validate_store_payloads,
    validate_store_view_payloads,
)
from anydataset.store.reader import read_store_dataset
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextView,
)


class StoreMigrationTest(unittest.TestCase):
    def test_migrate_store_converts_v1_to_independent_v3_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "v1"
            output = root / "v3"
            _write_v1_store(source)

            with self.assertRaisesRegex(ValueError, "expected 2 or 3"):
                read_store_dataset(source)

            migrated = migrate_store(source, output)
            dataset = read_store_dataset(migrated, preload=True)
            first = dataset[0]
            second = dataset[1]

            self.assertEqual(migrated, output.resolve())
            self.assertEqual(read_json(output / "dataset.json")["schema_version"], 3)
            self.assertNotIn("schema_version", read_json(source / "dataset.json"))
            self.assertEqual(
                first[Role.DEFAULT, Modality.TEXT].views[TextView.TEXT],
                "first",
            )
            self.assertTrue(
                torch.equal(
                    second[Role.DEFAULT, Modality.AUDIO].views[AudioView.WAVEFORM][0],
                    torch.tensor([[2.0]]),
                )
            )

            view = (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM)
            source_shard = view_shard_path(source, view, "000000.tar")
            output_shard = view_shard_path(output, view, "000000.tar")
            self.assertNotEqual(source_shard.stat().st_ino, output_shard.stat().st_ino)
            source_shard.write_bytes(b"changed-after-migration")
            self.assertTrue(
                torch.equal(
                    read_store_dataset(output)[0][Role.DEFAULT, Modality.AUDIO].views[
                        AudioView.WAVEFORM
                    ][0],
                    torch.tensor([[1.0]]),
                )
            )

    def test_migrate_store_accepts_explicit_v1_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "v1"
            output = Path(tmpdir) / "v3"
            _write_v1_store(source)
            manifest = read_json(source / "dataset.json")
            manifest["schema_version"] = 1
            write_json(source / "dataset.json", manifest)

            migrate_store(source, output)

            self.assertEqual(
                read_store_dataset(output).manifest.schema_version,
                STORE_SCHEMA_VERSION,
            )

    def test_migrate_store_does_not_publish_invalid_v1_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "v1"
            output = Path(tmpdir) / "v3"
            _write_v1_store(source)
            view = (Role.DEFAULT, Modality.TEXT, TextView.TEXT)
            _replace_first_sample_id(source, view, "unknown")

            with self.assertRaisesRegex(ValueError, "unknown sample_id 'unknown'"):
                migrate_store(source, output)

            self.assertFalse(output.exists())

    def test_migrate_store_rejects_current_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "v3"
            output = Path(tmpdir) / "output"
            _write_store(source)

            with self.assertRaisesRegex(ValueError, "already uses schema_version 3"):
                migrate_store(source, output)

    def test_migrate_store_rejects_boolean_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "v1"
            output = Path(tmpdir) / "output"
            _write_v1_store(source)
            manifest = read_json(source / "dataset.json")
            manifest["schema_version"] = True
            write_json(source / "dataset.json", manifest)

            with self.assertRaisesRegex(
                ValueError,
                "Unsupported source store schema_version: True",
            ):
                migrate_store(source, output)

    def test_migrate_store_rejects_non_portable_v1_path_segments(self):
        view = (Role.DEFAULT, Modality.TEXT, TextView.TEXT)
        for column, value, message in (
            ("shard", "nested\\000000.tar", "invalid shard name"),
            ("key", "nested\\payload.txt", "invalid payload key"),
        ):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as tmpdir:
                source = Path(tmpdir) / "v1"
                output = Path(tmpdir) / "output"
                _write_v1_store(source)
                _replace_first_view_value(source, view, column, value)

                with self.assertRaisesRegex(ValueError, message):
                    migrate_store(source, output)

                self.assertFalse(output.exists())


class StoreIntegrityTest(unittest.TestCase):
    def test_integrity_levels_validate_increasing_payload_detail(self):
        view = (Role.DEFAULT, Modality.TEXT, TextView.TEXT)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "store"
            _write_store(root)
            shard = view_shard_path(root, view, "000000.tar")

            validate_store_payloads((root,), level="full")
            shard.write_bytes(b"not a tar archive")
            validate_store_view_payloads(root, view, level="fast")
            with self.assertRaisesRegex(ValueError, "not a valid tar archive"):
                validate_store_view_payloads(root, view, level="normal")

            with tarfile.open(shard, "w"):
                pass
            validate_store_view_payloads(root, view, level="normal")
            with self.assertRaisesRegex(ValueError, "missing payload"):
                validate_store_view_payloads(root, view, level="full")

    def test_integrity_rejects_unknown_level(self):
        with self.assertRaisesRegex(ValueError, "integrity level"):
            validate_store_payloads((), level="unknown")  # type: ignore[arg-type]

    def test_fast_integrity_rejects_non_portable_manifest_payload_keys(self):
        view = (Role.DEFAULT, Modality.TEXT, TextView.TEXT)
        for key in ("", ".", "..", "nested/payload.txt", "nested\\payload.txt"):
            entry = ViewManifestEntry(*view, 0, "000000.tar", key)
            with self.subTest(key=key), mock.patch(
                "anydataset.store.payload.integrity.read_view_manifest",
                return_value=iter((entry,)),
            ):
                with self.assertRaisesRegex(ValueError, "invalid payload key"):
                    validate_store_view_payloads(Path("unused"), view, level="fast")

    def test_integrity_rejects_duplicate_tar_payload_members(self):
        view = (Role.DEFAULT, Modality.TEXT, TextView.TEXT)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "store"
            _write_store(root)
            shard = view_shard_path(root, view, "000000.tar")
            with tarfile.open(shard, "w") as archive:
                _add_tar_member(archive, "000000000000.txt")
                _add_tar_member(archive, "000000000000.txt")

            validate_store_view_payloads(root, view, level="fast")
            for level in ("normal", "full"):
                with self.subTest(level=level):
                    with self.assertRaisesRegex(ValueError, "duplicate payload key"):
                        validate_store_view_payloads(root, view, level=level)

    def test_integrity_ignores_non_file_member_with_payload_name(self):
        view = (Role.DEFAULT, Modality.TEXT, TextView.TEXT)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "store"
            _write_store(root)
            shard = view_shard_path(root, view, "000000.tar")
            with tarfile.open(shard, "w") as archive:
                _add_tar_member(archive, "000000000000.txt")
                _add_tar_member(archive, "000000000001.txt")
                directory = tarfile.TarInfo("000000000000.txt")
                directory.type = tarfile.DIRTYPE
                archive.addfile(directory)

            for level in ("normal", "full"):
                with self.subTest(level=level):
                    validate_store_view_payloads(root, view, level=level)

    def test_integrity_rejects_non_portable_tar_payload_keys(self):
        view = (Role.DEFAULT, Modality.TEXT, TextView.TEXT)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "store"
            _write_store(root)
            shard = view_shard_path(root, view, "000000.tar")
            with tarfile.open(shard, "w") as archive:
                _add_tar_member(archive, "nested\\payload.txt")

            validate_store_view_payloads(root, view, level="fast")
            for level in ("normal", "full"):
                with self.subTest(level=level):
                    with self.assertRaisesRegex(ValueError, "invalid payload key"):
                        validate_store_view_payloads(root, view, level=level)


class StoreFilesCleanupTest(unittest.TestCase):
    def test_reader_and_explicit_lease_preserve_retained_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "source.wav"
            audio.write_bytes(b"RIFF-data")
            store = root / "store"
            DatasetWriter(store, dataset_id="files").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={AudioView.FILE: audio}
                        )
                    }
                ]
            )
            dataset = read_store_dataset(store)
            retained = Path(
                dataset[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE]
            )

            with self.assertRaises(StoreFilesInUseError):
                cleanup_store_files(store)
            self.assertEqual(retained.read_bytes(), b"RIFF-data")
            result = _cleanup_in_subprocess(store)
            self.assertEqual(result.returncode, 0, result.stderr)

            lease = lease_store_files(store)
            del dataset
            gc.collect()
            with self.assertRaises(StoreFilesInUseError):
                cleanup_store_files(store)
            self.assertEqual(retained.read_bytes(), b"RIFF-data")

            lease.close()
            self.assertTrue(cleanup_store_files(store))
            self.assertFalse(retained.exists())
            self.assertFalse(cleanup_store_files(store))

            restored_dataset = read_store_dataset(store)
            restored = Path(
                restored_dataset[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE]
            )
            self.assertEqual(restored, retained)
            self.assertEqual(restored.read_bytes(), b"RIFF-data")

    def test_reader_lease_is_reacquired_when_unpickled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "source.wav"
            audio.write_bytes(b"RIFF-data")
            store = root / "store"
            DatasetWriter(store, dataset_id="files").write(
                [
                    {
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={AudioView.FILE: audio}
                        )
                    }
                ]
            )
            dataset = read_store_dataset(store)

            restored = pickle.loads(pickle.dumps(dataset))
            del dataset
            gc.collect()

            with self.assertRaises(StoreFilesInUseError):
                cleanup_store_files(store)

            del restored
            gc.collect()
            self.assertFalse(cleanup_store_files(store))

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "requires fork")
    def test_fork_child_close_does_not_release_parent_reader_lease(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = root / "store"
            _write_file_store(store, root / "source.wav")
            dataset = read_store_dataset(store)
            context = multiprocessing.get_context("fork")
            closed = context.Event()
            process = context.Process(
                target=_close_store_files_lease,
                args=(dataset, closed),
            )
            process.start()
            self.assertTrue(closed.wait(timeout=5))
            process.join(timeout=5)
            self.assertEqual(process.exitcode, 0)

            result = _cleanup_in_subprocess(store)
            self.assertEqual(result.returncode, 0, result.stderr)

            del dataset
            gc.collect()
            self.assertFalse(cleanup_store_files(store))

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "requires fork")
    def test_fork_parent_close_does_not_release_child_reader_lease(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = root / "store"
            _write_file_store(store, root / "source.wav")
            dataset = read_store_dataset(store)
            context = multiprocessing.get_context("fork")
            ready = context.Event()
            release = context.Event()
            process = context.Process(
                target=_use_store_files_after_release,
                args=(dataset, ready, release),
            )
            process.start()
            self.assertTrue(ready.wait(timeout=5))

            del dataset
            gc.collect()
            result = _cleanup_in_subprocess(store)
            self.assertEqual(result.returncode, 0, result.stderr)

            release.set()
            process.join(timeout=5)
            self.assertEqual(process.exitcode, 0)
            self.assertTrue(cleanup_store_files(store))


def _write_v1_store(path: Path) -> None:
    _write_store(path)
    manifest = read_json(path / "dataset.json")
    del manifest["schema_version"]
    del manifest["provenance"]
    write_json(path / "dataset.json", manifest)
    for view in (
        (Role.DEFAULT, Modality.AUDIO, AudioView.WAVEFORM),
        (Role.DEFAULT, Modality.TEXT, TextView.TEXT),
    ):
        _rewrite_view_manifest_as_v1(path, view)


def _write_store(path: Path) -> None:
    DatasetWriter(path, dataset_id="migration", split="train").write(
        [
            {
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: (torch.tensor([[1.0]]), 16_000)}
                ),
                (Role.DEFAULT, Modality.TEXT): TextItem(views={TextView.TEXT: "first"}),
            },
            {
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.WAVEFORM: (torch.tensor([[2.0]]), 16_000)}
                ),
                (Role.DEFAULT, Modality.TEXT): TextItem(
                    views={TextView.TEXT: "second"}
                ),
            },
        ]
    )


def _write_file_store(store: Path, audio: Path) -> None:
    audio.write_bytes(b"RIFF-data")
    DatasetWriter(store, dataset_id="files").write(
        [
            {
                (Role.DEFAULT, Modality.AUDIO): AudioItem(
                    views={AudioView.FILE: audio}
                )
            }
        ]
    )


def _close_store_files_lease(dataset, closed) -> None:
    dataset._file_lease.close()
    closed.set()


def _use_store_files_after_release(dataset, ready, release) -> None:
    ready.set()
    if not release.wait(timeout=5):
        raise TimeoutError("store files lease was not released")
    path = Path(dataset[0][Role.DEFAULT, Modality.AUDIO].views[AudioView.FILE])
    if path.read_bytes() != b"RIFF-data":
        raise AssertionError("forked Store reader returned the wrong payload")


def _rewrite_view_manifest_as_v1(root: Path, view) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = view_manifest_parquet_path(root, view)
    table = pq.read_table(path)
    sample_ids = dict(read_sample_manifest_index(root))
    indexes = table["sample_index"].to_pylist()
    columns = {
        "modality": table["modality"],
        "role": table["role"],
        "view": table["view"],
        "sample_id": pa.array([sample_ids[index] for index in indexes]),
        "shard": table["shard"],
        "key": table["key"],
    }
    schema = pa.schema(
        [
            ("modality", pa.string()),
            ("role", pa.string()),
            ("view", pa.string()),
            ("sample_id", pa.string()),
            ("shard", pa.string()),
            ("key", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), path)


def _replace_first_sample_id(root: Path, view, sample_id: str) -> None:
    _replace_first_view_value(root, view, "sample_id", sample_id)


def _replace_first_view_value(
    root: Path,
    view,
    column_name: str,
    value: str,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = view_manifest_parquet_path(root, view)
    table = pq.read_table(path)
    column = table.schema.get_field_index(column_name)
    values = table[column_name].to_pylist()
    values[0] = value
    pq.write_table(
        table.set_column(column, column_name, pa.array(values)),
        path,
    )


def _cleanup_in_subprocess(store: Path) -> subprocess.CompletedProcess[str]:
    source = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (source, env.get("PYTHONPATH", "")) if value
    )
    code = """
import sys
from anydataset.store.payload.files import StoreFilesInUseError, cleanup_store_files

try:
    cleanup_store_files(sys.argv[1])
except StoreFilesInUseError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    return subprocess.run(
        [sys.executable, "-c", code, str(store)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _add_tar_member(archive: tarfile.TarFile, name: str) -> None:
    data = b"payload"
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, BytesIO(data))


if __name__ == "__main__":
    unittest.main()
