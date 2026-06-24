import tempfile
import types
import unittest
from unittest import mock

from anydataset.api.cache import CacheManager
from anydataset.adapters import FSD50KAdapter, fsd50k_spec


class FSD50KAdapterTest(unittest.TestCase):
    def test_prepare_caches_split_file_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = fsd50k_spec(split="dev")
            cache = CacheManager(tmpdir).prepare(spec)
            dataset = FSD50KAdapter()
            with mock.patch(
                "anydataset.adapters.fsd50k._list_fsd50k_files",
                return_value=["clips/dev/a.wav", "clips/dev/b.wav"],
            ) as list_files:
                first = dataset.prepare(spec, cache)
                second = dataset.prepare(spec, cache)

        self.assertEqual(first["files"], ["clips/dev/a.wav", "clips/dev/b.wav"])
        self.assertEqual(second["files"], ["clips/dev/a.wav", "clips/dev/b.wav"])
        self.assertEqual(list_files.call_count, 1)

    def test_iter_indexed_samples_downloads_only_selected_shard(self):
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.hf_hub_download = mock.Mock(
            side_effect=lambda filename, **kwargs: f"/tmp/{filename.rsplit('/', 1)[-1]}"
        )
        manifest = {
            "repo_id": "Fhrozen/FSD50k",
            "cache_path": "/tmp/cache",
            "files": [
                "clips/dev/a.wav",
                "clips/dev/b.wav",
                "clips/dev/c.wav",
            ],
        }

        with mock.patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
            with mock.patch(
                "anydataset.adapters.fsd50k._load_audio",
                return_value=([0.1, 0.2], 44100),
            ):
                rows = list(
                    FSD50KAdapter().iter_indexed_samples(
                        manifest,
                        num_shards=2,
                        shard_id=1,
                    )
                )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 1)
        self.assertEqual(rows[0][1]["path"], "clips/dev/b.wav")
        self.assertEqual(rows[0][1]["audio_path"], "/tmp/b.wav")
        self.assertEqual(rows[0][1]["audio"]["array"], [0.1, 0.2])
        fake_hub.hf_hub_download.assert_called_once()
        self.assertEqual(
            fake_hub.hf_hub_download.call_args.kwargs["filename"],
            "clips/dev/b.wav",
        )

    def test_rejects_unknown_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = fsd50k_spec(split="train")
            cache = CacheManager(tmpdir).prepare(spec)
            with self.assertRaises(ValueError):
                FSD50KAdapter().prepare(spec, cache)


if __name__ == "__main__":
    unittest.main()
