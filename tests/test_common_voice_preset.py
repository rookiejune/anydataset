from pathlib import Path
import tempfile
import unittest
from unittest import mock

from anydataset import (
    AudioMeta,
    AudioView,
    Modality,
    Preset,
    Role,
    TextMeta,
    TextView,
    resolve_dataset,
)
from anydataset.dataset import MultipleAnyDataset
from anydataset.presets import CommonVoice


class CommonVoicePresetTest(unittest.TestCase):
    def test_resolves_latest_common_voice_spec_from_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cv-corpus-23.0-2025-09-17").mkdir()
            (root / "cv-corpus-24.0-2025-12-05").mkdir()

            spec = Preset.COMMON_VOICE.spec(root=root)

        self.assertEqual(spec.source, "tsv")
        self.assertEqual(spec.split, "train")
        self.assertEqual(spec.version, "24.0-2025-12-05")
        self.assertTrue(spec.path.endswith("cv-corpus-24.0-2025-12-05/en"), spec.path)

    def test_resolves_explicit_common_voice_version(self):
        spec = Preset.COMMON_VOICE.spec(
            root="/data",
            language="zh-CN",
            version="23.0-2025-09-17",
        )

        self.assertEqual(spec.version, "23.0-2025-09-17")
        self.assertEqual(spec.path, "/data/cv-corpus-23.0-2025-09-17/zh-CN")

    def test_resolves_common_voice_shorthand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cv-corpus-24.0-2025-12-05").mkdir()
            with mock.patch.dict("os.environ", {"COMMON_VOICE_DATASET_DIR": tmpdir}):
                spec = resolve_dataset("common_voice:dev")

        self.assertEqual(spec.source, "tsv")
        self.assertEqual(spec.split, "dev")
        self.assertTrue(spec.path.endswith("cv-corpus-24.0-2025-12-05/en"), spec.path)

    def test_multilingual_create_returns_multiple_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cv-corpus-24.0-2025-12-05").mkdir()
            dataset = Preset.COMMON_VOICE.create(
                root=root,
                languages=("en", "zh-CN"),
            )
            self.assertIsInstance(dataset, MultipleAnyDataset)
            self.assertEqual(
                [child.spec.path for child in dataset.datasets],
                [
                    str(root / "cv-corpus-24.0-2025-12-05" / "en"),
                    str(root / "cv-corpus-24.0-2025-12-05" / "zh-CN"),
                ],
            )

    def test_single_language_spec_rejects_multiple_languages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cv-corpus-24.0-2025-12-05").mkdir()
            with self.assertRaisesRegex(ValueError, "exactly one language"):
                Preset.COMMON_VOICE.spec(root=root, languages=("en", "zh-CN"))

    def test_missing_auto_version_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "cv-corpus"):
                Preset.COMMON_VOICE.spec(root=tmpdir)

    def test_accepts_corpus_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "cv-corpus-24.0-2025-12-05"
            root.mkdir()
            spec = Preset.COMMON_VOICE.spec(root=root)

            self.assertEqual(spec.path, str(root / "en"))
            self.assertEqual(spec.version, "24.0-2025-12-05")

    def test_reads_common_voice_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            language_root = root / "cv-corpus-24.0-2025-12-05" / "en"
            (language_root / "clips").mkdir(parents=True)
            (language_root / "clips" / "sample.mp3").write_bytes(b"")
            (language_root / "train.tsv").write_text(
                "client_id\tpath\tsentence_id\tsentence\tsentence_domain\t"
                "up_votes\tdown_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n"
                "speaker-1\tsample.mp3\tsentence-1\tHello there.\tgeneral\t"
                "2\t0\tthirties\tfemale\tUnited States English\t\ten\t\n",
                encoding="utf-8",
            )

            dataset = CommonVoice(root=root, cache_root=root / "cache")
            sample = next(iter(dataset))

        audio = sample[Role.DEFAULT, Modality.AUDIO]
        text = sample[Role.DEFAULT, Modality.TEXT]
        self.assertEqual(
            audio.views[AudioView.FILE],
            str(language_root / "clips" / "sample.mp3"),
        )
        self.assertEqual(audio.meta[AudioMeta.LABELS]["client_id"], "speaker-1")
        self.assertEqual(text.views[TextView.TEXT], "Hello there.")
        self.assertEqual(text.meta[TextMeta.LANG], "en")

    def test_requires_root_or_environment(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                ValueError,
                "COMMON_VOICE_DATASET_DIR",
            ):
                Preset.COMMON_VOICE.spec()


if __name__ == "__main__":
    unittest.main()
