from pathlib import Path
import tempfile
import unittest
from unittest import mock

from anydataset import Preset, resolve_dataset
from anydataset.types import (
    AudioMeta,
    AudioView,
    Lang,
    Modality,
    Role,
    TextMeta,
    TextView,
)
from anydataset.presets import CommonVoice


class CommonVoicePresetTest(unittest.TestCase):
    def test_resolves_latest_common_voice_spec_from_root_languages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cv-corpus-23.0-2025-09-17" / "en").mkdir(parents=True)
            (root / "cv-corpus-24.0-2025-12-05" / "zh-CN").mkdir(parents=True)
            (root / "cv-corpus-24.0-2025-12-05" / "en").mkdir()

            spec = Preset.COMMON_VOICE.spec(root=root)

        self.assertEqual(spec.source, "tsv")
        self.assertEqual(spec.split, "train")
        self.assertEqual(spec.version, "24.0-2025-12-05")
        self.assertTrue(spec.path.endswith("cv-corpus-24.0-2025-12-05"), spec.path)
        self.assertEqual(spec.load_options["subdirs"], ("en", "zh-CN"))

    def test_resolves_explicit_common_voice_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cv-corpus-23.0-2025-09-17" / "zh-CN").mkdir(parents=True)

            spec = Preset.COMMON_VOICE.spec(
                root=root,
                language="zh-CN",
                version="23.0-2025-09-17",
            )

        self.assertEqual(spec.version, "23.0-2025-09-17")
        self.assertEqual(spec.path, str(root / "cv-corpus-23.0-2025-09-17"))
        self.assertEqual(spec.load_options["subdirs"], ("zh-CN",))

    def test_resolves_common_voice_shorthand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cv-corpus-24.0-2025-12-05" / "en").mkdir(parents=True)
            with mock.patch.dict("os.environ", {"COMMON_VOICE_DATASET_DIR": tmpdir}):
                spec = resolve_dataset("common_voice:dev")

        self.assertEqual(spec.source, "tsv")
        self.assertEqual(spec.split, "dev")
        self.assertTrue(spec.path.endswith("cv-corpus-24.0-2025-12-05"), spec.path)
        self.assertEqual(spec.load_options["subdirs"], ("en",))

    def test_resolves_explicit_common_voice_languages_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cv-corpus-24.0-2025-12-05" / "zh-CN").mkdir(parents=True)
            (root / "cv-corpus-24.0-2025-12-05" / "en").mkdir()

            spec = Preset.COMMON_VOICE.spec(
                root=root,
                languages=("zh-CN", "en"),
            )

        self.assertTrue(spec.path.endswith("cv-corpus-24.0-2025-12-05"), spec.path)
        self.assertEqual(spec.load_options["subdirs"], ("en", "zh-CN"))

    def test_default_rejects_languages_missing_from_latest_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cv-corpus-23.0-2025-09-17" / "fr").mkdir(parents=True)
            (root / "cv-corpus-24.0-2025-12-05" / "en").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "fr"):
                Preset.COMMON_VOICE.spec(root=root)

    def test_explicit_language_must_exist_in_selected_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "cv-corpus-24.0-2025-12-05" / "en").mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "fr"):
                Preset.COMMON_VOICE.spec(root=root, languages=("fr",))

    def test_rejects_invalid_explicit_languages(self):
        cases = (
            (("en", "en"), ValueError, "Duplicate Common Voice language"),
            (("en", 1), TypeError, "must contain strings"),
            (("en", ""), ValueError, "empty strings"),
        )

        for languages, error, message in cases:
            with self.subTest(languages=languages):
                with self.assertRaisesRegex(error, message):
                    Preset.COMMON_VOICE.spec(root="unused", languages=languages)

    def test_rejects_empty_split(self):
        with self.assertRaisesRegex(ValueError, "split"):
            Preset.COMMON_VOICE.spec(split="")

    def test_missing_auto_version_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "cv-corpus"):
                Preset.COMMON_VOICE.spec(root=tmpdir)

    def test_accepts_corpus_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "cv-corpus-24.0-2025-12-05"
            (root / "en").mkdir(parents=True)
            spec = Preset.COMMON_VOICE.spec(root=root)

            self.assertEqual(spec.path, str(root))
            self.assertEqual(spec.version, "24.0-2025-12-05")
            self.assertEqual(spec.load_options["subdirs"], ("en",))

    def test_reads_common_voice_samples_by_language_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            corpus = root / "cv-corpus-24.0-2025-12-05"
            en_root = corpus / "en"
            zh_root = corpus / "zh-CN"
            _write_common_voice_tsv(en_root, "en.mp3", "Hello there.", "en")
            _write_common_voice_tsv(zh_root, "zh.mp3", "Ni hao.", "zh-CN")

            dataset = CommonVoice(root=root, languages=("zh-CN", "en"))
            samples = list(dataset)

        audio = samples[0][Role.DEFAULT, Modality.AUDIO]
        text = samples[0][Role.DEFAULT, Modality.TEXT]
        self.assertEqual(
            audio.views[AudioView.FILE],
            str(en_root / "clips" / "en.mp3"),
        )
        self.assertEqual(audio.meta[AudioMeta.SPEAKER_ID], "speaker-1")
        self.assertNotIn("client_id", audio.meta[AudioMeta.LABELS])
        self.assertEqual(text.views[TextView.TEXT], "Hello there.")
        self.assertEqual(text.meta[TextMeta.LANG], Lang.EN)

        zh_text = samples[1][Role.DEFAULT, Modality.TEXT]
        self.assertEqual(zh_text.views[TextView.TEXT], "Ni hao.")
        self.assertEqual(zh_text.meta[TextMeta.LANG], Lang.ZH)

    def test_requires_root_or_environment(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                ValueError,
                "COMMON_VOICE_DATASET_DIR",
            ):
                Preset.COMMON_VOICE.spec()


def _write_common_voice_tsv(
    language_root: Path,
    audio_name: str,
    sentence: str,
    locale: str,
) -> None:
    (language_root / "clips").mkdir(parents=True)
    (language_root / "clips" / audio_name).write_bytes(b"")
    language_root.joinpath("train.tsv").write_text(
        "client_id\tpath\tsentence_id\tsentence\tsentence_domain\t"
        "up_votes\tdown_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n"
        f"speaker-1\t{audio_name}\tsentence-1\t{sentence}\tgeneral\t"
        f"2\t0\tthirties\tfemale\tUnited States English\t\t{locale}\t\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
