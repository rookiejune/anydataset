import sys
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import torch
from torch import Tensor

from anydataset import AudioItem, AudioKey, AudioView, Modality, Role, ViewRef
from anydataset.provider.longcat import LongCatViewProvider
from anydataset.store import DatasetWriter, ViewInput
from anydataset.store.jsonio import read_json
from anydataset.store.manifest import DatasetManifest, SampleItemEntry, SampleManifestEntry
from anydataset.store.manifestio import read_view_manifest
from anydataset.store.paths import dataset_json_path, view_shard_path


class FakeLongCatCodec:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int, int | None]] = []

    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
        *,
        n_acoustic_codebooks: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        self.calls.append((tuple(audio.shape), sample_rate, n_acoustic_codebooks))
        return (
            torch.tensor([[1, 2, 3]], device=audio.device),
            torch.tensor([[[4, 5, 6], [7, 8, 9]]], device=audio.device),
        )


class LongCatViewProviderTest(unittest.TestCase):
    def test_materializer_writes_longcat_codes_with_external_codec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [
                    _audio_sample(
                        waveform=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
                        sample_rate=16000,
                    )
                ]
            )

            codec = FakeLongCatCodec()
            provider = LongCatViewProvider(
                codec=codec,
                n_acoustic_codebooks=2,
                provider_version="fake",
            )
            provider.materializer(source, target).write()

            manifest = DatasetManifest.from_dict(read_json(dataset_json_path(target)))
            output_ref = ViewRef(Modality.AUDIO, AudioView.LONGCAT)
            revision = next(
                selection.revision
                for selection in manifest.views
                if selection.ref == output_ref
            )
            entry = next(read_view_manifest(target, output_ref, revision))
            with tarfile.open(
                view_shard_path(target, output_ref, entry.revision, entry.shard),
                "r",
            ) as archive:
                payload = archive.extractfile(entry.key).read()
            loaded = torch.load(BytesIO(payload), map_location="cpu")

            self.assertEqual(codec.calls, [((1, 1, 4), 16000, 2)])
            self.assertTrue(
                torch.equal(loaded["semantic_codes"], torch.tensor([[1, 2, 3]]))
            )
            self.assertTrue(
                torch.equal(
                    loaded["acoustic_codes"],
                    torch.tensor([[[4, 5, 6], [7, 8, 9]]]),
                )
            )
            self.assertEqual(loaded["sample_rate"], 16000)
            self.assertEqual(entry.provenance["name"], "longcat")
            self.assertEqual(entry.provenance["version"], "fake")
            self.assertEqual(entry.provenance["config"], {"n_acoustic_codebooks": 2})

    def test_materializer_accepts_file_input_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = root / "audio.wav"
            audio_path.write_bytes(b"fake-wav")
            source = root / "source"
            target = root / "target"
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [_audio_sample(file=str(audio_path), sample_rate=16000)]
            )

            class FakeTorchAudio:
                @staticmethod
                def load(source):
                    self.assertIsInstance(source, BytesIO)
                    self.assertEqual(source.getvalue(), b"fake-wav")
                    return torch.tensor([[0.1, 0.2, 0.3, 0.4]]), 24000

            codec = FakeLongCatCodec()
            provider = LongCatViewProvider(codec=codec, provider_version="fake")
            with patch.dict(sys.modules, {"torchaudio": FakeTorchAudio()}):
                provider.materializer(
                    source,
                    target,
                    input_ref=ViewRef(Modality.AUDIO, AudioView.FILE),
                ).write()

            self.assertEqual(codec.calls, [((1, 1, 4), 24000, None)])

    def test_provider_requires_sample_rate(self):
        provider = LongCatViewProvider(codec=FakeLongCatCodec())
        sample = SampleManifestEntry(
            sample_id="sample",
            dataset_name="dataset",
            items=(SampleItemEntry(ref=(Role.DEFAULT, Modality.AUDIO)),),
        )
        ref = ViewRef(Modality.AUDIO, AudioView.WAVEFORM)

        with self.assertRaisesRegex(ValueError, "sample_rate"):
            provider(
                ViewInput(sample=sample, ref=ref, revision="raw", value=torch.zeros(4))
            )

    def test_call_for_waveform_accepts_single_sample_shapes(self):
        codec = FakeLongCatCodec()
        provider = LongCatViewProvider(codec=codec)

        provider.call_for_waveform(torch.zeros(4), sample_rate=16000)
        provider.call_for_waveform(torch.zeros(1, 4), sample_rate=16000)

        self.assertEqual(
            codec.calls,
            [
                ((1, 1, 4), 16000, None),
                ((1, 1, 4), 16000, None),
            ],
        )

    def test_call_for_batch_accepts_explicit_batch_shape(self):
        codec = FakeLongCatCodec()
        provider = LongCatViewProvider(codec=codec)

        provider.call_for_batch(torch.zeros(2, 1, 4), sample_rate=16000)

        self.assertEqual(codec.calls, [((2, 1, 4), 16000, None)])

    def test_call_for_waveform_rejects_batched_input(self):
        provider = LongCatViewProvider(codec=FakeLongCatCodec())

        with self.assertRaisesRegex(ValueError, r"\[time\] or \[channel, time\]"):
            provider.call_for_waveform(torch.zeros(1, 1, 4), sample_rate=16000)

    def test_call_for_waveform_requires_mono_audio(self):
        provider = LongCatViewProvider(codec=FakeLongCatCodec())

        with self.assertRaisesRegex(ValueError, "one channel"):
            provider.call_for_waveform(torch.zeros(2, 4), sample_rate=16000)

    def test_call_for_batch_requires_mono_audio(self):
        provider = LongCatViewProvider(codec=FakeLongCatCodec())

        with self.assertRaisesRegex(ValueError, "one channel"):
            provider.call_for_batch(torch.zeros(1, 2, 4), sample_rate=16000)

    def test_decoders_must_not_be_single_string(self):
        with self.assertRaisesRegex(TypeError, "sequence"):
            LongCatViewProvider(codec=FakeLongCatCodec(), decoders="16k_4codebooks")

    def test_n_acoustic_codebooks_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            LongCatViewProvider(codec=FakeLongCatCodec(), n_acoustic_codebooks=0)


def _audio_sample(
    *,
    waveform=None,
    file=None,
    sample_rate: int,
):
    views = {}
    if waveform is not None:
        views[AudioView.WAVEFORM] = waveform
    if file is not None:
        views[AudioView.FILE] = file
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views=views,
            required={AudioKey.SAMPLE_RATE: sample_rate},
        )
    }


if __name__ == "__main__":
    unittest.main()
