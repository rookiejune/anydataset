import tempfile
import unittest
from io import BytesIO
from pathlib import Path
import tarfile

import torch
from torch import Tensor

from anydataset import AudioKey, AudioView, ModalityKey
from anydataset.providers.longcat import LongCatViewProvider
from anydataset.samples import Sample
from anydataset.store import (
    DatasetManifest,
    DatasetWriter,
    SampleManifestEntry,
    ViewInput,
    ViewManifestEntry,
    dataset_json_path,
    read_json,
    read_jsonl,
    view_manifest_path,
    view_ref_from_dict,
    view_shard_path,
)


class FakeLongCatCodec:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int | None, int | None]] = []

    def encode(
        self,
        audio: Tensor,
        sample_rate: int | None = None,
        *,
        n_acoustic_codebooks: int | None = None,
    ):
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
                    Sample(
                        data={
                            ModalityKey.AUDIO: {
                                AudioKey.SAMPLE_RATE: 16000,
                                AudioKey.VIEWS: {
                                    AudioView.WAVEFORM: torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
                                },
                            },
                        },
                        dataset_name="source:train",
                        sample_index=0,
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
            output_ref = view_ref_from_dict(
                {"modality": ModalityKey.AUDIO, "role": None, "view_key": AudioView.LONGCAT}
            )
            longcat_revision = next(
                selection.revision for selection in manifest.views if selection.ref == output_ref
            )
            entry = ViewManifestEntry.from_dict(
                next(read_jsonl(view_manifest_path(target, output_ref, longcat_revision)))
            )
            with tarfile.open(
                view_shard_path(target, output_ref, entry.revision, entry.shard),
                "r",
            ) as archive:
                payload = archive.extractfile(entry.key).read()
            loaded = torch.load(BytesIO(payload), map_location="cpu")

            self.assertEqual(codec.calls, [((1, 1, 4), 16000, 2)])
            self.assertTrue(torch.equal(loaded["semantic_codes"], torch.tensor([[1, 2, 3]])))
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

    def test_provider_requires_sample_rate(self):
        provider = LongCatViewProvider(codec=FakeLongCatCodec())
        sample = SampleManifestEntry(sample_id="sample", dataset_name="dataset")
        ref = view_ref_from_dict(
            {"modality": ModalityKey.AUDIO, "role": None, "view_key": AudioView.WAVEFORM}
        )

        with self.assertRaisesRegex(ValueError, "sample_rate"):
            provider(ViewInput(sample=sample, ref=ref, revision="raw", value=torch.zeros(4)))

    def test_decoders_must_not_be_single_string(self):
        with self.assertRaisesRegex(TypeError, "sequence"):
            LongCatViewProvider(codec=FakeLongCatCodec(), decoders="16k_4codebooks")


if __name__ == "__main__":
    unittest.main()
