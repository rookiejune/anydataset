import sys
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import Tensor

from anydataset import (
    AnyDataset,
    AudioItem,
    AudioView,
    Modality,
    Role,
    Source,
    Spec,
)
from anydataset.provider.longcat import LongCatViewProvider
from anydataset.store import DatasetWriter, ViewMaterializer
from anydataset.store.manifestio import read_view_manifest
from anydataset.store.paths import view_shard_path


class FakeLongCatCodec:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.dtypes: list[torch.dtype] = []

    def eval(self) -> None:
        return None

    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> tuple[Tensor, Tensor]:
        self.calls.append((tuple(audio.shape), sample_rate))
        self.dtypes.append(audio.dtype)
        return (
            torch.tensor([[1, 2, 3]], device=audio.device),
            torch.tensor([[[4, 5, 6], [7, 8, 9]]], device=audio.device),
        )


class FakeLongCatCodecLoader:
    calls: list[dict[str, object]] = []
    codec = FakeLongCatCodec()

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls.codec


class LongCatViewProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeLongCatCodecLoader.calls = []
        FakeLongCatCodecLoader.codec = FakeLongCatCodec()

    def test_materializer_writes_longcat_codes_with_loaded_codec(self):
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

            with _fake_anytrain(FakeLongCatCodecLoader):
                provider = LongCatViewProvider()
                ViewMaterializer(target, dataset_id="toy-audio", split="train").write(
                    _store_dataset(source, root),
                    provider,
                )

            view = (Role.DEFAULT, Modality.AUDIO, AudioView.LONGCAT)
            entry = next(read_view_manifest(target, view))
            with tarfile.open(
                view_shard_path(target, view, entry.shard),
                "r",
            ) as archive:
                payload = archive.extractfile(entry.key).read()
            loaded = torch.load(BytesIO(payload), map_location="cpu")

            self.assertEqual(FakeLongCatCodecLoader.codec.calls, [((1, 1, 4), 16000)])
            self.assertTrue(
                torch.equal(loaded["semantic_codes"], torch.tensor([1, 2, 3]))
            )
            self.assertTrue(
                torch.equal(
                    loaded["acoustic_codes"],
                    torch.tensor([[4, 5, 6], [7, 8, 9]]),
                )
            )
            self.assertNotIn("sample_rate", loaded)

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
                    self.assertEqual(Path(source).read_bytes(), b"fake-wav")
                    return torch.tensor([[0.1, 0.2, 0.3, 0.4]]), 24000

            with patch("anydataset.provider.abc.torchaudio", FakeTorchAudio()):
                with _fake_anytrain(FakeLongCatCodecLoader):
                    provider = LongCatViewProvider()
                    ViewMaterializer(target, dataset_id="toy-audio", split="train").write(
                        _store_dataset(source, root),
                        provider,
                    )

            self.assertEqual(FakeLongCatCodecLoader.codec.calls, [((1, 1, 4), 24000)])

    def test_provider_batches_waveform_without_shape_policy(self):
        with _fake_anytrain(FakeLongCatCodecLoader):
            provider = LongCatViewProvider()

        provider({AudioView.WAVEFORM: (torch.zeros(1, 4), 16000)})
        provider({AudioView.WAVEFORM: (torch.zeros(2, 4), 16000)})

        self.assertEqual(
            FakeLongCatCodecLoader.codec.calls,
            [
                ((1, 1, 4), 16000),
                ((1, 2, 4), 16000),
            ],
        )

    def test_provider_accepts_numpy_waveform(self):
        with _fake_anytrain(FakeLongCatCodecLoader):
            provider = LongCatViewProvider()

        provider({AudioView.WAVEFORM: (np.array([0.1, 0.2, 0.3, 0.4]), 16000)})

        self.assertEqual(FakeLongCatCodecLoader.codec.calls, [((1, 1, 4), 16000)])
        self.assertEqual(FakeLongCatCodecLoader.codec.dtypes, [torch.float32])

    def test_loads_anytrain_codec_with_decoder_config(self):
        with _fake_anytrain(FakeLongCatCodecLoader):
            provider = LongCatViewProvider(
                cache_dir="/cache",
                decoders=("24k_2codebooks",),
                device="cpu",
                local_files_only=True,
                force_download=True,
            )

        self.assertIs(provider.longcat_codec, FakeLongCatCodecLoader.codec)
        self.assertEqual(
            FakeLongCatCodecLoader.calls,
            [
                {
                    "cache_dir": "/cache",
                    "decoders": ("24k_2codebooks",),
                    "device": "cpu",
                    "local_files_only": True,
                    "force_download": True,
                }
            ],
        )


def _audio_sample(
    *,
    waveform=None,
    file=None,
    sample_rate: int,
):
    views = {}
    if waveform is not None:
        views[AudioView.WAVEFORM] = (waveform, sample_rate)
    if file is not None:
        views[AudioView.FILE] = file
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views=views,
        )
    }


def _store_dataset(path: Path, root: Path):
    return AnyDataset(
        Spec(source=Source.STORE, path=str(path), split="train"),
        cache_root=root / "cache-source",
    )


class _fake_anytrain:
    def __init__(self, codec_class) -> None:
        self.codec_class = codec_class
        self.previous = {}

    def __enter__(self):
        import types

        modules = {
            "anytrain": types.ModuleType("anytrain"),
            "anytrain.codec": types.ModuleType("anytrain.codec"),
            "anytrain.codec.longcat": types.ModuleType("anytrain.codec.longcat"),
        }
        modules["anytrain.codec.longcat"].LongCatAudioCodec = self.codec_class
        self.previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, module in self.previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module




if __name__ == "__main__":
    unittest.main()
