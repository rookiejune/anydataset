from __future__ import annotations

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

from anydataset import AnyDataset, Source, Spec
from anydataset.dataset import collate_fn
from anydataset.types import (
    AudioItem,
    AudioReq,
    AudioView,
    Modality,
    Role,
)
from anydataset.provider.codec import CodecProvider
from anydataset.provider.longcat import LongCatProvider
from anydataset.store import DatasetWriter, ViewMaterializer
from anydataset.store.manifestio import read_view_manifest
from anydataset.store.paths import view_shard_path


class FakeLongCatCodec:
    codebook_sizes = (8192, 90, 90)

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.dtypes: list[torch.dtype] = []

    def eval(self) -> None:
        return None

    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> Tensor:
        self.calls.append((tuple(audio.shape), sample_rate))
        self.dtypes.append(audio.dtype)
        return torch.tensor(
            [[[1, 4, 7], [2, 5, 8], [3, 6, 9]]],
            device=audio.device,
        )


class FakeLongCatCodecLoader:
    calls: list[dict[str, object]] = []
    codec = FakeLongCatCodec()

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls.codec


class FakeBatchedLongCatCodec:
    codebook_sizes = (8192, 90, 90)

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def eval(self) -> None:
        return None

    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
    ) -> Tensor:
        self.calls.append((tuple(audio.shape), sample_rate))
        batch_size = audio.shape[0]
        return torch.tensor(
            [
                [[10, 30, 40], [11, 31, 41], [12, 32, 42], [13, 33, 43]],
                [[20, 50, 60], [21, 51, 61], [22, 52, 62], [23, 53, 63]],
            ],
            device=audio.device,
        )[:batch_size]


class FakeBatchedLongCatCodecLoader:
    calls: list[dict[str, object]] = []
    codec = FakeBatchedLongCatCodec()

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls.codec


class FakeCodec:
    def __init__(self, codes: Tensor, codebook_sizes: tuple[int, ...]) -> None:
        self.codes = codes
        self.codebook_sizes = codebook_sizes

    def eval(self) -> None:
        return None

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        return self.codes.to(audio.device)


class CodecProviderTest(unittest.TestCase):
    def test_accepts_ids_at_each_codebook_boundary(self):
        codes = torch.tensor([[[0, 0, 0], [3, 4, 8]]])
        provider = CodecProvider(FakeCodec(codes, (4, 5, 9)), AudioView.STABLE)

        output = provider({AudioView.WAVEFORM: (torch.zeros(4), 16000)})

        self.assertTrue(torch.equal(output, codes[0]))

    def test_single_rejects_negative_id_for_its_codebook(self):
        codes = torch.tensor([[[0, 4, 8], [3, -1, 0]]])
        provider = CodecProvider(FakeCodec(codes, (4, 5, 9)), AudioView.STABLE)

        with self.assertRaisesRegex(
            ValueError,
            r"codebook 1 observed \[-1, 4\], expected \[0, 5\)",
        ):
            provider({AudioView.WAVEFORM: (torch.zeros(4), 16000)})

    def test_batch_rejects_upper_bound_id_for_its_codebook(self):
        codes = torch.tensor([[[0, 0, 0]], [[3, 5, 8]]])
        provider = CodecProvider(FakeCodec(codes, (4, 5, 9)), AudioView.STABLE)

        with self.assertRaisesRegex(
            ValueError,
            r"codebook 1 observed \[0, 5\], expected \[0, 5\)",
        ):
            provider.call_batch(
                _provider_batch(
                    _audio_sample(waveform=torch.zeros(4), sample_rate=16000),
                    _audio_sample(waveform=torch.zeros(4), sample_rate=16000),
                )
            )


class LongCatProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeLongCatCodecLoader.calls = []
        FakeLongCatCodecLoader.codec = FakeLongCatCodec()
        FakeBatchedLongCatCodecLoader.calls = []
        FakeBatchedLongCatCodecLoader.codec = FakeBatchedLongCatCodec()

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
                ViewMaterializer(target, split="train").write(
                    dataset_factory=_StoreDatasetFactory(source, root),
                    provider_factory=_LongCatProviderFactory(),
                    devices="cpu",
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
                torch.equal(
                    loaded,
                    torch.tensor([[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
                )
            )

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
                    ViewMaterializer(target, split="train").write(
                        dataset_factory=_StoreDatasetFactory(source, root),
                        provider_factory=_LongCatProviderFactory(),
                        devices="cpu",
                    )

            self.assertEqual(FakeLongCatCodecLoader.codec.calls, [((1, 1, 4), 24000)])

    def test_materializer_batches_file_input_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_path = root / "first.wav"
            second_path = root / "second.wav"
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")
            source = root / "source"
            target = root / "target"
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [
                    _audio_sample(file=str(first_path), sample_rate=16000),
                    _audio_sample(file=str(second_path), sample_rate=16000),
                ]
            )

            class FakeTorchAudio:
                @staticmethod
                def load(source):
                    payload = Path(source).read_bytes()
                    if payload == b"first":
                        return torch.tensor([[1.0, 2.0, 3.0, 4.0]]), 16000
                    if payload == b"second":
                        return torch.tensor([[5.0, 6.0]]), 16000
                    raise AssertionError(source)

            with patch("anydataset.provider.abc.torchaudio", FakeTorchAudio()):
                with _fake_anytrain(FakeBatchedLongCatCodecLoader):
                    ViewMaterializer(target, split="train", batch_size=2).write(
                        dataset_factory=_StoreDatasetFactory(source, root),
                        provider_factory=_LongCatProviderFactory(),
                        devices="cpu",
                    )

            self.assertEqual(
                FakeBatchedLongCatCodecLoader.codec.calls,
                [((2, 1, 4), 16000)],
            )

            dataset = _store_dataset(target, root)
            first = dataset[0][(Role.DEFAULT, Modality.AUDIO)].views[AudioView.LONGCAT]
            second = dataset[1][(Role.DEFAULT, Modality.AUDIO)].views[AudioView.LONGCAT]
            self.assertTrue(
                torch.equal(
                    first,
                    torch.tensor(
                        [[10, 30, 40], [11, 31, 41], [12, 32, 42], [13, 33, 43]]
                    ),
                )
            )
            self.assertTrue(
                torch.equal(second, torch.tensor([[20, 50, 60], [21, 51, 61]]))
            )

    def test_provider_batches_waveform_without_shape_policy(self):
        with _fake_anytrain(FakeLongCatCodecLoader):
            provider = LongCatProvider()

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
            provider = LongCatProvider()

        provider({AudioView.WAVEFORM: (np.array([0.1, 0.2, 0.3, 0.4]), 16000)})

        self.assertEqual(FakeLongCatCodecLoader.codec.calls, [((1, 1, 4), 16000)])
        self.assertEqual(FakeLongCatCodecLoader.codec.dtypes, [torch.float32])

    def test_provider_requires_all_configured_codebooks(self):
        with _fake_anytrain(FakeLongCatCodecLoader):
            provider = LongCatProvider()
        provider.codec.codebook_sizes = (8192, 90, 90, 90)

        with self.assertRaisesRegex(ValueError, "all configured 4 codebooks"):
            provider({AudioView.WAVEFORM: (torch.zeros(1, 4), 16000)})

    def test_call_batch_trims_codes_from_waveform_mask(self):
        with _fake_anytrain(FakeBatchedLongCatCodecLoader):
            provider = LongCatProvider()

        outputs = provider.call_batch(
            _provider_batch(
                _audio_sample(
                    waveform=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                    sample_rate=16000,
                ),
                _audio_sample(
                    waveform=torch.tensor([[5.0, 6.0]]),
                    sample_rate=16000,
                ),
            )
        )

        self.assertEqual(
            FakeBatchedLongCatCodecLoader.codec.calls,
            [((2, 1, 4), 16000)],
        )
        self.assertTrue(
            torch.equal(
                outputs[0],
                torch.tensor(
                    [[10, 30, 40], [11, 31, 41], [12, 32, 42], [13, 33, 43]]
                ),
            )
        )
        self.assertTrue(
            torch.equal(outputs[1], torch.tensor([[20, 50, 60], [21, 51, 61]]))
        )

    def test_call_batch_requires_one_sample_rate_per_batch(self):
        with _fake_anytrain(FakeBatchedLongCatCodecLoader):
            provider = LongCatProvider()

        with self.assertRaisesRegex(ValueError, "one sample rate"):
            provider.call_batch(
                _provider_batch(
                    _audio_sample(
                        waveform=torch.tensor([[1.0, 2.0]]),
                        sample_rate=16000,
                    ),
                    _audio_sample(
                        waveform=torch.tensor([[3.0, 4.0]]),
                        sample_rate=24000,
                    ),
                )
            )

    def test_call_batch_adds_channel_dim_for_mono_waveforms(self):
        with _fake_anytrain(FakeBatchedLongCatCodecLoader):
            provider = LongCatProvider()

        provider.call_batch(
            _provider_batch(
                _audio_sample(
                    waveform=torch.tensor([1.0, 2.0, 3.0, 4.0]),
                    sample_rate=16000,
                ),
                _audio_sample(
                    waveform=torch.tensor([5.0, 6.0]),
                    sample_rate=16000,
                ),
            )
        )

        self.assertEqual(
            FakeBatchedLongCatCodecLoader.codec.calls,
            [((2, 1, 4), 16000)],
        )

    def test_call_batch_encodes_multiple_audio_roles(self):
        with _fake_anytrain(FakeBatchedLongCatCodecLoader):
            provider = LongCatProvider()

        outputs = provider.call_batch(
            _provider_batch(
                _role_audio_sample(
                    source_waveform=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                    target_waveform=torch.tensor([[5.0, 6.0]]),
                    sample_rate=16000,
                ),
                _role_audio_sample(
                    source_waveform=torch.tensor([[7.0, 8.0]]),
                    target_waveform=torch.tensor([[9.0, 10.0, 11.0, 12.0]]),
                    sample_rate=16000,
                ),
            )
        )

        self.assertEqual(
            FakeBatchedLongCatCodecLoader.codec.calls,
            [
                ((2, 1, 4), 16000),
                ((2, 1, 4), 16000),
            ],
        )
        self.assertIsInstance(outputs, dict)
        source = outputs[(Role.SOURCE, Modality.AUDIO)]
        target = outputs[(Role.TARGET, Modality.AUDIO)]
        self.assertTrue(
            torch.equal(source[0][:, 0], torch.tensor([10, 11, 12, 13]))
        )
        self.assertTrue(
            torch.equal(source[1][:, 0], torch.tensor([20, 21]))
        )
        self.assertTrue(
            torch.equal(target[0][:, 0], torch.tensor([10, 11]))
        )
        self.assertTrue(
            torch.equal(target[1][:, 0], torch.tensor([20, 21, 22, 23]))
        )

    def test_materializer_batches_longcat_and_trims_codes_from_waveform_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [
                    _audio_sample(
                        waveform=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                        sample_rate=16000,
                    ),
                    _audio_sample(
                        waveform=torch.tensor([[5.0, 6.0]]),
                        sample_rate=16000,
                    ),
                ]
            )

            with _fake_anytrain(FakeBatchedLongCatCodecLoader):
                ViewMaterializer(target, split="train", batch_size=2).write(
                    dataset_factory=_StoreDatasetFactory(source, root),
                    provider_factory=_LongCatProviderFactory(),
                    devices="cpu",
                )

            self.assertEqual(
                FakeBatchedLongCatCodecLoader.codec.calls,
                [((2, 1, 4), 16000)],
            )

            dataset = _store_dataset(target, root)
            first = dataset[0][(Role.DEFAULT, Modality.AUDIO)].views[AudioView.LONGCAT]
            second = dataset[1][(Role.DEFAULT, Modality.AUDIO)].views[AudioView.LONGCAT]
            self.assertTrue(
                torch.equal(
                    first,
                    torch.tensor(
                        [[10, 30, 40], [11, 31, 41], [12, 32, 42], [13, 33, 43]]
                    ),
                )
            )
            self.assertTrue(
                torch.equal(second, torch.tensor([[20, 50, 60], [21, 51, 61]]))
            )

    def test_materializer_batches_longcat_for_multiple_audio_roles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target = root / "target"
            DatasetWriter(source, dataset_id="toy-audio", split="train").write(
                [
                    _role_audio_sample(
                        source_waveform=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                        target_waveform=torch.tensor([[5.0, 6.0]]),
                        sample_rate=16000,
                    ),
                    _role_audio_sample(
                        source_waveform=torch.tensor([[7.0, 8.0]]),
                        target_waveform=torch.tensor([[9.0, 10.0, 11.0, 12.0]]),
                        sample_rate=16000,
                    ),
                ]
            )

            with _fake_anytrain(FakeBatchedLongCatCodecLoader):
                ViewMaterializer(target, split="train", batch_size=2).write(
                    dataset_factory=_StoreDatasetFactory(source, root),
                    provider_factory=_LongCatProviderFactory(),
                    devices="cpu",
                )

            self.assertEqual(
                FakeBatchedLongCatCodecLoader.codec.calls,
                [
                    ((2, 1, 4), 16000),
                    ((2, 1, 4), 16000),
                ],
            )

            dataset = _store_dataset(target, root)
            first_source = dataset[0][(Role.SOURCE, Modality.AUDIO)].views[
                AudioView.LONGCAT
            ]
            first_target = dataset[0][(Role.TARGET, Modality.AUDIO)].views[
                AudioView.LONGCAT
            ]
            second_source = dataset[1][(Role.SOURCE, Modality.AUDIO)].views[
                AudioView.LONGCAT
            ]
            second_target = dataset[1][(Role.TARGET, Modality.AUDIO)].views[
                AudioView.LONGCAT
            ]

            self.assertTrue(
                torch.equal(
                    first_source[:, 0],
                    torch.tensor([10, 11, 12, 13]),
                )
            )
            self.assertTrue(
                torch.equal(
                    first_target[:, 0],
                    torch.tensor([10, 11]),
                )
            )
            self.assertTrue(
                torch.equal(
                    second_source[:, 0],
                    torch.tensor([20, 21]),
                )
            )
            self.assertTrue(
                torch.equal(
                    second_target[:, 0],
                    torch.tensor([20, 21, 22, 23]),
                )
            )

    def test_loads_anytrain_codec_with_decoder_config(self):
        with _fake_anytrain(FakeLongCatCodecLoader):
            provider = LongCatProvider(
                cache_dir="/cache",
                decoder="24k_2codebooks",
                device="cpu",
                local_files_only=True,
                force_download=True,
            )

        self.assertIs(provider.codec, FakeLongCatCodecLoader.codec)
        self.assertEqual(
            FakeLongCatCodecLoader.calls,
            [
                {
                    "cache_dir": "/cache",
                    "decoder": "24k_2codebooks",
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


def _role_audio_sample(
    *,
    source_waveform,
    target_waveform,
    sample_rate: int,
):
    return {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (source_waveform, sample_rate)},
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (target_waveform, sample_rate)},
        ),
    }


def _store_dataset(path: Path, root: Path):
    return AnyDataset(
        Spec(source=Source.STORE, path=str(path), split="train"),
    )


def _provider_batch(*samples):
    refs = tuple(ref for ref in samples[0] if ref[1] is Modality.AUDIO)
    return collate_fn(
        {
            ref: AudioReq(views=frozenset({AudioView.WAVEFORM}))
            for ref in refs
        }
    )(samples)


class _StoreDatasetFactory:
    def __init__(self, path: Path, root: Path) -> None:
        self.path = path
        self.root = root

    def __call__(self):
        return _store_dataset(self.path, self.root)


class _LongCatProviderFactory:
    def __call__(self, device: str):
        return LongCatProvider(device=device if device != "cpu" else None)


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
        modules["anytrain.codec.longcat"].LongCat = self.codec_class
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
