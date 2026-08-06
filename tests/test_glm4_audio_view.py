from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

from anydataset.dataset.collate import FieldGroup, FieldRef, collate_fn
from anydataset.provider.glm4 import GLM4Provider
from anydataset.store import MaterializationStatus, ViewMaterializer
from anydataset.store.reader import read_store_dataset, read_store_views
from anydataset.types import AudioItem, AudioReq, AudioView, Modality, Role


_AUDIO_REF = (Role.DEFAULT, Modality.AUDIO)


class Glm4AudioViewTest(unittest.TestCase):
    def test_online_provider_rejects_mismatched_tokenizer_view(self):
        with _fake_anytrain(spec_view="stable"):
            with self.assertRaisesRegex(
                ValueError,
                "spec view 'stable' does not match provider output 'glm4'",
            ):
                GLM4Provider(device="cpu")

    def test_online_provider_uses_tokenizer_only_capability(self):
        with _fake_anytrain() as fake:
            provider = GLM4Provider(device="cpu")
            output = provider(
                {AudioView.WAVEFORM: (torch.tensor([[1.0, 2.0, 3.0]]), 16_000)}
            )

        self.assertEqual(fake.loader_calls, [("glm4", "cpu")])
        self.assertFalse(hasattr(fake.tokenizer.backend, "decode"))
        self.assertFalse(fake.tokenizer.backend.training)
        self.assertEqual(fake.tokenizer.calls, [((1, 1, 3), 16_000)])
        self.assertTrue(torch.equal(output, torch.tensor([[1], [2], [3]])))

    def test_online_provider_batches_frame_codes_without_a_decoder(self):
        schema = {_AUDIO_REF: AudioReq(views=frozenset({AudioView.WAVEFORM}))}
        batch = collate_fn(schema)(
            (
                _waveform_sample(torch.tensor([[1.0, 2.0]])),
                _waveform_sample(torch.tensor([[3.0, 4.0]])),
            )
        )

        with _fake_anytrain() as fake:
            outputs = GLM4Provider(device="cpu").call_batch(batch)

        self.assertEqual(fake.tokenizer.calls, [((2, 1, 2), 16_000)])
        self.assertEqual([tuple(output.shape) for output in outputs], [(2, 1), (2, 1)])
        self.assertTrue(torch.equal(outputs[0], torch.tensor([[1], [2]])))
        self.assertTrue(torch.equal(outputs[1], torch.tensor([[3], [4]])))

    def test_existing_audio_view_values_remain_stable(self):
        expected = {
            AudioView.WAVEFORM: "waveform",
            AudioView.FILE: "file",
            AudioView.BICODEC: "bicodec",
            AudioView.LONGCAT: "longcat",
            AudioView.DAC: "dac",
            AudioView.STABLE: "stable",
            AudioView.UNICODEC: "unicodec",
            AudioView.SPEAKERS: "speakers",
            AudioView.SPEAKER_LENGTHS: "speaker_lengths",
        }

        self.assertEqual(
            {view: view.value for view in expected},
            expected,
        )
        self.assertEqual(AudioView.GLM4.value, "glm4")

    def test_collate_fn_batches_glm4_codes_by_frame(self):
        schema = {_AUDIO_REF: AudioReq(views=frozenset({AudioView.GLM4}))}
        samples = (
            {
                _AUDIO_REF: AudioItem(
                    views={AudioView.GLM4: torch.tensor([[1], [2], [3]])}
                )
            },
            {
                _AUDIO_REF: AudioItem(
                    views={AudioView.GLM4: torch.tensor([[4], [5]])}
                )
            },
        )

        batch = collate_fn(schema)(samples)

        self.assertTrue(
            torch.equal(
                batch.sample[_AUDIO_REF].views[AudioView.GLM4],
                torch.tensor([[[1], [2], [3]], [[4], [5], [0]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.masks[
                    FieldRef(_AUDIO_REF, FieldGroup.VIEWS, AudioView.GLM4)
                ],
                torch.tensor([[True, True, True], [True, True, False]]),
            )
        )

    def test_incremental_materialization_snapshot_round_trips_glm4(self):
        samples = (
            _waveform_sample(torch.tensor([[1.0, 2.0]])),
            _waveform_sample(torch.tensor([[3.0]])),
        )
        dataset_factory = _DatasetFactory(samples)
        provider_factory = _ProviderFactory()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "target"
            snapshot = root / "snapshot"
            materializer = ViewMaterializer(target, commit_samples=1)

            with _fake_anytrain():
                status = materializer.write(
                    dataset_factory=dataset_factory,
                    provider_factory=provider_factory,
                    devices="cpu",
                    max_new_samples=1,
                    finalize=False,
                )
                self.assertIsInstance(status, MaterializationStatus)
                self.assertEqual(status.completed, 1)
                self.assertEqual(status.pending, 1)

                materializer.snapshot(
                    snapshot,
                    dataset_factory=dataset_factory,
                    provider_factory=provider_factory,
                )

            self.assertEqual(
                read_store_views(snapshot),
                ((Role.DEFAULT, Modality.AUDIO, AudioView.GLM4),),
            )
            with read_store_dataset(snapshot) as dataset:
                codes = dataset[0][_AUDIO_REF].views[AudioView.GLM4]
                self.assertTrue(torch.equal(codes, torch.tensor([[1], [2]])))


class _Dataset:
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]


class _DatasetFactory:
    def __init__(self, samples):
        self._samples = samples

    def __call__(self):
        return _Dataset(self._samples)


class _ProviderFactory:
    def __call__(self, device):
        return GLM4Provider(device=device)


class _FrameSpec:
    schema = "frame"
    frame_codebook_sizes = (16_384,)

    def __init__(self, view):
        self.view = view


class _TokenizerBackend(torch.nn.Module):
    pass


class _Tokenizer:
    def __init__(self, spec_view):
        self.spec = _FrameSpec(spec_view)
        self.backend = _TokenizerBackend()
        self.calls = []

    def tokenize(self, audio, sample_rate):
        self.calls.append((tuple(audio.shape), sample_rate))
        return audio[:, 0].to(torch.int64).unsqueeze(-1)


class _fake_anytrain:
    def __init__(self, spec_view="glm4"):
        self.loader_calls = []
        self.tokenizer = _Tokenizer(spec_view)
        self.previous = {}

    def __enter__(self):
        package = types.ModuleType("anytrain")
        codec = types.ModuleType("anytrain.codec")

        def load_audio_tokenizer(name, *, device=None):
            self.loader_calls.append((name, device))
            return self.tokenizer

        codec.load_audio_tokenizer = load_audio_tokenizer
        package.codec = codec
        modules = {"anytrain": package, "anytrain.codec": codec}
        self.previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, module in self.previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _waveform_sample(waveform):
    return {
        _AUDIO_REF: AudioItem(
            views={AudioView.WAVEFORM: (waveform, 16_000)},
        )
    }


if __name__ == "__main__":
    unittest.main()
