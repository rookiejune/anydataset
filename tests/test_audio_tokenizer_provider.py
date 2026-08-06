from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch

from anydataset.dataset.collate import FieldGroup, FieldRef, collate_fn
from anydataset.provider.codec import AudioTokenizerProvider
from anydataset.types import AudioItem, AudioReq, AudioView, Modality, Role


_AUDIO_REF = (Role.DEFAULT, Modality.AUDIO)


@dataclass(frozen=True)
class _Spec:
    view: str
    schema: str
    frame_codebook_sizes: tuple[int, ...] = ()
    semantic_codebook_sizes: tuple[int, ...] = ()
    acoustic_codebook_sizes: tuple[int, ...] = ()
    global_codebook_sizes: tuple[int, ...] = ()
    acoustic_layout: str | None = None
    acoustic_unit_length: int | None = None
    global_unit_length: int | None = None


@dataclass(frozen=True)
class _SemanticAcousticCodes:
    semantic: torch.Tensor
    acoustic: torch.Tensor


@dataclass(frozen=True)
class _SemanticGlobalCodes:
    semantic: torch.Tensor
    global_codes: torch.Tensor


class _Backend(torch.nn.Module):
    pass


class _Tokenizer:
    def __init__(
        self,
        spec: _Spec,
        tokenize: Callable[[torch.Tensor, int], object],
    ) -> None:
        self.spec = spec
        self.backend = _Backend()
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self._tokenize = tokenize

    def tokenize(self, audio: torch.Tensor, sample_rate: int) -> object:
        self.calls.append((tuple(audio.shape), sample_rate))
        return self._tokenize(audio, sample_rate)


class AudioTokenizerProviderTest(unittest.TestCase):
    def test_frame_schema_returns_one_signed_tensor_per_sample(self):
        tokenizer = _Tokenizer(
            _Spec(
                view="stable",
                schema="frame",
                frame_codebook_sizes=(16,),
            ),
            lambda audio, _sample_rate: audio[:, 0].to(torch.int64).unsqueeze(-1),
        )
        provider = AudioTokenizerProvider(tokenizer, AudioView.STABLE)

        output = provider({AudioView.WAVEFORM: (torch.tensor([1.0, 2.0, 3.0]), 16_000)})

        self.assertFalse(tokenizer.backend.training)
        self.assertEqual(tokenizer.calls, [((1, 1, 3), 16_000)])
        self.assertTrue(torch.equal(output, torch.tensor([[1], [2], [3]])))

    def test_frame_schema_rejects_unsigned_ids(self):
        tokenizer = _Tokenizer(
            _Spec(
                view="stable",
                schema="frame",
                frame_codebook_sizes=(16,),
            ),
            lambda audio, _sample_rate: torch.zeros(
                (audio.shape[0], 1, 1),
                dtype=torch.uint8,
            ),
        )

        with self.assertRaisesRegex(TypeError, "signed integer ids"):
            AudioTokenizerProvider(tokenizer, AudioView.STABLE)(
                {AudioView.WAVEFORM: (torch.zeros(2), 16_000)}
            )

    def test_semantic_acoustic_schema_returns_native_mapping(self):
        tokenizer = _Tokenizer(
            _Spec(
                view="longcat",
                schema="semantic_acoustic",
                semantic_codebook_sizes=(8,),
                acoustic_codebook_sizes=(16, 16),
                acoustic_layout="frame_aligned",
            ),
            lambda audio, _sample_rate: _semantic_acoustic(audio),
        )

        with _fake_anytrain_codes():
            output = AudioTokenizerProvider(tokenizer, AudioView.LONGCAT)(
                {AudioView.WAVEFORM: (torch.tensor([1.0, 2.0, 3.0]), 16_000)}
            )

        self.assertEqual(set(output), {"semantic", "acoustic"})
        self.assertTrue(torch.equal(output["semantic"], torch.tensor([[1], [2], [3]])))
        self.assertTrue(
            torch.equal(
                output["acoustic"],
                torch.tensor([[5, 9], [6, 10], [7, 11]]),
            )
        )
        self.assertEqual(output["semantic"].device.type, "cpu")
        self.assertTrue(output["semantic"].is_contiguous())

    def test_batch_groups_by_rate_and_unpadded_length_then_restores_order(self):
        tokenizer = _Tokenizer(
            _Spec(
                view="longcat",
                schema="semantic_acoustic",
                semantic_codebook_sizes=(32,),
                acoustic_codebook_sizes=(64, 64),
                acoustic_layout="frame_aligned",
            ),
            lambda audio, _sample_rate: _semantic_acoustic(audio),
        )
        batch = _batch(
            _sample(torch.tensor([1.0, 2.0, 3.0, 4.0]), 16_000),
            _sample(torch.tensor([5.0, 6.0]), 24_000),
            _sample(torch.tensor([7.0, 8.0, 9.0, 10.0]), 16_000),
        )

        with _fake_anytrain_codes():
            outputs = AudioTokenizerProvider(
                tokenizer,
                AudioView.LONGCAT,
            ).call_batch(batch)

        self.assertEqual(
            tokenizer.calls,
            [((2, 1, 4), 16_000), ((1, 1, 2), 24_000)],
        )
        self.assertTrue(
            torch.equal(outputs[0]["semantic"][:, 0], torch.tensor([1, 2, 3, 4]))
        )
        self.assertTrue(torch.equal(outputs[1]["semantic"][:, 0], torch.tensor([5, 6])))
        self.assertTrue(
            torch.equal(
                outputs[2]["semantic"][:, 0],
                torch.tensor([7, 8, 9, 10]),
            )
        )

    def test_fixed_acoustic_schema_preserves_independent_unit_axis(self):
        def tokenize(audio: torch.Tensor, _sample_rate: int) -> object:
            semantic = audio[:, 0].to(torch.int64).unsqueeze(-1)
            acoustic = torch.tensor(
                [[[1], [2]], [[3], [4]]],
                dtype=torch.int64,
            )[: audio.shape[0]]
            return _SemanticAcousticCodes(semantic, acoustic)

        tokenizer = _Tokenizer(
            _Spec(
                view="stable",
                schema="semantic_acoustic",
                semantic_codebook_sizes=(16,),
                acoustic_codebook_sizes=(8,),
                acoustic_layout="fixed_length",
                acoustic_unit_length=2,
            ),
            tokenize,
        )

        with _fake_anytrain_codes():
            output = AudioTokenizerProvider(tokenizer, AudioView.STABLE)(
                {AudioView.WAVEFORM: (torch.tensor([1.0, 2.0, 3.0]), 16_000)}
            )

        self.assertEqual(tuple(output["semantic"].shape), (3, 1))
        self.assertEqual(tuple(output["acoustic"].shape), (2, 1))

    def test_semantic_global_schema_renames_native_global_codes_field(self):
        def tokenize(audio: torch.Tensor, _sample_rate: int) -> object:
            semantic = audio[:, 0].to(torch.int64).unsqueeze(-1)
            base = semantic[:, 0, 0]
            global_codes = torch.stack((base, base + 1), dim=1).unsqueeze(-1)
            return _SemanticGlobalCodes(semantic, global_codes)

        tokenizer = _Tokenizer(
            _Spec(
                view="bicodec",
                schema="semantic_global",
                semantic_codebook_sizes=(16,),
                global_codebook_sizes=(16,),
                global_unit_length=2,
            ),
            tokenize,
        )

        with _fake_anytrain_codes():
            output = AudioTokenizerProvider(tokenizer, AudioView.BICODEC)(
                {AudioView.WAVEFORM: (torch.tensor([2.0, 3.0]), 16_000)}
            )

        self.assertEqual(set(output), {"semantic", "global"})
        self.assertTrue(torch.equal(output["global"], torch.tensor([[2], [3]])))

    def test_structured_schema_rejects_wrong_native_result_type(self):
        tokenizer = _Tokenizer(
            _Spec(
                view="longcat",
                schema="semantic_acoustic",
                semantic_codebook_sizes=(8,),
                acoustic_codebook_sizes=(8,),
                acoustic_layout="frame_aligned",
            ),
            lambda _audio, _sample_rate: {
                "semantic": torch.zeros((1, 1, 1), dtype=torch.int64),
                "acoustic": torch.zeros((1, 1, 1), dtype=torch.int64),
            },
        )

        with _fake_anytrain_codes():
            with self.assertRaisesRegex(TypeError, "SemanticAcousticCodes"):
                AudioTokenizerProvider(tokenizer, AudioView.LONGCAT)(
                    {AudioView.WAVEFORM: (torch.zeros(2), 16_000)}
                )

    def test_semantic_acoustic_validates_alignment_and_codebook_ranges(self):
        misaligned = _Tokenizer(
            _Spec(
                view="longcat",
                schema="semantic_acoustic",
                semantic_codebook_sizes=(4,),
                acoustic_codebook_sizes=(4,),
                acoustic_layout="frame_aligned",
            ),
            lambda _audio, _sample_rate: _SemanticAcousticCodes(
                semantic=torch.tensor([[[0], [1]]]),
                acoustic=torch.tensor([[[0]]]),
            ),
        )
        out_of_range = _Tokenizer(
            misaligned.spec,
            lambda _audio, _sample_rate: _SemanticAcousticCodes(
                semantic=torch.tensor([[[0], [4]]]),
                acoustic=torch.tensor([[[0], [1]]]),
            ),
        )

        with _fake_anytrain_codes():
            with self.assertRaisesRegex(ValueError, "align on batch and time"):
                AudioTokenizerProvider(misaligned, AudioView.LONGCAT)(
                    {AudioView.WAVEFORM: (torch.zeros(2), 16_000)}
                )
            with self.assertRaisesRegex(
                ValueError,
                r"codebook 0 observed \[0, 4\], expected \[0, 4\)",
            ):
                AudioTokenizerProvider(out_of_range, AudioView.LONGCAT)(
                    {AudioView.WAVEFORM: (torch.zeros(2), 16_000)}
                )


class StructuredAudioCollateTest(unittest.TestCase):
    def test_frame_aligned_mapping_uses_one_shared_temporal_mask(self):
        samples = (
            _token_sample(
                AudioView.LONGCAT,
                {
                    "semantic": torch.tensor([[1], [2]]),
                    "acoustic": torch.tensor([[3, 4], [5, 6]]),
                },
            ),
            _token_sample(
                AudioView.LONGCAT,
                {
                    "semantic": torch.tensor([[7]]),
                    "acoustic": torch.tensor([[8, 9]]),
                },
            ),
        )

        batch = collate_fn(
            {_AUDIO_REF: AudioReq(views=frozenset({AudioView.LONGCAT}))}
        )(samples)
        values = batch.sample[_AUDIO_REF].views[AudioView.LONGCAT]

        self.assertEqual(tuple(values["semantic"].shape), (2, 2, 1))
        self.assertEqual(tuple(values["acoustic"].shape), (2, 2, 2))
        self.assertTrue(
            torch.equal(
                batch.masks[FieldRef(_AUDIO_REF, FieldGroup.VIEWS, AudioView.LONGCAT)],
                torch.tensor([[True, True], [True, False]]),
            )
        )

    def test_fixed_acoustic_mapping_is_inferred_for_any_audio_view(self):
        samples = (
            _token_sample(
                AudioView.STABLE,
                {
                    "semantic": torch.tensor([[1], [2]]),
                    "acoustic": torch.tensor([[3], [4], [5]]),
                },
            ),
            _token_sample(
                AudioView.STABLE,
                {
                    "semantic": torch.tensor([[6]]),
                    "acoustic": torch.tensor([[7], [8], [9]]),
                },
            ),
        )

        batch = collate_fn({_AUDIO_REF: AudioReq(views=frozenset({AudioView.STABLE}))})(
            samples
        )
        values = batch.sample[_AUDIO_REF].views[AudioView.STABLE]

        self.assertEqual(tuple(values["semantic"].shape), (2, 2, 1))
        self.assertEqual(tuple(values["acoustic"].shape), (2, 3, 1))


def _semantic_acoustic(audio: torch.Tensor) -> _SemanticAcousticCodes:
    semantic = audio[:, 0].to(torch.int64).unsqueeze(-1)
    acoustic = torch.cat((semantic + 4, semantic + 8), dim=-1)
    return _SemanticAcousticCodes(semantic, acoustic)


def _sample(waveform: torch.Tensor, sample_rate: int) -> dict[Any, Any]:
    return {
        _AUDIO_REF: AudioItem(
            views={AudioView.WAVEFORM: (waveform, sample_rate)},
        )
    }


def _batch(*samples: dict[Any, Any]):
    return collate_fn({_AUDIO_REF: AudioReq(views=frozenset({AudioView.WAVEFORM}))})(
        samples
    )


def _token_sample(view: AudioView, value: object) -> dict[Any, Any]:
    return {_AUDIO_REF: AudioItem(views={view: value})}


@contextmanager
def _fake_anytrain_codes():
    package = types.ModuleType("anytrain")
    package.__path__ = []
    codec = types.ModuleType("anytrain.codec")
    codec.SemanticAcousticCodes = _SemanticAcousticCodes
    codec.SemanticGlobalCodes = _SemanticGlobalCodes
    package.codec = codec
    modules = {"anytrain": package, "anytrain.codec": codec}
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
