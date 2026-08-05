from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from warnings import warn

import torch

from ..dataset.collate import Batch, FieldGroup, FieldRef
from ..types.item import Modality, Role
from ..types.item import AudioItem, AudioMeta, AudioView, FileBytes, TextItem, TextView


class MossTTSProvider:
    output = AudioView.WAVEFORM

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        options: Any | None = None,
        reference_role: Role | None = None,
        max_reference_files: int | None = None,
        runtime_kwargs: Mapping[str, object] | None = None,
        **load_options: Any,
    ) -> None:
        try:
            from anytrain.tts import EncodedAudioReference, WaveformReference
            from anytrain.tts.moss import MossTTS
        except ImportError as exc:
            raise ImportError("MossTTSProvider requires `anytrain[moss-tts]`.") from exc
        kwargs = {"runtime_kwargs": runtime_kwargs, **load_options}
        if model is None:
            self.tts = MossTTS.from_pretrained(**kwargs)
        else:
            self.tts = MossTTS.from_pretrained(model, **kwargs)
        self.options = options
        self.reference_role = reference_role
        self._encoded_audio_reference = EncodedAudioReference
        self._waveform_reference = WaveformReference
        if max_reference_files is not None:
            _positive_int("max_reference_files", max_reference_files)
            warn(
                "max_reference_files is deprecated and ignored; anytrain now "
                "owns temporary reference-file materialization.",
                DeprecationWarning,
                stacklevel=2,
            )

    def __call__(self, views: Mapping[Any, Any]) -> Any:
        text = views[TextView.TEXT]
        if not isinstance(text, str):
            raise TypeError("MossTTSProvider expects a string TextView.TEXT input.")
        output = self.tts.synthesize(
            text,
            self.options,
            reference_audio=self._reference(views),
        )
        return _audio_output(output)

    def call_batch(
        self,
        batch: Batch,
    ) -> Sequence[Any] | Mapping[tuple[Role, Modality], Sequence[Any]]:
        refs = _text_refs(batch)
        outputs = {ref: self._synthesize_ref_batch(batch, ref) for ref in refs}
        if len(refs) == 1:
            return outputs[refs[0]]
        return outputs

    def _synthesize_ref_batch(
        self,
        batch: Batch,
        ref: tuple[Role, Modality],
    ) -> Sequence[Any]:
        item = batch.sample[ref]
        if not isinstance(item, TextItem):
            raise TypeError(f"{ref!r} requires a collated TextItem.")
        texts = _text_batch(item.views[TextView.TEXT])
        outputs = self.tts.synthesize(
            texts,
            self.options,
            reference_audios=self._references(batch, len(texts)),
        )
        if not isinstance(outputs, Sequence):
            raise TypeError("MossTTS batch synthesize output must be a sequence.")
        return [_audio_output(output) for output in outputs]

    def _reference(self, views: Mapping[Any, Any]) -> Any | None:
        if self.reference_role is None:
            return None
        if AudioView.FILE in views:
            return self._file_reference(views[AudioView.FILE])
        if AudioView.WAVEFORM in views:
            waveform, sample_rate = views[AudioView.WAVEFORM]
            return self._waveform_reference(
                torch.as_tensor(waveform),
                int(sample_rate),
            )
        raise ValueError(
            "MossTTSProvider reference input requires AudioView.FILE or "
            "AudioView.WAVEFORM."
        )

    def _references(self, batch: Batch, count: int) -> Sequence[Any] | None:
        if self.reference_role is None:
            return None
        ref = (self.reference_role, Modality.AUDIO)
        item = batch.sample[ref]
        if not isinstance(item, AudioItem):
            raise TypeError(f"{ref!r} requires a collated AudioItem.")
        views = item.views
        if AudioView.FILE in views:
            values = _file_batch(views[AudioView.FILE])
            if len(values) != count:
                raise ValueError(
                    "reference file batch size must match text batch size."
                )
            return [self._file_reference(value) for value in values]
        if AudioView.WAVEFORM in views:
            waveform, sample_rates = views[AudioView.WAVEFORM]
            lengths = batch.lengths(FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM))
            if waveform.shape[0] != count:
                raise ValueError(
                    "reference audio batch size must match text batch size."
                )
            return [
                self._waveform_reference(
                    waveform[index, :, : int(length.item())],
                    int(sample_rates[index].item()),
                )
                for index, length in enumerate(lengths)
            ]
        raise ValueError(
            "MossTTSProvider reference batch requires AudioView.FILE or "
            "AudioView.WAVEFORM."
        )

    def _file_reference(self, value: Any) -> Any:
        if isinstance(value, (str, Path)):
            return str(Path(value).expanduser())
        if isinstance(value, FileBytes):
            return self._encoded_audio_reference(value.data, value.suffix)
        if isinstance(value, bytes):
            return self._encoded_audio_reference(value, ".wav")
        raise TypeError(
            "reference file view must be a path, FileBytes, or bytes."
        )


def _text_refs(batch: Batch) -> tuple[tuple[Role, Modality], ...]:
    refs = tuple(
        ref
        for ref in batch.sample
        if ref[1] is Modality.TEXT and TextView.TEXT in batch.sample[ref].views
    )
    if not refs:
        raise ValueError("MossTTSProvider.call_batch expects at least one text input.")
    return refs


def _text_batch(value: Any) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("MossTTSProvider.call_batch expects a text sequence.")
    texts = list(value)
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("MossTTSProvider.call_batch expects string text inputs.")
    return texts


def _file_batch(value: Any) -> list[Any]:
    if isinstance(value, (str, Path, bytes, FileBytes)):
        return [value]
    if not isinstance(value, Sequence):
        raise TypeError("batched reference file view must be a sequence.")
    return list(value)


def _audio_output(output: Any) -> AudioItem:
    return AudioItem(
        views={AudioView.WAVEFORM: (output.waveform, output.sample_rate)},
        meta={
            AudioMeta.DURATION: (
                float(output.waveform.shape[-1]) / float(output.sample_rate)
            )
        },
    )


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value
