from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..dataset.collate import Batch
from ..types.item import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
    TextMeta,
    TextView,
)


class QwenTTSProvider:
    output = AudioView.WAVEFORM
    batch_only = True

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        options: Any | None = None,
        default_language: str = "Auto",
        default_instruct: str | None = None,
        runtime_kwargs: Mapping[str, object] | None = None,
        **load_options: Any,
    ) -> None:
        try:
            from anytrain.tts.qwen import QwenCustomVoiceTTS
        except ImportError as exc:
            raise ImportError("QwenTTSProvider requires anytrain[qwen-tts].") from exc
        kwargs = {"runtime_kwargs": runtime_kwargs, **load_options}
        if model is None:
            self.tts = QwenCustomVoiceTTS.from_pretrained(**kwargs)
        else:
            self.tts = QwenCustomVoiceTTS.from_pretrained(model, **kwargs)
        self.options = options
        self.default_language = _non_empty_string("default_language", default_language)
        self.default_instruct = default_instruct

    def __call__(self, views: Mapping[Any, Any]) -> AudioItem:
        text = views[TextView.TEXT]
        speaker = views.get(TextView.SPEAKERS)
        language = views.get(TextMeta.LANG, self.default_language)
        if not isinstance(text, str):
            raise TypeError("QwenTTSProvider expects a string TextView.TEXT input.")
        if not isinstance(speaker, str):
            raise TypeError("QwenTTSProvider requires TextView.SPEAKERS input.")
        output = self.tts.synthesize_custom_voice(
            text,
            speakers=speaker,
            languages=str(language),
            instructs=self.default_instruct,
            options=self.options,
        )
        return _audio_item(output, speaker)

    def call_batch(
        self,
        batch: Batch,
    ) -> Sequence[AudioItem] | Mapping[tuple[Role, Modality], Sequence[AudioItem]]:
        refs = _text_refs(batch)
        outputs = {ref: self._synthesize_ref_batch(batch, ref) for ref in refs}
        if len(refs) == 1:
            return outputs[refs[0]]
        return outputs

    def _synthesize_ref_batch(
        self,
        batch: Batch,
        ref: tuple[Role, Modality],
    ) -> Sequence[AudioItem]:
        text_item = batch.sample[ref]
        texts = _text_batch(text_item.views[TextView.TEXT])
        speakers = _string_batch(text_item.views[TextView.SPEAKERS], "speaker ids")
        if len(speakers) != len(texts):
            raise ValueError("speaker id batch size must match text batch size.")
        languages = _language_batch(
            text_item.meta.get(TextMeta.LANG),
            count=len(texts),
            default=self.default_language,
        )
        outputs = self.tts.synthesize_custom_voice(
            texts,
            speakers=speakers,
            languages=languages,
            instructs=None
            if self.default_instruct is None
            else [self.default_instruct] * len(texts),
            options=self.options,
        )
        if not isinstance(outputs, Sequence):
            raise TypeError("Qwen TTS batch synthesize output must be a sequence.")
        if len(outputs) != len(texts):
            raise ValueError("Qwen TTS batch output size must match text batch size.")
        return [_audio_item(output, speaker) for output, speaker in zip(outputs, speakers)]


def _text_refs(batch: Batch) -> tuple[tuple[Role, Modality], ...]:
    refs = tuple(
        ref
        for ref in batch.sample
        if ref[1] is Modality.TEXT
        and TextView.TEXT in batch.sample[ref].views
        and TextView.SPEAKERS in batch.sample[ref].views
    )
    if not refs:
        raise ValueError(
            "QwenTTSProvider.call_batch expects text inputs with TextView.SPEAKERS."
        )
    return refs


def _text_batch(value: Any) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("QwenTTSProvider.call_batch expects a text sequence.")
    texts = list(value)
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("QwenTTSProvider.call_batch expects string text inputs.")
    return texts


def _string_batch(value: Any, name: str) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"QwenTTSProvider.call_batch expects a {name} sequence.")
    values = list(value)
    if any(not isinstance(item, str) or not item for item in values):
        raise TypeError(f"QwenTTSProvider.call_batch expects non-empty string {name}.")
    return values


def _language_batch(value: Any, *, count: int, default: str) -> list[str]:
    if value is None:
        return [default] * count
    if isinstance(value, str) or not isinstance(value, Sequence):
        return [str(value)] * count
    values = [str(item) for item in value]
    if len(values) != count:
        raise ValueError("language batch size must match text batch size.")
    return values


def _audio_item(output: Any, speaker: str) -> AudioItem:
    return AudioItem(
        views={AudioView.WAVEFORM: (output.waveform, output.sample_rate)},
        meta={
            AudioMeta.DURATION: (
                float(output.waveform.shape[-1]) / float(output.sample_rate)
            ),
            AudioMeta.SPEAKER_ID: speaker,
        },
    )


def _non_empty_string(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value
