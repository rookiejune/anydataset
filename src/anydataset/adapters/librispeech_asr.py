from __future__ import annotations

from typing import Any, Mapping

from ..api.spec import DatasetSpec
from ..modalities import ModalityRole
from .huggingface import HuggingFaceAdapter
from .local_files import LocalFilesAdapter


class LibriSpeechASRAdapter(HuggingFaceAdapter):
    def __init__(self, text_field: str = "text", lang: str = "en"):
        self._fields = LocalFilesAdapter(
            audio_field="audio",
            text_field=text_field,
            lang_value=lang,
        )

    def audio(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        return self._fields.audio(row, role=role)

    def text(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        return self._fields.text(row, role=role)


def librispeech_asr_spec(
    config_name: str = "clean",
    split: str = "train.100",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="openslr/librispeech_asr",
        name="librispeech_asr",
        split=split,
        load_options={
            "config_name": config_name,
            "streaming": streaming,
        },
    )
