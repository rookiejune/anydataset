from __future__ import annotations

from typing import Any, Mapping

from ..api.spec import DatasetSpec
from ..modalities import ModalityRole
from .huggingface import HuggingFaceAdapter
from .local_files import LocalFilesAdapter


class FleursAdapter(HuggingFaceAdapter):
    def __init__(self, text_field: str = "transcription", lang: str = "en_us"):
        self._fields = LocalFilesAdapter(
            audio_field="audio",
            text_field=text_field,
            lang_value=lang,
        )

    def audio(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        return self._fields.audio(row, role=role)

    def text(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        return self._fields.text(row, role=role)


def fleurs_spec(
    language: str = "en_us",
    split: str = "train",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="google/fleurs",
        name="fleurs",
        split=split,
        load_options={
            "config_name": language,
            "streaming": streaming,
        },
    )
