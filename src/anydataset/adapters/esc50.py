from __future__ import annotations

from typing import Any, Mapping

from ..api.spec import DatasetSpec
from ..modalities import ModalityRole
from .huggingface import HuggingFaceAdapter
from .local_files import LocalFilesAdapter


class ESC50Adapter(HuggingFaceAdapter):
    def __init__(self):
        self._fields = LocalFilesAdapter(
            audio_field="audio",
            label_field="category",
            labels_fields={
                "target": "target",
                "esc10": "esc10",
            },
        )

    def audio(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        return self._fields.audio(row, role=role)


def esc50_spec(
    split: str = "train",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="ashraq/esc50",
        name="esc50",
        split=split,
        load_options={
            "streaming": streaming,
        },
    )
