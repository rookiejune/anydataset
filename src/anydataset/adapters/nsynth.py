from __future__ import annotations

from typing import Any, Mapping

from ..api.spec import DatasetSpec
from ..modalities import ModalityRole
from .huggingface import HuggingFaceAdapter
from .local_files import LocalFilesAdapter


class NSynthAdapter(HuggingFaceAdapter):
    def __init__(self):
        self._fields = LocalFilesAdapter(
            audio_field="audio",
            labels_fields={
                "instrument_family": "instrument_family_str",
                "instrument_source": "instrument_source_str",
                "pitch": "pitch",
                "velocity": "velocity",
            },
        )

    def audio(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        return self._fields.audio(row, role=role)


def nsynth_spec(
    config_name: str = "instrument",
    split: str = "train",
    streaming: bool = True,
) -> DatasetSpec:
    return DatasetSpec(
        source="huggingface",
        path="confit/nsynth-parquet",
        name="nsynth",
        split=split,
        load_options={
            "config_name": config_name,
            "streaming": streaming,
        },
    )
