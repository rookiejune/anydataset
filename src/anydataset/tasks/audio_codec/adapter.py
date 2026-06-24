from __future__ import annotations

from typing import Any, Mapping

from ...adapters import MissingModalityError, ModalityAdapter
from ...modalities import ModalityKey
from ..base import TaskAdapter


class AudioCodecAdapter(TaskAdapter):
    def adapt(
        self,
        row: Mapping[str, Any],
        adapter: ModalityAdapter,
    ) -> Mapping[str, Any]:
        data: dict[str, Any] = {
            ModalityKey.AUDIO: adapter.audio(row),
        }
        try:
            data[ModalityKey.TEXT] = adapter.text(row)
        except MissingModalityError as exc:
            if exc.modality != ModalityKey.TEXT:
                raise
        return data
