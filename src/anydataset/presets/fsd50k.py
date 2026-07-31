from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..dataset.abc import AnyDataset
from ..types import AudioItem, AudioView, Modality, Role, Source, Spec
from ..types.item import Transforms


class FSD50K(AnyDataset):
    def __init__(
        self,
        split: str | None = None,
        *,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> None:
        if split is not None and split not in _VALID_SPLITS:
            raise ValueError("FSD50K split must be 'dev' or 'eval'.")
        extra = set(load_options) - {"revision"}
        if extra:
            name = min(extra)
            raise TypeError(f"Unexpected FSD50K load option: {name}.")
        revision = load_options.get("revision", "main")
        if not isinstance(revision, str) or not revision:
            raise ValueError("FSD50K revision must be a non-empty string.")
        super().__init__(
            spec=fsd50k_spec(split=split, revision=revision),
            parse_fn=parse_fsd50k_row,
            transforms=transforms,
        )


def fsd50k_spec(split: str | None = None, *, revision: str = "main") -> Spec:
    resolved_split = "dev" if split is None else split
    if resolved_split not in _VALID_SPLITS:
        raise ValueError("FSD50K split must be 'dev' or 'eval'.")
    if not isinstance(revision, str) or not revision:
        raise ValueError("FSD50K revision must be a non-empty string.")
    return Spec(
        source=Source.HF_FILES,
        path="Fhrozen/FSD50k",
        split=resolved_split,
        version=revision,
        load_options={
            "repo_type": "dataset",
            "path_template": "clips/{split}",
            "suffixes": (".wav",),
        },
    )


def parse_fsd50k_row(row: Any):
    if not isinstance(row, Mapping):
        raise TypeError("FSD50K rows must be mappings.")
    local_path = row.get("local_path")
    if not isinstance(local_path, str):
        raise ValueError("FSD50K rows require a local_path string.")
    waveform, sample_rate = _load_audio(local_path)
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (waveform, sample_rate)}
        )
    }


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio
    except (ImportError, OSError) as exc:
        raise ImportError(
            "FSD50K audio loading requires a working torchaudio installation "
            "('pip install anydataset[audio]')."
        ) from exc

    waveform, sample_rate = torchaudio.load(str(path))
    return waveform.to(dtype=torch.float32), int(sample_rate)


_VALID_SPLITS = frozenset({"dev", "eval"})
