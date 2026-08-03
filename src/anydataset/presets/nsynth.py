from __future__ import annotations

from functools import partial
from typing import Any

from ..dataset.abc import AnyDataset
from ..rowmap import labels, sample_from_row
from ..types import AudioView, Source, Spec
from ..types.item import Transforms


class NSynth(AnyDataset):
    def __init__(
        self,
        split: str | None = None,
        *,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> None:
        super().__init__(
            spec=create_spec(split=split, **load_options),
            parse_fn=partial(
                sample_from_row,
                audio={
                    "audio": AudioView.WAVEFORM,
                    "instrument_family_str": labels("instrument_family"),
                    "instrument_source_str": labels("instrument_source"),
                    "pitch": labels("pitch"),
                    "velocity": labels("velocity"),
                },
            ),
            transforms=transforms,
        )


def create_spec(split: str | None = None, **load_options: Any) -> Spec:
    return Spec(
        source=Source.HF,
        path="confit/nsynth-parquet",
        split="train" if split is None else split,
        load_options={"config_name": "instrument", **load_options},
    )
