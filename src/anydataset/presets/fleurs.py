from __future__ import annotations

from functools import partial
from typing import Any

from ..dataset.abc import AnyDataset
from ..rowmap import sample_from_row
from ..types import AudioView, Source, Spec, TextMeta, TextView
from ..types.item import Transforms


class Fleurs(AnyDataset):
    def __init__(
        self,
        split: str | None = None,
        *,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> None:
        lang = str(load_options.get("config_name", "en_us"))
        super().__init__(
            spec=create_spec(split=split, **load_options),
            parse_fn=partial(
                sample_from_row,
                audio={"audio": AudioView.WAVEFORM},
                text={"transcription": TextView.TEXT},
                text_values={TextMeta.LANG: lang},
            ),
            transforms=transforms,
        )


def create_spec(split: str | None = None, **load_options: Any) -> Spec:
    return Spec(
        source=Source.HF,
        path="google/fleurs",
        split="train" if split is None else split,
        load_options={"config_name": "en_us", **load_options},
    )
