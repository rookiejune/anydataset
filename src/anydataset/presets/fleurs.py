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
        options, lang = _options(load_options)
        super().__init__(
            spec=_spec(split, options),
            parse_fn=partial(
                sample_from_row,
                audio={"audio": AudioView.WAVEFORM},
                text={"transcription": TextView.TEXT},
                text_values={TextMeta.LANG: lang},
            ),
            transforms=transforms,
        )


def create_spec(split: str | None = None, **load_options: Any) -> Spec:
    options, _lang = _options(load_options)
    return _spec(split, options)


def _spec(split: str | None, load_options: dict[str, Any]) -> Spec:
    return Spec(
        source=Source.HF,
        path="google/fleurs",
        split="train" if split is None else split,
        load_options=load_options,
    )


def _options(load_options: dict[str, Any]) -> tuple[dict[str, Any], str]:
    options = dict(load_options)
    config_name = options.get("config_name", "en_us")
    if type(config_name) is not str:
        raise TypeError("FLEURS config_name must be a string.")
    if not config_name or config_name.strip() != config_name:
        raise ValueError("FLEURS config_name must be a non-empty string.")
    options["config_name"] = config_name
    return options, config_name
