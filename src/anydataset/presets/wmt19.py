from __future__ import annotations

from functools import partial
from typing import Any

from ..types import Modality, Role, TextMeta, TextView
from ..dataset.abc import IterableAnyDataset
from ..types import Preset
from ..types.item import Transforms
from ..rowmap import sample_from_row, text_map
from .registry import preset_spec


class WMT19(IterableAnyDataset):
    def __init__(
        self,
        split: str | None = None,
        *,
        source_lang: str | None = None,
        target_lang: str | None = None,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> None:
        config_name = (
            str(load_options["config_name"]) if "config_name" in load_options else None
        )
        source_lang, target_lang = _langs(
            config_name,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        load_options["config_name"] = f"{source_lang}-{target_lang}"
        super().__init__(
            spec=preset_spec(Preset.WMT19, split=split, **load_options),
            parse_fn=partial(
                sample_from_row,
                items={
                    (Role.SOURCE, Modality.TEXT): text_map(
                        {("translation", source_lang): TextView.TEXT},
                        values={TextMeta.LANG: source_lang},
                    ),
                    (Role.TARGET, Modality.TEXT): text_map(
                        {("translation", target_lang): TextView.TEXT},
                        values={TextMeta.LANG: target_lang},
                    ),
                },
            ),
            transforms=transforms,
        )


def _langs(
    config_name: str | None,
    *,
    source_lang: str | None,
    target_lang: str | None,
) -> tuple[str, str]:
    if config_name is None:
        return source_lang or "cs", target_lang or "en"

    config_source, config_target = _split_config(config_name)
    source = source_lang or config_source
    target = target_lang or config_target
    if source_lang is not None and source_lang != config_source:
        raise ValueError("source_lang must match config_name.")
    if target_lang is not None and target_lang != config_target:
        raise ValueError("target_lang must match config_name.")
    return source, target


def _split_config(config_name: str) -> tuple[str, str]:
    parts = config_name.split("-")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("WMT19 config_name must use `<source>-<target>`.")
    return parts[0], parts[1]
