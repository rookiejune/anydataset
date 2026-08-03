from __future__ import annotations

import re
from functools import partial
from typing import Any

from ..dataset.abc import AnyDataset
from ..rowmap import sample_from_row, text_map
from ..types import Lang, Modality, Role, Source, Spec, TextMeta, TextView
from ..types.item import Transforms


class WMT19(AnyDataset):
    def __init__(
        self,
        split: str | None = None,
        *,
        source_lang: Lang | str | None = None,
        target_lang: Lang | str | None = None,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> None:
        options, source, target = _options(
            load_options,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        super().__init__(
            spec=_spec(split, options),
            parse_fn=partial(
                sample_from_row,
                items={
                    (Role.SOURCE, Modality.TEXT): text_map(
                        {("translation", source): TextView.TEXT},
                        values={TextMeta.LANG: source},
                    ),
                    (Role.TARGET, Modality.TEXT): text_map(
                        {("translation", target): TextView.TEXT},
                        values={TextMeta.LANG: target},
                    ),
                },
            ),
            transforms=transforms,
        )


def create_spec(
    split: str | None = None,
    *,
    source_lang: Lang | str | None = None,
    target_lang: Lang | str | None = None,
    **load_options: Any,
) -> Spec:
    options, _source, _target = _options(
        load_options,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    return _spec(split, options)


def _spec(split: str | None, load_options: dict[str, Any]) -> Spec:
    return Spec(
        source=Source.HF,
        path="wmt/wmt19",
        split="train" if split is None else split,
        load_options=load_options,
    )


def _options(
    load_options: dict[str, Any],
    *,
    source_lang: Lang | str | None,
    target_lang: Lang | str | None,
) -> tuple[dict[str, Any], str, str]:
    options = dict(load_options)
    config_name = options.pop("config_name", None)
    source, target = _langs(
        config_name,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    options["config_name"] = f"{source}-{target}"
    return options, source, target


def _langs(
    config_name: object | None,
    *,
    source_lang: Lang | str | None,
    target_lang: Lang | str | None,
) -> tuple[str, str]:
    source = _optional_lang("source_lang", source_lang)
    target = _optional_lang("target_lang", target_lang)
    if config_name is None:
        return source or "cs", target or "en"

    config_source, config_target = _split_config(config_name)
    if source is not None and source != config_source:
        raise ValueError("source_lang must match config_name.")
    if target is not None and target != config_target:
        raise ValueError("target_lang must match config_name.")
    return source or config_source, target or config_target


def _optional_lang(name: str, value: Lang | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Lang):
        return _language_code(name, value.value)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a Lang or string.")
    return _language_code(name, value)


def _language_code(name: str, value: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty language code.")
    normalized = value.lower()
    if re.fullmatch(r"[a-z][a-z0-9_]*", normalized) is None:
        raise ValueError(f"{name} must contain one unambiguous ASCII language code.")
    return normalized


def _split_config(config_name: object) -> tuple[str, str]:
    if not isinstance(config_name, str):
        raise TypeError("WMT19 config_name must be a string.")
    if not config_name or config_name.strip() != config_name:
        raise ValueError("WMT19 config_name must be a non-empty string.")
    parts = config_name.split("-")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("WMT19 config_name must use `<source>-<target>`.")
    return (
        _language_code("WMT19 config_name source", parts[0]),
        _language_code("WMT19 config_name target", parts[1]),
    )
