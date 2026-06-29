from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..dataset import AudioMeta, AudioView, MultipleAnyDataset, TextMeta, TextView
from ..dataset.abc import IterableAnyDataset
from ..types import Spec
from ..types.item import Transforms
from ..utils import labels, sample_from_row


DEFAULT_COMMON_VOICE_LANGUAGE = "en"

type Languages = str | Sequence[str]

_SPLITS = frozenset(
    {
        "train",
        "dev",
        "test",
        "validated",
        "invalidated",
        "other",
    }
)

_AUDIO_LABEL_FIELDS = (
    "sentence_id",
    "sentence_domain",
    "up_votes",
    "down_votes",
    "age",
    "gender",
    "accents",
    "variant",
    "segment",
)


class CommonVoice(IterableAnyDataset):
    def __init__(
        self,
        split: str | None = None,
        *,
        root: str | Path | None = None,
        language: str | None = None,
        languages: Languages | None = None,
        langs: Languages | None = None,
        version: str | None = None,
        transforms: Transforms | None = None,
        **load_options: Any,
    ) -> None:
        resolved_language = _single_language(language, languages=languages, langs=langs)
        spec = common_voice_spec(
            split=split,
            root=root,
            language=resolved_language,
            version=version,
            **load_options,
        )
        super().__init__(
            spec=spec,
            parse_fn=CommonVoiceParser(Path(spec.path)),
            transforms=transforms,
        )


class CommonVoiceParser:
    def __init__(self, language_root: Path) -> None:
        self.language_root = language_root

    def __call__(self, row: Mapping[str, Any]):
        enriched = dict(row)
        enriched["audio_path"] = str(self.language_root / "clips" / str(row["path"]))
        return sample_from_row(
            enriched,
            audio={
                "audio_path": AudioView.FILE,
                "client_id": AudioMeta.SPEAKER_ID,
                **{field: labels(field) for field in _AUDIO_LABEL_FIELDS},
            },
            text={
                "sentence": TextView.TEXT,
                "locale": TextMeta.LANG,
            },
        )


def common_voice_spec(
    split: str | None = None,
    *,
    root: str | Path | None = None,
    language: str | None = None,
    languages: Languages | None = None,
    langs: Languages | None = None,
    version: str | None = None,
    **load_options: Any,
) -> Spec:
    resolved_language = _single_language(language, languages=languages, langs=langs)
    resolved_split = split or "train"
    if resolved_split not in _SPLITS:
        valid = ", ".join(sorted(_SPLITS))
        raise ValueError(f"Common Voice split must be one of: {valid}.")

    language_root = _language_root(
        root,
        version=version,
        language=resolved_language,
    )
    resolved_version = _corpus_version(language_root.parent)
    return Spec(
        source="tsv",
        path=str(language_root),
        split=resolved_split,
        version=resolved_version,
        load_options=load_options,
    )


def create_common_voice(
    split: str | None = None,
    *,
    root: str | Path | None = None,
    language: str | None = None,
    languages: Languages | None = None,
    langs: Languages | None = None,
    version: str | None = None,
    transforms: Transforms | None = None,
    **load_options: Any,
) -> IterableAnyDataset | MultipleAnyDataset:
    resolved_languages = _languages(language, languages=languages, langs=langs)
    datasets = tuple(
        CommonVoice(
            split=split,
            root=root,
            language=language,
            version=version,
            transforms=transforms,
            **load_options,
        )
        for language in resolved_languages
    )
    if len(datasets) == 1:
        return datasets[0]
    return MultipleAnyDataset(datasets)


def _language_root(
    root: str | Path | None,
    *,
    version: str | None,
    language: str,
) -> Path:
    base = _base_root(root)
    if base.name == language and base.parent.name.startswith("cv-corpus-"):
        return base
    if base.name.startswith("cv-corpus-"):
        return base / language

    corpus = _versioned_corpus_root(base, version)
    return corpus / language


def _versioned_corpus_root(root: Path, version: str | None) -> Path:
    if version is not None:
        return root / _corpus_name(version)
    candidates = [path for path in root.iterdir() if _is_corpus_root(path)]
    if not candidates:
        raise FileNotFoundError(f"No cv-corpus-* directories found under: {root}")
    return max(candidates, key=_corpus_sort_key)


def _corpus_name(version: str) -> str:
    if version.startswith("cv-corpus-"):
        return version
    return f"cv-corpus-{version}"


def _is_corpus_root(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("cv-corpus-")


def _corpus_version(corpus_root: Path) -> str:
    name = corpus_root.name
    if not name.startswith("cv-corpus-"):
        raise ValueError(f"Common Voice language root must be under cv-corpus-* directory: {corpus_root}")
    return name.removeprefix("cv-corpus-")


def _corpus_sort_key(path: Path) -> tuple[int | str, ...]:
    return tuple(
        int(part) if part.isdecimal() else part
        for part in re.split(r"([0-9]+)", path.name.removeprefix("cv-corpus-"))
        if part
    )


def _single_language(
    language: str | None,
    *,
    languages: Languages | None,
    langs: Languages | None,
) -> str:
    resolved = _languages(language, languages=languages, langs=langs)
    if len(resolved) != 1:
        raise ValueError("Common Voice spec requires exactly one language.")
    return resolved[0]


def _languages(
    language: str | None,
    *,
    languages: Languages | None,
    langs: Languages | None,
) -> tuple[str, ...]:
    provided = [
        value
        for value in (language, languages, langs)
        if value is not None
    ]
    if len(provided) > 1:
        raise ValueError("Use only one of language, languages or langs.")
    if not provided:
        return (DEFAULT_COMMON_VOICE_LANGUAGE,)

    value = provided[0]
    if isinstance(value, str):
        result = (value,)
    else:
        result = tuple(value)
    if not result:
        raise ValueError("Common Voice languages must not be empty.")
    return result


def _base_root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root)

    value = os.environ.get("COMMON_VOICE_DATASET_DIR")
    if value is None:
        raise ValueError(
            "Common Voice preset requires root or COMMON_VOICE_DATASET_DIR."
        )
    return Path(value)
