from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Union

from ..types import AudioMeta, AudioView, TextMeta, TextView
from ..dataset.abc import IterableAnyDataset
from ..types import Spec
from ..types.item import Transforms
from ..rowmap import labels, sample_from_row


_LANGUAGE_ROOT_FIELD = "__anydataset_root__"

Languages = Union[str, Sequence[str]]

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
        spec = common_voice_spec(
            split=split,
            root=root,
            language=language,
            languages=languages,
            langs=langs,
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
        language_root = Path(str(row.get(_LANGUAGE_ROOT_FIELD, self.language_root)))
        enriched = dict(row)
        enriched.pop(_LANGUAGE_ROOT_FIELD, None)
        enriched["audio_path"] = str(language_root / "clips" / str(row["path"]))
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
    requested_languages = _requested_languages(
        language,
        languages=languages,
        langs=langs,
    )
    resolved_split = "train" if split is None else split
    if resolved_split not in _SPLITS:
        valid = ", ".join(sorted(_SPLITS))
        raise ValueError(f"Common Voice split must be one of: {valid}.")

    corpus_root, resolved_languages = _corpus_root_and_languages(
        root,
        version=version,
        languages=requested_languages,
    )
    resolved_version = _corpus_version(corpus_root)
    return Spec(
        source="tsv",
        path=str(corpus_root),
        split=resolved_split,
        version=resolved_version,
        load_options=_load_options(
            load_options,
            languages=resolved_languages,
        ),
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
) -> IterableAnyDataset:
    return CommonVoice(
        split=split,
        root=root,
        language=language,
        languages=languages,
        langs=langs,
        version=version,
        transforms=transforms,
        **load_options,
    )


def _corpus_root_and_languages(
    root: str | Path | None,
    *,
    version: str | None,
    languages: tuple[str, ...] | None,
) -> tuple[Path, tuple[str, ...]]:
    base = _base_root(root)
    if base.name.startswith("cv-corpus-"):
        return base, _resolve_languages(base, languages)

    if base.parent.name.startswith("cv-corpus-"):
        if languages is not None and languages != (base.name,):
            raise ValueError(
                "Common Voice language root only supports its own language. "
                "Use the corpus root to select multiple languages."
            )
        return base.parent, (base.name,)

    corpus = _versioned_corpus_root(base, version)
    if languages is None and version is None:
        _ensure_latest_has_all_languages(base, corpus)
    return corpus, _resolve_languages(corpus, languages)


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


def _resolve_languages(
    corpus_root: Path,
    languages: tuple[str, ...] | None,
) -> tuple[str, ...]:
    available = _language_names(corpus_root)
    if languages is not None:
        missing = sorted(set(languages).difference(available))
        if missing:
            missing_text = ", ".join(missing)
            raise FileNotFoundError(
                f"Common Voice corpus {corpus_root} does not contain languages: "
                f"{missing_text}"
            )
        return tuple(sorted(languages))
    names = tuple(sorted(available))
    if not names:
        raise FileNotFoundError(f"No Common Voice language directories found under: {corpus_root}")
    return names


def _language_names(corpus_root: Path) -> set[str]:
    return {
        path.name
        for path in corpus_root.iterdir()
        if _is_language_root(path)
    }


def _is_language_root(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(".")


def _ensure_latest_has_all_languages(root: Path, latest: Path) -> None:
    latest_languages = _language_names(latest)
    missing = sorted(
        {
            language
            for corpus in root.iterdir()
            if _is_corpus_root(corpus) and corpus != latest
            for language in _language_names(corpus)
            if language not in latest_languages
        }
    )
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Latest Common Voice corpus {latest.name} is missing languages "
            f"found in older corpora: {missing_text}. "
            "Symlink or move those language directories into the latest corpus, "
            "or pass languages explicitly."
        )


def _requested_languages(
    language: str | None,
    *,
    languages: Languages | None,
    langs: Languages | None,
) -> tuple[str, ...] | None:
    provided = [
        value
        for value in (language, languages, langs)
        if value is not None
    ]
    if len(provided) > 1:
        raise ValueError("Use only one of language, languages or langs.")
    if not provided:
        return None

    value = provided[0]
    if isinstance(value, str):
        result = (value,)
    else:
        if not isinstance(value, Sequence):
            raise TypeError("Common Voice languages must be a string sequence.")
        result = tuple(value)
    if not result:
        raise ValueError("Common Voice languages must not be empty.")
    seen: set[str] = set()
    for item in result:
        if not isinstance(item, str):
            raise TypeError("Common Voice languages must contain strings.")
        if not item:
            raise ValueError("Common Voice languages must not contain empty strings.")
        if item in seen:
            raise ValueError(f"Duplicate Common Voice language: {item!r}.")
        seen.add(item)
    return result


def _load_options(
    load_options: Mapping[str, Any],
    *,
    languages: tuple[str, ...],
) -> dict[str, Any]:
    reserved = {"subdirs", "root_field"}
    overlap = reserved.intersection(load_options)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Common Voice load options reserve: {names}.")
    return {
        **load_options,
        "subdirs": languages,
        "root_field": _LANGUAGE_ROOT_FIELD,
    }


def _base_root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root)

    value = os.environ.get("COMMON_VOICE_DATASET_DIR")
    if value is None:
        raise ValueError(
            "Common Voice preset requires root or COMMON_VOICE_DATASET_DIR."
        )
    return Path(value)
