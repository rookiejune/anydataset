from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import auto
from typing import Literal, Union, cast

from ..._compat import StrEnum
from ...dataset import MapStyleABC
from ...types import (
    AudioItem,
    Lang,
    Modality,
    Reference,
    Role,
    Sample,
    Schema,
    TextItem,
    TextMeta,
    TextView,
)

DatasetFactory = Callable[[], MapStyleABC]
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class S2STLayout(StrEnum):
    PAIRS = auto()
    SOURCES = auto()


class S2STStage(StrEnum):
    SOURCE = auto()
    TRANSLATION = auto()
    TTS = auto()


class GrowthPhase(StrEnum):
    INITIAL = auto()
    LANGUAGE_BACKFILL = auto()
    LANGUAGE_SOURCES = auto()
    VERTICAL = auto()
    IDLE = auto()


@dataclass(frozen=True)
class SpeakerList:
    speakers: tuple[str, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        _speaker_ids(self.speakers)
        _integer("seed", self.seed, minimum=0)


@dataclass(frozen=True)
class ReferenceAudio:
    """Use each source row's configured audio item as its immutable voice."""


Voice = Union[SpeakerList, ReferenceAudio]


@dataclass(frozen=True)
class SourceSlot:
    name: str
    dataset_id: str
    dataset: DatasetFactory = field(repr=False, compare=False)
    text: Reference = (Role.DEFAULT, Modality.TEXT)
    audio: Reference | None = None

    def __post_init__(self) -> None:
        _identifier("SourceSlot.name", self.name)
        _identifier("SourceSlot.dataset_id", self.dataset_id)
        if not callable(self.dataset):
            raise TypeError("SourceSlot.dataset must be callable.")
        _reference("SourceSlot.text", self.text, Modality.TEXT)
        if self.audio is not None:
            _reference("SourceSlot.audio", self.audio, Modality.AUDIO)

    @property
    def signature(self) -> SourceSlotSignature:
        return SourceSlotSignature(
            name=self.name,
            dataset_id=self.dataset_id,
            text=self.text,
            audio=self.audio,
        )


@dataclass(frozen=True)
class LanguageSources:
    language: Lang
    sources: tuple[SourceSlot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.language, Lang) or self.language is Lang.UND:
            raise ValueError("LanguageSources.language must be a declared Lang value.")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("LanguageSources.sources must be a non-empty tuple.")
        if any(not isinstance(source, SourceSlot) for source in self.sources):
            raise TypeError("LanguageSources.sources must contain SourceSlot values.")
        _unique("source slot names", (source.name for source in self.sources))


@dataclass(frozen=True)
class Growth:
    initial_sources: int
    interval_sources: int

    def __post_init__(self) -> None:
        _integer("initial_sources", self.initial_sources, minimum=1)
        _integer("interval_sources", self.interval_sources, minimum=1)


@dataclass(frozen=True)
class SourceSlotSignature:
    name: str
    dataset_id: str
    text: Reference
    audio: Reference | None

    def __post_init__(self) -> None:
        _identifier("SourceSlotSignature.name", self.name)
        _identifier("SourceSlotSignature.dataset_id", self.dataset_id)
        _reference("SourceSlotSignature.text", self.text, Modality.TEXT)
        if self.audio is not None:
            _reference("SourceSlotSignature.audio", self.audio, Modality.AUDIO)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "text": _reference_dict(self.text),
            "audio": None if self.audio is None else _reference_dict(self.audio),
        }

    @classmethod
    def from_dict(cls, value: object) -> SourceSlotSignature:
        data = _dict(value, "source slot declaration")
        return cls(
            name=_string(data.get("name"), "source slot name"),
            dataset_id=_string(data.get("dataset_id"), "source slot dataset_id"),
            text=_reference_value(data.get("text"), "source slot text"),
            audio=(
                None
                if data.get("audio") is None
                else _reference_value(data.get("audio"), "source slot audio")
            ),
        )


@dataclass(frozen=True)
class LanguageDeclaration:
    language: Lang
    sources: tuple[SourceSlotSignature, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.language, Lang) or self.language is Lang.UND:
            raise ValueError("LanguageDeclaration.language must be a declared Lang value.")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("LanguageDeclaration.sources must be a non-empty tuple.")
        if any(not isinstance(source, SourceSlotSignature) for source in self.sources):
            raise TypeError(
                "LanguageDeclaration.sources must contain SourceSlotSignature values."
            )
        _unique("source slot declaration names", (source.name for source in self.sources))

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language.value,
            "sources": [source.to_dict() for source in self.sources],
        }

    @classmethod
    def from_dict(cls, value: object) -> LanguageDeclaration:
        data = _dict(value, "language declaration")
        raw_sources = _list(data.get("sources"), "language declaration sources")
        return cls(
            language=Lang(_string(data.get("language"), "language declaration language")),
            sources=tuple(SourceSlotSignature.from_dict(source) for source in raw_sources),
        )


@dataclass(frozen=True)
class S2STDeclaration:
    languages: tuple[LanguageDeclaration, ...]
    voice_mode: Literal["speaker", "reference"]
    speaker_seed: int | None
    speakers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.languages, tuple) or len(self.languages) < 2:
            raise ValueError("S2STDeclaration.languages must contain at least two languages.")
        if any(not isinstance(language, LanguageDeclaration) for language in self.languages):
            raise TypeError(
                "S2STDeclaration.languages must contain LanguageDeclaration values."
            )
        _unique("declared languages", (language.language.value for language in self.languages))
        _unique(
            "declared source slots",
            (source.name for language in self.languages for source in language.sources),
        )
        if self.voice_mode == "speaker":
            if self.speaker_seed is None:
                raise ValueError("speaker voice mode requires speaker_seed.")
            _speaker_ids(self.speakers)
        elif self.voice_mode == "reference":
            if self.speaker_seed is not None or self.speakers:
                raise ValueError(
                    "reference voice mode does not accept speaker_seed or speakers."
                )
        else:
            raise ValueError("voice_mode must be speaker or reference.")

    @property
    def revision(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "languages": [language.to_dict() for language in self.languages],
            "voice_mode": self.voice_mode,
            "speaker_seed": self.speaker_seed,
            "speakers": list(self.speakers),
        }

    @classmethod
    def from_dict(cls, value: object) -> S2STDeclaration:
        data = _dict(value, "S2ST declaration")
        raw_languages = _list(data.get("languages"), "S2ST declaration languages")
        mode = _string(data.get("voice_mode"), "S2ST declaration voice_mode")
        if mode not in {"speaker", "reference"}:
            raise ValueError("S2ST declaration voice_mode must be speaker or reference.")
        raw_seed = data.get("speaker_seed")
        seed = None if raw_seed is None else _integer("speaker_seed", raw_seed, minimum=0)
        speakers = tuple(
            _string(value, "S2ST declaration speaker")
            for value in _list(data.get("speakers"), "S2ST declaration speakers")
        )
        return cls(
            languages=tuple(
                LanguageDeclaration.from_dict(language) for language in raw_languages
            ),
            voice_mode=cast(Literal["speaker", "reference"], mode),
            speaker_seed=seed,
            speakers=speakers,
        )


@dataclass(frozen=True)
class S2STConfig:
    name: str
    languages: tuple[LanguageSources, ...]
    translator_id: str
    tts_id: str
    voice: Voice
    growth: Growth

    def __post_init__(self) -> None:
        _identifier("S2STConfig.name", self.name)
        _identifier("S2STConfig.translator_id", self.translator_id)
        _identifier("S2STConfig.tts_id", self.tts_id)
        if not isinstance(self.languages, tuple) or len(self.languages) < 2:
            raise ValueError("S2STConfig.languages must contain at least two languages.")
        if any(not isinstance(language, LanguageSources) for language in self.languages):
            raise TypeError("S2STConfig.languages must contain LanguageSources values.")
        _unique("languages", (language.language.value for language in self.languages))
        _unique(
            "source slot names",
            (source.name for language in self.languages for source in language.sources),
        )
        if not isinstance(self.growth, Growth):
            raise TypeError("S2STConfig.growth must be a Growth value.")
        if isinstance(self.voice, SpeakerList):
            pass
        elif isinstance(self.voice, ReferenceAudio):
            missing = [source.name for source in self.slots if source.audio is None]
            if missing:
                raise ValueError(
                    "reference voice requires SourceSlot.audio for: " + ", ".join(missing)
                )
        else:
            raise TypeError("S2STConfig.voice must be SpeakerList or ReferenceAudio.")

    @property
    def slots(self) -> tuple[SourceSlot, ...]:
        return tuple(source for language in self.languages for source in language.sources)

    @property
    def declaration(self) -> S2STDeclaration:
        speaker = self.voice if isinstance(self.voice, SpeakerList) else None
        return S2STDeclaration(
            languages=tuple(
                LanguageDeclaration(
                    language=language.language,
                    sources=tuple(source.signature for source in language.sources),
                )
                for language in self.languages
            ),
            voice_mode="speaker" if speaker is not None else "reference",
            speaker_seed=None if speaker is None else speaker.seed,
            speakers=() if speaker is None else speaker.speakers,
        )

    @property
    def lineage_id(self) -> str:
        speaker = self.voice if isinstance(self.voice, SpeakerList) else None
        return _digest(
            {
                "name": self.name,
                "translator_id": self.translator_id,
                "tts_id": self.tts_id,
                "voice_mode": "speaker" if speaker is not None else "reference",
                "speaker_seed": None if speaker is None else speaker.seed,
            }
        )

    def slot(self, name: str) -> tuple[Lang, SourceSlot]:
        for language in self.languages:
            for source in language.sources:
                if source.name == name:
                    return language.language, source
        raise KeyError(f"unknown S2ST source slot: {name!r}.")

    def source(self, key: SourceKey) -> SourceInput:
        if not isinstance(key, SourceKey):
            raise TypeError("key must be a SourceKey.")
        language, slot = self.slot(key.slot)
        dataset = _dataset(slot)
        try:
            if key.row >= len(dataset):
                raise IndexError(
                    f"S2ST source row {key.row} is outside slot {slot.name!r}."
                )
            sample = dataset[key.row]
        finally:
            _close_dataset(dataset)
        if not isinstance(sample, Mapping):
            raise TypeError("S2ST source datasets must return canonical Sample mappings.")
        text = _text(sample, slot.text, language)
        audio = None if slot.audio is None else _audio(sample, slot.audio)
        return SourceInput(key=key, language=language, text=text, audio=audio)


@dataclass(frozen=True, order=True)
class SourceKey:
    slot: str
    row: int

    def __post_init__(self) -> None:
        _identifier("SourceKey.slot", self.slot)
        _integer("SourceKey.row", self.row, minimum=0)

    @property
    def id(self) -> str:
        return f"{self.slot}:{self.row}"

    @classmethod
    def from_dict(cls, value: object) -> SourceKey:
        data = _dict(value, "source key")
        return cls(
            slot=_string(data.get("slot"), "source key slot"),
            row=_integer("source key row", data.get("row"), minimum=0),
        )

    def to_dict(self) -> dict[str, object]:
        return {"slot": self.slot, "row": self.row}


@dataclass(frozen=True, order=True)
class PairKey:
    source: SourceKey
    target_language: Lang

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceKey):
            raise TypeError("PairKey.source must be a SourceKey.")
        if not isinstance(self.target_language, Lang) or self.target_language is Lang.UND:
            raise ValueError("PairKey.target_language must be a declared Lang value.")

    @property
    def id(self) -> str:
        return f"{self.source.id}->{self.target_language.value}"

    @classmethod
    def from_dict(cls, value: object) -> PairKey:
        data = _dict(value, "pair key")
        return cls(
            source=SourceKey.from_dict(data.get("source")),
            target_language=Lang(
                _string(data.get("target_language"), "pair target language")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "target_language": self.target_language.value,
        }


@dataclass(frozen=True)
class SpeakerVoice:
    speaker_id: str
    pool_revision: int

    def __post_init__(self) -> None:
        _non_empty("speaker_id", self.speaker_id)
        _integer("pool_revision", self.pool_revision, minimum=0)


@dataclass(frozen=True)
class ReferenceVoice:
    source: SourceKey

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceKey):
            raise TypeError("ReferenceVoice.source must be a SourceKey.")


VoiceAssignment = Union[SpeakerVoice, ReferenceVoice]


@dataclass(frozen=True)
class SourceFamily:
    key: SourceKey
    sequence: int
    language: Lang
    voice: VoiceAssignment

    def __post_init__(self) -> None:
        if not isinstance(self.key, SourceKey):
            raise TypeError("SourceFamily.key must be a SourceKey.")
        _integer("SourceFamily.sequence", self.sequence, minimum=0)
        if not isinstance(self.language, Lang) or self.language is Lang.UND:
            raise ValueError("SourceFamily.language must be a declared Lang value.")
        if not isinstance(self.voice, (SpeakerVoice, ReferenceVoice)):
            raise TypeError("SourceFamily.voice has an unsupported type.")

    def to_dict(self) -> dict[str, object]:
        if isinstance(self.voice, SpeakerVoice):
            voice = {
                "mode": "speaker",
                "speaker_id": self.voice.speaker_id,
                "pool_revision": self.voice.pool_revision,
            }
        else:
            voice = {"mode": "reference", "source": self.voice.source.to_dict()}
        return {
            "key": self.key.to_dict(),
            "sequence": self.sequence,
            "language": self.language.value,
            "voice": voice,
        }

    @classmethod
    def from_dict(cls, value: object) -> SourceFamily:
        data = _dict(value, "source family")
        voice_data = _dict(data.get("voice"), "source family voice")
        mode = voice_data.get("mode")
        voice: VoiceAssignment
        if mode == "speaker":
            voice = SpeakerVoice(
                speaker_id=_string(voice_data.get("speaker_id"), "speaker id"),
                pool_revision=_integer(
                    "speaker pool revision",
                    voice_data.get("pool_revision"),
                    minimum=0,
                ),
            )
        elif mode == "reference":
            voice = ReferenceVoice(SourceKey.from_dict(voice_data.get("source")))
        else:
            raise ValueError("source family voice mode must be speaker or reference.")
        return cls(
            key=SourceKey.from_dict(data.get("key")),
            sequence=_integer("source family sequence", data.get("sequence"), minimum=0),
            language=Lang(_string(data.get("language"), "source family language")),
            voice=voice,
        )


@dataclass(frozen=True)
class PairPlan:
    key: PairKey
    source_sequence: int
    first_for_source: bool

    def __post_init__(self) -> None:
        if not isinstance(self.key, PairKey):
            raise TypeError("PairPlan.key must be a PairKey.")
        _integer("PairPlan.source_sequence", self.source_sequence, minimum=0)
        if type(self.first_for_source) is not bool:
            raise TypeError("PairPlan.first_for_source must be a boolean.")

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.to_dict(),
            "source_sequence": self.source_sequence,
            "first_for_source": self.first_for_source,
        }

    @classmethod
    def from_dict(cls, value: object) -> PairPlan:
        data = _dict(value, "pair plan")
        first = data.get("first_for_source")
        if type(first) is not bool:
            raise TypeError("pair plan first_for_source must be a boolean.")
        return cls(
            key=PairKey.from_dict(data.get("key")),
            source_sequence=_integer(
                "pair source_sequence", data.get("source_sequence"), minimum=0
            ),
            first_for_source=first,
        )


@dataclass(frozen=True)
class SlotCursor:
    slot: str
    next_row: int = 0
    sample_count: int | None = None
    exhausted: bool = False

    def __post_init__(self) -> None:
        _identifier("SlotCursor.slot", self.slot)
        _integer("SlotCursor.next_row", self.next_row, minimum=0)
        if self.sample_count is not None:
            _integer("SlotCursor.sample_count", self.sample_count, minimum=0)
        if type(self.exhausted) is not bool:
            raise TypeError("SlotCursor.exhausted must be a boolean.")

    def to_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "next_row": self.next_row,
            "sample_count": self.sample_count,
            "exhausted": self.exhausted,
        }

    @classmethod
    def from_dict(cls, value: object) -> SlotCursor:
        data = _dict(value, "slot cursor")
        count = data.get("sample_count")
        exhausted = data.get("exhausted")
        if type(exhausted) is not bool:
            raise TypeError("slot cursor exhausted must be a boolean.")
        return cls(
            slot=_string(data.get("slot"), "slot cursor name"),
            next_row=_integer("slot cursor next_row", data.get("next_row"), minimum=0),
            sample_count=(
                None
                if count is None
                else _integer("slot cursor sample_count", count, minimum=0)
            ),
            exhausted=exhausted,
        )


@dataclass(frozen=True)
class LanguageCatchup:
    language: Lang
    source_target: int

    def __post_init__(self) -> None:
        if not isinstance(self.language, Lang) or self.language is Lang.UND:
            raise ValueError("LanguageCatchup.language must be a declared Lang value.")
        _integer("LanguageCatchup.source_target", self.source_target, minimum=0)

    def to_dict(self) -> dict[str, object]:
        return {"language": self.language.value, "source_target": self.source_target}

    @classmethod
    def from_dict(cls, value: object) -> LanguageCatchup:
        data = _dict(value, "language catchup")
        return cls(
            language=Lang(_string(data.get("language"), "catchup language")),
            source_target=_integer(
                "catchup source_target", data.get("source_target"), minimum=0
            ),
        )


@dataclass(frozen=True)
class S2STState:
    lineage_id: str
    declaration: S2STDeclaration
    revision: int = -1
    speaker_pool_revision: int = 0
    families: tuple[SourceFamily, ...] = ()
    pairs: tuple[PairPlan, ...] = ()
    cursors: tuple[SlotCursor, ...] = ()
    next_slot_index: int = 0
    catchup: tuple[LanguageCatchup, ...] = ()

    def __post_init__(self) -> None:
        _non_empty("S2STState.lineage_id", self.lineage_id)
        if not isinstance(self.declaration, S2STDeclaration):
            raise TypeError("S2STState.declaration must be an S2STDeclaration.")
        _integer("S2STState.revision", self.revision, minimum=-1)
        _integer("S2STState.speaker_pool_revision", self.speaker_pool_revision, minimum=0)
        _integer("S2STState.next_slot_index", self.next_slot_index, minimum=0)
        _unique("family source keys", (family.key.id for family in self.families))
        _unique("pair keys", (pair.key.id for pair in self.pairs))
        _unique("slot cursors", (cursor.slot for cursor in self.cursors))
        _unique("catchup languages", (item.language.value for item in self.catchup))

    @classmethod
    def empty(cls, config: S2STConfig) -> S2STState:
        return cls(
            lineage_id=config.lineage_id,
            declaration=config.declaration,
            cursors=tuple(SlotCursor(source.name) for source in config.slots),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "declaration": self.declaration.to_dict(),
            "revision": self.revision,
            "speaker_pool_revision": self.speaker_pool_revision,
            "families": [family.to_dict() for family in self.families],
            "pairs": [pair.to_dict() for pair in self.pairs],
            "cursors": [cursor.to_dict() for cursor in self.cursors],
            "next_slot_index": self.next_slot_index,
            "catchup": [item.to_dict() for item in self.catchup],
        }

    @classmethod
    def from_dict(cls, value: object) -> S2STState:
        data = _dict(value, "S2ST state")
        return cls(
            lineage_id=_string(data.get("lineage_id"), "S2ST state lineage_id"),
            declaration=S2STDeclaration.from_dict(data.get("declaration")),
            revision=_integer("S2ST state revision", data.get("revision"), minimum=-1),
            speaker_pool_revision=_integer(
                "S2ST state speaker_pool_revision",
                data.get("speaker_pool_revision"),
                minimum=0,
            ),
            families=tuple(
                SourceFamily.from_dict(item)
                for item in _list(data.get("families"), "S2ST state families")
            ),
            pairs=tuple(
                PairPlan.from_dict(item)
                for item in _list(data.get("pairs"), "S2ST state pairs")
            ),
            cursors=tuple(
                SlotCursor.from_dict(item)
                for item in _list(data.get("cursors"), "S2ST state cursors")
            ),
            next_slot_index=_integer(
                "S2ST state next_slot_index", data.get("next_slot_index"), minimum=0
            ),
            catchup=tuple(
                LanguageCatchup.from_dict(item)
                for item in _list(data.get("catchup"), "S2ST state catchup")
            ),
        )


@dataclass(frozen=True)
class GrowthPlan:
    phase: GrowthPhase
    revision: int | None
    added_families: tuple[SourceFamily, ...]
    added_pairs: tuple[PairPlan, ...]
    state: S2STState

    @property
    def changed(self) -> bool:
        return self.revision is not None


@dataclass(frozen=True)
class S2STView:
    layout: S2STLayout = S2STLayout.PAIRS
    source_languages: frozenset[Lang] | None = None
    target_languages: frozenset[Lang] | None = None
    source_slots: frozenset[str] | None = None
    speakers: frozenset[str] | None = None
    schema: Schema | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, S2STLayout):
            raise TypeError("S2STView.layout must be an S2STLayout.")
        _optional_set("source_languages", self.source_languages, Lang)
        _optional_set("target_languages", self.target_languages, Lang)
        _optional_set("source_slots", self.source_slots, str, non_empty=True)
        _optional_set("speakers", self.speakers, str, non_empty=True)
        if self.layout is S2STLayout.SOURCES and self.target_languages is not None:
            raise ValueError("sources layout does not accept target_languages.")
        if self.schema is not None and not isinstance(self.schema, Mapping):
            raise TypeError("S2STView.schema must be a Schema mapping or None.")


@dataclass(frozen=True)
class SourceInput:
    key: SourceKey
    language: Lang
    text: TextItem
    audio: AudioItem | None


def validate_successor(previous: S2STDeclaration, current: S2STDeclaration) -> None:
    """Require append-only languages, source slots, and speaker ids."""

    if previous.voice_mode != current.voice_mode:
        raise ValueError("S2ST voice mode cannot change within one lineage.")
    if previous.speaker_seed != current.speaker_seed:
        raise ValueError("S2ST speaker seed cannot change within one lineage.")
    if len(current.languages) < len(previous.languages):
        raise ValueError("S2ST languages cannot be removed within one lineage.")
    for index, old_language in enumerate(previous.languages):
        new_language = current.languages[index]
        if new_language.language is not old_language.language:
            raise ValueError("existing S2ST languages cannot be removed or reordered.")
        if len(new_language.sources) < len(old_language.sources):
            raise ValueError("existing S2ST source slots cannot be removed.")
        if new_language.sources[: len(old_language.sources)] != old_language.sources:
            raise ValueError("existing S2ST source slots cannot change or be reordered.")
    if current.speakers[: len(previous.speakers)] != previous.speakers:
        raise ValueError("existing S2ST speakers cannot change or be reordered.")


def _dataset(slot: SourceSlot) -> MapStyleABC:
    dataset = slot.dataset()
    if not isinstance(dataset, MapStyleABC):
        raise TypeError(
            f"S2ST source slot {slot.name!r} factory must return MapStyleABC."
        )
    return dataset


def _close_dataset(dataset: MapStyleABC) -> None:
    close = getattr(dataset, "close", None)
    if callable(close):
        close()
        return
    prepared = getattr(dataset, "dataset", None)
    close = getattr(prepared, "close", None)
    if callable(close):
        close()


def _text(sample: Sample, reference: Reference, language: Lang) -> TextItem:
    try:
        item = sample[reference]
    except KeyError as exc:
        raise KeyError(f"S2ST source sample is missing text reference {reference!r}.") from exc
    if not isinstance(item, TextItem):
        raise TypeError(f"S2ST source reference {reference!r} must contain a TextItem.")
    text = item.views.get(TextView.TEXT)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("S2ST source text must contain non-empty TextView.TEXT.")
    existing = item.meta.get(TextMeta.LANG)
    if existing is not None and existing is not language:
        raise ValueError(
            f"S2ST source language {existing!r} does not match declared {language!r}."
        )
    return TextItem(views=item.views, meta={**item.meta, TextMeta.LANG: language})


def _audio(sample: Sample, reference: Reference) -> AudioItem:
    try:
        item = sample[reference]
    except KeyError as exc:
        raise KeyError(f"S2ST source sample is missing audio reference {reference!r}.") from exc
    if not isinstance(item, AudioItem):
        raise TypeError(f"S2ST source reference {reference!r} must contain an AudioItem.")
    return item


def _reference(name: str, value: object, modality: Modality) -> Reference:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or not isinstance(value[0], Role)
        or value[1] is not modality
    ):
        raise TypeError(f"{name} must be a (Role, Modality.{modality.name}) reference.")
    return value


def _reference_dict(value: Reference) -> dict[str, str]:
    return {"role": value[0].value, "modality": value[1].value}


def _reference_value(value: object, name: str) -> Reference:
    data = _dict(value, name)
    return (
        Role(_string(data.get("role"), f"{name} role")),
        Modality(_string(data.get("modality"), f"{name} modality")),
    )


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identifier(name: str, value: object) -> str:
    text = _non_empty(name, value)
    if _ID_PATTERN.fullmatch(text) is None:
        raise ValueError(
            f"{name} must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-'."
        )
    return text


def _speaker_ids(values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("speakers must be a non-empty tuple.")
    speakers = tuple(_non_empty("speaker", value) for value in values)
    _unique("speakers", speakers)
    return speakers


def _unique(name: str, values) -> None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{name} must be unique; duplicate {value!r}.")
        seen.add(value)


def _optional_set(
    name: str,
    value: object,
    item_type: type,
    *,
    non_empty: bool = False,
) -> None:
    if value is None:
        return
    if not isinstance(value, frozenset):
        raise TypeError(f"{name} must be a frozenset or None.")
    for item in value:
        if not isinstance(item, item_type):
            raise TypeError(f"{name} contains an invalid value.")
        if non_empty and isinstance(item, str) and not item:
            raise ValueError(f"{name} must contain non-empty strings.")


def _dict(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list.")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _non_empty(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")
    return value


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


__all__ = [
    "DatasetFactory",
    "Growth",
    "GrowthPhase",
    "GrowthPlan",
    "LanguageCatchup",
    "LanguageDeclaration",
    "LanguageSources",
    "PairKey",
    "PairPlan",
    "ReferenceAudio",
    "ReferenceVoice",
    "S2STConfig",
    "S2STDeclaration",
    "S2STLayout",
    "S2STStage",
    "S2STState",
    "S2STView",
    "SlotCursor",
    "SourceFamily",
    "SourceInput",
    "SourceKey",
    "SourceSlot",
    "SourceSlotSignature",
    "SpeakerList",
    "SpeakerVoice",
    "Voice",
    "VoiceAssignment",
    "validate_successor",
]
