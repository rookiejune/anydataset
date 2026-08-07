from __future__ import annotations

import hashlib

import torch

from ...dataset import MapStyleABC
from ...types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Lang,
    Modality,
    Role,
    Sample,
    TextItem,
    TextMeta,
    TextView,
)
from .catalog import PairIndexRecord
from .dataset import project_sample
from .model import ReferenceAudio, S2STConfig, S2STLayout, S2STView, SpeakerList


class ToyS2STDataset(MapStyleABC):
    """Deterministic schema-compatible S2ST pairs without model inference."""

    def __init__(
        self,
        config: S2STConfig,
        *,
        sources: int | None = None,
        view: S2STView = S2STView(),
        frames: int = 8,
        sample_rate: int = 16000,
    ) -> None:
        if not isinstance(config, S2STConfig):
            raise TypeError("config must be an S2STConfig.")
        if sources is None:
            sources = config.growth.initial_sources
        _positive_int("sources", sources)
        _positive_int("frames", frames)
        _positive_int("sample_rate", sample_rate)
        if not isinstance(view, S2STView):
            raise TypeError("view must be an S2STView.")
        self.config = config
        self.view = view
        self.frames = frames
        self.sample_rate = sample_rate
        self._rows = _rows(config, sources, view)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Sample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("toy S2ST index out of range.")
        record = self._rows[index]
        source_lang = Lang(record.source_language)
        target_lang = Lang(record.target_language)
        source = _audio(record.source_sequence, source_lang, self.frames, self.sample_rate)
        target = _audio(record.source_sequence, target_lang, self.frames, self.sample_rate)
        if record.speaker_id is not None:
            source = _speaker(source, record.speaker_id)
            target = _speaker(target, record.speaker_id)
        sample = {
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={
                    TextView.TEXT: (
                        f"toy {source_lang.value} source {record.source_sequence}"
                    )
                },
                meta={TextMeta.LANG: source_lang},
            ),
            (Role.SOURCE, Modality.AUDIO): source,
            (Role.TARGET, Modality.TEXT): TextItem(
                views={
                    TextView.TEXT: (
                        f"toy {target_lang.value} translation {record.source_sequence}"
                    )
                },
                meta={TextMeta.LANG: target_lang},
            ),
            (Role.TARGET, Modality.AUDIO): target,
        }
        return project_sample(sample, self.view)


def _rows(
    config: S2STConfig,
    source_count: int,
    view: S2STView,
) -> list[PairIndexRecord]:
    languages = tuple(item.language for item in config.languages)
    slots = tuple(
        (language.language, slot)
        for language in config.languages
        for slot in language.sources
    )
    next_rows = {slot.name: 0 for _language, slot in slots}
    rows: list[PairIndexRecord] = []
    for sequence in range(source_count):
        source_language, source_slot = slots[sequence % len(slots)]
        source_row = next_rows[source_slot.name]
        next_rows[source_slot.name] = source_row + 1
        speaker = _speaker_id(config, source_slot.name, source_row)
        first = True
        for target_language in languages:
            if target_language is source_language:
                continue
            record = PairIndexRecord(
                pair_id=(
                    f"{source_slot.name}:{source_row}->{target_language.value}"
                ),
                source_slot=source_slot.name,
                source_row=source_row,
                source_sequence=sequence,
                source_language=source_language.value,
                target_language=target_language.value,
                speaker_id=speaker,
                first_for_source=first,
            )
            first = False
            if _matches(record, view):
                rows.append(record)
    if view.layout is S2STLayout.SOURCES:
        seen: set[tuple[str, int]] = set()
        rows = [
            record
            for record in rows
            if record.first_for_source
            and (record.source_slot, record.source_row) not in seen
            and not seen.add((record.source_slot, record.source_row))
        ]
    return rows


def _matches(record: PairIndexRecord, view: S2STView) -> bool:
    if view.source_languages is not None and Lang(record.source_language) not in view.source_languages:
        return False
    if view.target_languages is not None and Lang(record.target_language) not in view.target_languages:
        return False
    if view.source_slots is not None and record.source_slot not in view.source_slots:
        return False
    return not (view.speakers is not None and record.speaker_id not in view.speakers)


def _speaker_id(config: S2STConfig, slot: str, row: int) -> str | None:
    if isinstance(config.voice, ReferenceAudio):
        return None
    if not isinstance(config.voice, SpeakerList):
        raise TypeError("unsupported S2ST voice configuration.")
    payload = f"{config.voice.seed}:0:{slot}:{row}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return config.voice.speakers[value % len(config.voice.speakers)]


def _audio(sequence: int, language: Lang, frames: int, sample_rate: int) -> AudioItem:
    offset = sequence + list(Lang).index(language)
    waveform = torch.arange(frames, dtype=torch.float32).unsqueeze(0) + float(offset)
    return AudioItem(
        views={AudioView.WAVEFORM: (waveform, sample_rate)},
        meta={AudioMeta.DURATION: frames / sample_rate},
    )


def _speaker(item: AudioItem, speaker_id: str) -> AudioItem:
    return AudioItem(
        views=item.views,
        meta={**item.meta, AudioMeta.SPEAKER_ID: speaker_id},
    )


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


__all__ = ["ToyS2STDataset"]
