from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..types import Modality, Role, Sample, TextItem, TextMeta, TextView

SpeakerMode = Literal["aligned", "cycle"]
TextRef = tuple[Role, Modality]


@dataclass(frozen=True)
class SpeakerIdDataset:
    dataset: Any
    speaker_ids: Sequence[str]
    mode: SpeakerMode = "aligned"
    text_ref: TextRef = (Role.DEFAULT, Modality.TEXT)

    def __post_init__(self) -> None:
        speakers = tuple(_speaker_id(value) for value in self.speaker_ids)
        object.__setattr__(self, "speaker_ids", speakers)
        if self.mode not in {"aligned", "cycle"}:
            raise ValueError("mode must be 'aligned' or 'cycle'.")
        if self.mode == "aligned" and len(speakers) != len(self.dataset):
            raise ValueError("aligned speaker ids must match dataset length.")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        sample = dict(self.dataset[index])
        speaker_id = speaker_for_index(index, self.speaker_ids, self.mode)
        item = sample[self.text_ref]
        if not isinstance(item, TextItem):
            raise TypeError(f"{self.text_ref!r} must contain a TextItem.")
        existing = item.views.get(TextView.SPEAKERS)
        if existing is not None and existing != speaker_id:
            raise ValueError(
                f"sample {index} already has speaker id {existing!r}, "
                f"but assignment selected {speaker_id!r}."
            )
        sample[self.text_ref] = TextItem(
            views={**item.views, TextView.SPEAKERS: speaker_id},
            meta=item.meta,
        )
        return sample


@dataclass(frozen=True)
class SpeakerCartesianDataset:
    dataset: Any
    speaker_ids: Sequence[str]
    text_ref: TextRef = (Role.DEFAULT, Modality.TEXT)

    def __post_init__(self) -> None:
        speakers = tuple(_speaker_id(value) for value in self.speaker_ids)
        if not speakers:
            raise ValueError("speaker_ids must not be empty.")
        object.__setattr__(self, "speaker_ids", speakers)

    def __len__(self) -> int:
        return len(self.dataset) * len(self.speaker_ids)

    def __getitem__(self, index: int) -> Sample:
        source_index, speaker_index = speaker_cartesian_indexes(
            index,
            len(self.speaker_ids),
        )
        sample = dict(self.dataset[source_index])
        speaker_id = self.speaker_ids[speaker_index]
        item = sample[self.text_ref]
        if not isinstance(item, TextItem):
            raise TypeError(f"{self.text_ref!r} must contain a TextItem.")
        existing = item.views.get(TextView.SPEAKERS)
        if existing is not None and existing != speaker_id:
            raise ValueError(
                f"source sample {source_index} already has speaker id {existing!r}, "
                f"but assignment selected {speaker_id!r}."
            )
        sample[self.text_ref] = TextItem(
            views={**item.views, TextView.SPEAKERS: speaker_id},
            meta={
                **item.meta,
                TextMeta.SOURCE_INDEX: source_index,
            },
        )
        return sample


def speaker_for_index(
    index: int,
    speaker_ids: Sequence[str],
    mode: SpeakerMode,
) -> str:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer.")
    if index < 0:
        raise ValueError("index must be non-negative.")
    speakers = tuple(_speaker_id(value) for value in speaker_ids)
    if not speakers:
        raise ValueError("speaker_ids must not be empty.")
    if mode == "aligned":
        if index >= len(speakers):
            raise IndexError("speaker index exceeds aligned speaker list.")
        return speakers[index]
    if mode == "cycle":
        return speakers[index % len(speakers)]
    raise ValueError("mode must be 'aligned' or 'cycle'.")


def speaker_cartesian_indexes(index: int, speaker_count: int) -> tuple[int, int]:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer.")
    if index < 0:
        raise ValueError("index must be non-negative.")
    if isinstance(speaker_count, bool) or not isinstance(speaker_count, int):
        raise TypeError("speaker_count must be an integer.")
    if speaker_count <= 0:
        raise ValueError("speaker_count must be positive.")
    return divmod(index, speaker_count)


def _speaker_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("speaker ids must be non-empty strings.")
    return value
