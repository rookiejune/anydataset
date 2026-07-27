from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, cast

import torch
import torch.nn.functional as F

from ..types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Modality,
    Role,
    Sample,
    TextItem,
    TextMeta,
    TextView,
)

SpeakerMode = Literal["aligned", "cycle"]
TextRef = tuple[Role, Modality]


class MapDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Sample: ...


class SpeakerAssignment:
    """Assign one speaker id sequence to a text reference."""

    def __init__(
        self,
        speaker_ids: Sequence[str],
        mode: SpeakerMode = "aligned",
    ) -> None:
        self._mode: SpeakerMode = _speaker_mode(mode)
        self._speaker_ids: tuple[str, ...] = _speaker_ids(
            speaker_ids,
            allow_empty=self._mode == "aligned",
        )

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        return self._speaker_ids

    @property
    def mode(self) -> SpeakerMode:
        return self._mode


class SpeakerIdDataset:
    """Add speaker views to one or more text items without changing length."""

    def __init__(
        self,
        dataset: MapDataset,
        assignments: Mapping[TextRef, SpeakerAssignment],
    ) -> None:
        self._dataset: MapDataset = dataset
        self._assignments: dict[TextRef, SpeakerAssignment] = _assignments(
            assignments,
            dataset_length=len(dataset),
        )

    @property
    def dataset(self) -> MapDataset:
        return self._dataset

    @property
    def assignments(self) -> Mapping[TextRef, SpeakerAssignment]:
        return MappingProxyType(self._assignments)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        sample = dict(self.dataset[index])
        for text_ref, assignment in self._assignments.items():
            speaker_id = _speaker_for_index(
                index,
                assignment.speaker_ids,
                assignment.mode,
            )
            item = sample.get(text_ref)
            if not isinstance(item, TextItem):
                raise TypeError(f"{text_ref!r} must contain a TextItem.")
            existing = item.views.get(TextView.SPEAKERS)
            if existing is not None and existing != speaker_id:
                raise ValueError(
                    f"sample {index} already has speaker id {existing!r} at {text_ref!r}, but assignment selected {speaker_id!r}."
                )
            sample[text_ref] = TextItem(
                views={**item.views, TextView.SPEAKERS: speaker_id},
                meta=item.meta,
            )
        return sample


class SpeakerCartesianDataset:
    """Expand one text reference over every configured speaker id."""

    def __init__(
        self,
        dataset: MapDataset,
        speaker_ids: Sequence[str],
        text_ref: TextRef = (Role.DEFAULT, Modality.TEXT),
    ) -> None:
        self._dataset: MapDataset = dataset
        self._speaker_ids: tuple[str, ...] = _speaker_ids(speaker_ids)
        self._text_ref: TextRef = _text_ref(text_ref)

    @property
    def dataset(self) -> MapDataset:
        return self._dataset

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        return self._speaker_ids

    @property
    def text_ref(self) -> TextRef:
        return self._text_ref

    def __len__(self) -> int:
        return len(self.dataset) * len(self.speaker_ids)

    def __getitem__(self, index: int) -> Sample:
        source_index, speaker_index = speaker_cartesian_indexes(
            index,
            len(self.speaker_ids),
        )
        sample = dict(self.dataset[source_index])
        speaker_id = self.speaker_ids[speaker_index]
        item = sample.get(self.text_ref)
        if not isinstance(item, TextItem):
            raise TypeError(f"{self.text_ref!r} must contain a TextItem.")
        existing = item.views.get(TextView.SPEAKERS)
        if existing is not None and existing != speaker_id:
            raise ValueError(
                f"source sample {source_index} already has speaker id {existing!r}, but assignment selected {speaker_id!r}."
            )
        sample[self.text_ref] = TextItem(
            views={**item.views, TextView.SPEAKERS: speaker_id},
            meta={
                **item.meta,
                TextMeta.SOURCE_INDEX: source_index,
            },
        )
        return sample


class GroupedSpeakerAudioDataset:
    """Group a speaker-cartesian audio dataset back onto its text axis."""

    def __init__(
        self,
        dataset: MapDataset,
        speaker_ids: Sequence[str],
        text_ref: TextRef = (Role.DEFAULT, Modality.TEXT),
        audio_ref: tuple[Role, Modality] = (Role.DEFAULT, Modality.AUDIO),
    ) -> None:
        self._dataset: MapDataset = dataset
        self._speaker_ids: tuple[str, ...] = _speaker_ids(speaker_ids)
        self._text_ref: TextRef = _text_ref(text_ref)
        self._audio_ref: tuple[Role, Modality] = _audio_ref(audio_ref)
        if len(dataset) % len(self.speaker_ids) != 0:
            raise ValueError("flat speaker dataset length must be divisible by speaker count.")

    @property
    def dataset(self) -> MapDataset:
        return self._dataset

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        return self._speaker_ids

    @property
    def text_ref(self) -> TextRef:
        return self._text_ref

    @property
    def audio_ref(self) -> tuple[Role, Modality]:
        return self._audio_ref

    def __len__(self) -> int:
        return len(self.dataset) // len(self.speaker_ids)

    def __getitem__(self, index: int) -> Sample:
        _index(index, len(self))
        start = index * len(self.speaker_ids)
        first_sample = self.dataset[start]
        first_text = _text_item(first_sample.get(self.text_ref), self.text_ref)
        source_text = _source_text(first_text)
        grouped: Sample = dict(first_sample)
        grouped[self.text_ref] = TextItem(
            views=source_text.views,
            meta={
                **source_text.meta,
                TextMeta.SOURCE_INDEX: index,
            },
        )

        waveforms: list[torch.Tensor] = []
        lengths: list[int] = []
        sample_rate: int | None = None
        for offset, speaker_id in enumerate(self.speaker_ids):
            sample = first_sample if offset == 0 else self.dataset[start + offset]
            text_item = _text_item(sample.get(self.text_ref), self.text_ref)
            if _source_text(text_item) != source_text:
                raise ValueError(
                    f"flat sample {start + offset} text content differs from source index {index}."
                )
            source_index = text_item.meta.get(TextMeta.SOURCE_INDEX)
            if source_index != index:
                raise ValueError(
                    f"flat sample {start + offset} has source index {source_index!r}; expected {index!r}."
                )
            actual_speaker = text_item.views.get(TextView.SPEAKERS)
            if actual_speaker != speaker_id:
                raise ValueError(
                    f"flat sample {start + offset} has speaker {actual_speaker!r}; expected {speaker_id!r}."
                )
            audio_item = _audio_item(sample.get(self.audio_ref), self.audio_ref)
            audio_speaker = audio_item.meta.get(AudioMeta.SPEAKER_ID)
            if audio_speaker is not None and audio_speaker != speaker_id:
                raise ValueError(
                    f"flat sample {start + offset} has audio speaker {audio_speaker!r}; expected {speaker_id!r}."
                )
            waveform, current_sample_rate = _waveform(
                audio_item.views.get(AudioView.WAVEFORM)
            )
            if sample_rate is None:
                sample_rate = current_sample_rate
            elif current_sample_rate != sample_rate:
                raise ValueError("grouped speaker waveforms must share one sample rate.")
            waveforms.append(waveform)
            lengths.append(int(waveform.shape[-1]))

        if sample_rate is None:
            raise ValueError("grouped speaker waveforms must not be empty.")
        grouped[self.audio_ref] = AudioItem(
            views={
                AudioView.WAVEFORM: (_stack_waveforms(waveforms), sample_rate),
                AudioView.SPEAKERS: self.speaker_ids,
                AudioView.SPEAKER_LENGTHS: _speaker_lengths(lengths),
            },
            meta={AudioMeta.DURATION: max(lengths) / sample_rate},
        )
        return grouped


@dataclass(frozen=True)
class SpeakerAudioBlock:
    """One materialized rectangular selection from a speaker audio grid."""

    text_indices: tuple[int, ...]
    texts: tuple[str, ...]
    speaker_ids: tuple[str, ...]
    waveforms: torch.Tensor
    lengths: torch.Tensor
    sample_rate: int

    def __post_init__(self) -> None:
        if not isinstance(self.text_indices, tuple) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.text_indices
        ):
            raise TypeError("text_indices must be a tuple of non-negative integers.")
        if not isinstance(self.texts, tuple) or any(
            not isinstance(text, str) for text in self.texts
        ):
            raise TypeError("texts must be a tuple of strings.")
        if len(self.text_indices) != len(self.texts):
            raise ValueError("text_indices and texts must have the same length.")
        if not isinstance(self.speaker_ids, tuple):
            raise TypeError("speaker_ids must be a tuple of strings.")
        _unique_speaker_ids(self.speaker_ids)
        if not isinstance(self.waveforms, torch.Tensor):
            raise TypeError("waveforms must be a Tensor.")
        if self.waveforms.ndim != 4:
            raise ValueError(
                "waveforms must have shape [text, speaker, channel, time]."
            )
        shape = (len(self.texts), len(self.speaker_ids))
        if tuple(self.waveforms.shape[:2]) != shape:
            raise ValueError(
                f"waveforms grid shape {tuple(self.waveforms.shape[:2])} does not match {shape}."
            )
        if not isinstance(self.lengths, torch.Tensor):
            raise TypeError("lengths must be a Tensor.")
        if self.lengths.dtype is not torch.int64:
            raise TypeError("lengths must have dtype torch.int64.")
        if tuple(self.lengths.shape) != shape:
            raise ValueError(f"lengths must have shape {shape}.")
        if bool(torch.any(self.lengths < 0).item()):
            raise ValueError("lengths must be non-negative.")
        if bool(torch.any(self.lengths > self.waveforms.shape[-1]).item()):
            raise ValueError("lengths must not exceed the waveform time axis.")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.texts), len(self.speaker_ids)


class SpeakerAudioGrid:
    """Expose a flat text-major speaker store as a logical two-axis grid."""

    def __init__(
        self,
        dataset: MapDataset,
        speaker_ids: Sequence[str],
        text_ref: TextRef = (Role.DEFAULT, Modality.TEXT),
        audio_ref: tuple[Role, Modality] = (Role.DEFAULT, Modality.AUDIO),
    ) -> None:
        self._cells: MapDataset = dataset
        self._speaker_ids: tuple[str, ...] = _unique_speaker_ids(speaker_ids)
        self._rows = GroupedSpeakerAudioDataset(
            dataset,
            self.speaker_ids,
            text_ref=text_ref,
            audio_ref=audio_ref,
        )

    @property
    def cells(self) -> MapDataset:
        return self._cells

    @property
    def dataset(self) -> MapDataset:
        """Return the flat cell dataset retained for grouped-dataset compatibility."""

        return self.cells

    @property
    def rows(self) -> GroupedSpeakerAudioDataset:
        return self._rows

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        return self._speaker_ids

    @property
    def text_ref(self) -> TextRef:
        return self.rows.text_ref

    @property
    def audio_ref(self) -> tuple[Role, Modality]:
        return self.rows.audio_ref

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.speaker_ids)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Sample:
        return self.rows[index]

    def select(
        self,
        *,
        text: int | None = None,
        speaker: str | None = None,
    ) -> SpeakerAudioSelection:
        """Select one text row, one speaker column, one cell, or the full grid."""

        text_indices = (
            tuple(range(len(self)))
            if text is None
            else (_selection_index(text, len(self)),)
        )
        if speaker is None:
            speaker_indices = tuple(range(len(self.speaker_ids)))
        else:
            speaker_id = _speaker_id(speaker)
            try:
                speaker_indices = (self.speaker_ids.index(speaker_id),)
            except ValueError as error:
                raise ValueError(
                    f"speaker id {speaker_id!r} is not present in the grid."
                ) from error
        return SpeakerAudioSelection(self, text_indices, speaker_indices)

    def _load(
        self,
        text_indices: tuple[int, ...],
        speaker_indices: tuple[int, ...],
    ) -> SpeakerAudioBlock:
        if not text_indices or not speaker_indices:
            raise ValueError("speaker audio selection must not be empty.")
        texts: list[str] = []
        waveforms: list[torch.Tensor] = []
        lengths: list[int] = []
        sample_rate: int | None = None
        for text_index in text_indices:
            source_text: TextItem | None = None
            for speaker_index in speaker_indices:
                flat_index = text_index * len(self.speaker_ids) + speaker_index
                sample = self.cells[flat_index]
                text_item = _text_item(sample.get(self.text_ref), self.text_ref)
                current_text = _source_text(text_item)
                if source_text is None:
                    source_text = current_text
                elif current_text != source_text:
                    raise ValueError(
                        f"flat sample {flat_index} text content differs from source index {text_index}."
                    )
                source_index = text_item.meta.get(TextMeta.SOURCE_INDEX)
                if source_index != text_index:
                    raise ValueError(
                        f"flat sample {flat_index} has source index {source_index!r}; expected {text_index!r}."
                    )
                speaker_id = self.speaker_ids[speaker_index]
                actual_speaker = text_item.views.get(TextView.SPEAKERS)
                if actual_speaker != speaker_id:
                    raise ValueError(
                        f"flat sample {flat_index} has speaker {actual_speaker!r}; expected {speaker_id!r}."
                    )
                audio_item = _audio_item(sample.get(self.audio_ref), self.audio_ref)
                audio_speaker = audio_item.meta.get(AudioMeta.SPEAKER_ID)
                if audio_speaker is not None and audio_speaker != speaker_id:
                    raise ValueError(
                        f"flat sample {flat_index} has audio speaker {audio_speaker!r}; expected {speaker_id!r}."
                    )
                waveform, current_sample_rate = _waveform(
                    audio_item.views.get(AudioView.WAVEFORM)
                )
                if sample_rate is None:
                    sample_rate = current_sample_rate
                elif current_sample_rate != sample_rate:
                    raise ValueError(
                        "selected speaker waveforms must share one sample rate."
                    )
                waveforms.append(waveform)
                lengths.append(int(waveform.shape[-1]))
            if source_text is None:
                raise ValueError("speaker audio selection must include a speaker.")
            texts.append(_text(source_text))
        if sample_rate is None:
            raise ValueError("speaker audio selection must not be empty.")
        stacked = _stack_waveforms(waveforms)
        grid_shape = (len(text_indices), len(speaker_indices))
        return SpeakerAudioBlock(
            text_indices=text_indices,
            texts=tuple(texts),
            speaker_ids=tuple(
                self.speaker_ids[speaker_index] for speaker_index in speaker_indices
            ),
            waveforms=stacked.reshape(*grid_shape, *stacked.shape[1:]),
            lengths=_speaker_lengths(lengths).reshape(grid_shape),
            sample_rate=sample_rate,
        )


class SpeakerAudioSelection:
    """A lazy rectangular selection from a speaker audio grid."""

    def __init__(
        self,
        grid: SpeakerAudioGrid,
        text_indices: tuple[int, ...],
        speaker_indices: tuple[int, ...],
    ) -> None:
        if not isinstance(grid, SpeakerAudioGrid):
            raise TypeError("grid must be a SpeakerAudioGrid.")
        if not isinstance(text_indices, tuple):
            raise TypeError("text_indices must be a tuple of integers.")
        if not isinstance(speaker_indices, tuple):
            raise TypeError("speaker_indices must be a tuple of integers.")
        for text_index in text_indices:
            _selection_index(text_index, len(grid))
        for speaker_index in speaker_indices:
            if _integer(speaker_index, "speaker index") >= len(grid.speaker_ids):
                raise IndexError("speaker index out of range.")
        if len(set(text_indices)) != len(text_indices):
            raise ValueError("text_indices must not contain duplicates.")
        if len(set(speaker_indices)) != len(speaker_indices):
            raise ValueError("speaker_indices must not contain duplicates.")
        self._grid = grid
        self._text_indices = text_indices
        self._speaker_indices = speaker_indices

    @property
    def grid(self) -> SpeakerAudioGrid:
        return self._grid

    @property
    def text_indices(self) -> tuple[int, ...]:
        return self._text_indices

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        return tuple(
            self.grid.speaker_ids[index] for index in self._speaker_indices
        )

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.text_indices), len(self.speaker_ids)

    def load(self) -> SpeakerAudioBlock:
        """Read and pad only the selected rectangle."""

        return self.grid._load(self.text_indices, self._speaker_indices)


def speaker_for_index(
    index: object,
    speaker_ids: Sequence[str],
    mode: SpeakerMode,
) -> str:
    resolved_mode = _speaker_mode(mode)
    return _speaker_for_index(
        index,
        _speaker_ids(speaker_ids, allow_empty=resolved_mode == "aligned"),
        resolved_mode,
    )


def _speaker_for_index(
    index: object,
    speaker_ids: tuple[str, ...],
    mode: SpeakerMode,
) -> str:
    resolved_index = _integer(index, "index")
    if mode == "aligned":
        if resolved_index >= len(speaker_ids):
            raise IndexError("speaker index exceeds aligned speaker list.")
        return speaker_ids[resolved_index]
    return speaker_ids[resolved_index % len(speaker_ids)]


def speaker_cartesian_indexes(index: object, speaker_count: object) -> tuple[int, int]:
    resolved_index = _integer(index, "index")
    resolved_count = _integer(speaker_count, "speaker_count")
    if resolved_count <= 0:
        raise ValueError("speaker_count must be positive.")
    return divmod(resolved_index, resolved_count)


def _assignments(
    value: object,
    *,
    dataset_length: int,
) -> dict[TextRef, SpeakerAssignment]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("assignments must be a non-empty mapping.")
    raw_assignments = cast(Mapping[object, object], value)
    assignments: dict[TextRef, SpeakerAssignment] = {}
    for ref, assignment in raw_assignments.items():
        text_ref = _text_ref(ref)
        if not isinstance(assignment, SpeakerAssignment):
            raise TypeError("assignment values must be SpeakerAssignment instances.")
        if assignment.mode == "aligned" and len(assignment.speaker_ids) != dataset_length:
            raise ValueError(
                f"aligned speaker ids for {text_ref!r} must match dataset length."
            )
        assignments[text_ref] = assignment
    return assignments


def _speaker_ids(value: object, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("speaker_ids must be a sequence of strings.")
    speakers = tuple(_speaker_id(item) for item in value)
    if not speakers and not allow_empty:
        raise ValueError("speaker_ids must not be empty.")
    return speakers


def _unique_speaker_ids(value: object) -> tuple[str, ...]:
    speakers = _speaker_ids(value)
    if len(set(speakers)) != len(speakers):
        raise ValueError("speaker_ids must not contain duplicates.")
    return speakers


def _speaker_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("speaker ids must be non-empty strings.")
    return value


def _speaker_mode(value: object) -> SpeakerMode:
    if value == "aligned":
        return "aligned"
    if value == "cycle":
        return "cycle"
    raise ValueError("mode must be 'aligned' or 'cycle'.")


def _text_ref(value: object) -> TextRef:
    if not isinstance(value, tuple):
        raise TypeError("text references must be (Role, Modality.TEXT) tuples.")
    ref = cast(tuple[object, ...], value)
    if len(ref) != 2 or not isinstance(ref[0], Role) or ref[1] is not Modality.TEXT:
        raise TypeError("text references must be (Role, Modality.TEXT) tuples.")
    return cast(TextRef, ref)


def _audio_ref(value: object) -> tuple[Role, Modality]:
    if not isinstance(value, tuple):
        raise TypeError("audio references must be (Role, Modality.AUDIO) tuples.")
    ref = cast(tuple[object, ...], value)
    if len(ref) != 2 or not isinstance(ref[0], Role) or ref[1] is not Modality.AUDIO:
        raise TypeError("audio references must be (Role, Modality.AUDIO) tuples.")
    return cast(tuple[Role, Modality], ref)


def _index(value: object, length: int) -> None:
    if _integer(value, "index") >= length:
        raise IndexError("grouped sample index out of range.")


def _selection_index(value: object, length: int) -> int:
    index = _integer(value, "text")
    if index >= length:
        raise IndexError("text index out of range.")
    return index


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _text_item(value: object, ref: TextRef) -> TextItem:
    if not isinstance(value, TextItem):
        raise TypeError(f"{ref!r} must contain a TextItem.")
    return value


def _source_text(item: TextItem) -> TextItem:
    views = {
        view: value
        for view, value in cast(Mapping[TextView, object], item.views).items()
        if view is not TextView.SPEAKERS
    }
    meta = {
        key: value
        for key, value in cast(Mapping[TextMeta, object], item.meta).items()
        if key is not TextMeta.SOURCE_INDEX
    }
    return TextItem(views=views, meta=meta)


def _text(item: TextItem) -> str:
    value = item.views.get(TextView.TEXT)
    if not isinstance(value, str):
        raise TypeError("TextView.TEXT must be a string.")
    return value


def _audio_item(value: object, ref: tuple[Role, Modality]) -> AudioItem:
    if not isinstance(value, AudioItem):
        raise TypeError(f"{ref!r} must contain an AudioItem.")
    return value


def _waveform(value: object) -> tuple[torch.Tensor, int]:
    if not isinstance(value, tuple):
        raise TypeError("AudioView.WAVEFORM must be a (waveform, sample_rate) tuple.")
    pair = cast(tuple[object, ...], value)
    if len(pair) != 2:
        raise TypeError("AudioView.WAVEFORM must be a (waveform, sample_rate) tuple.")
    waveform, sample_rate = pair
    if not isinstance(waveform, torch.Tensor):
        raise TypeError("AudioView.WAVEFORM waveform must be a Tensor.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("AudioView.WAVEFORM sample rate must be an integer.")
    if sample_rate <= 0:
        raise ValueError("AudioView.WAVEFORM sample rate must be positive.")
    if waveform.ndim != 2:
        raise ValueError("AudioView.WAVEFORM waveform must have shape [channel, time].")
    return waveform, sample_rate


def _stack_waveforms(waveforms: Sequence[torch.Tensor]) -> torch.Tensor:
    if not waveforms:
        raise ValueError("speaker waveform list must not be empty.")
    prefix_shape = tuple(waveforms[0].shape[:-1])
    lengths: list[int] = []
    for offset, waveform in enumerate(waveforms):
        if tuple(waveform.shape[:-1]) != prefix_shape:
            raise ValueError(
                f"speaker waveform {offset} has shape {tuple(waveform.shape)}; expected prefix shape {prefix_shape}."
            )
        lengths.append(int(waveform.shape[-1]))
    max_length = max(lengths)
    padded = [
        F.pad(waveform, (0, max_length - length))
        for waveform, length in zip(waveforms, lengths)
    ]
    return torch.stack(padded, dim=0)


def _speaker_lengths(lengths: Sequence[int]) -> torch.Tensor:
    return torch.as_tensor(lengths, dtype=torch.int64)
