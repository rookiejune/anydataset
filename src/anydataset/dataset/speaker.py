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
_FRAME_CODEC_VIEWS = frozenset(
    {
        AudioView.LONGCAT,
        AudioView.DAC,
        AudioView.STABLE,
        AudioView.UNICODEC,
    }
)


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


@dataclass(frozen=True)
class SpeakerAudioRow:
    """Identify one text row in a speaker audio grid."""

    source_index: int
    role: Role

    def __post_init__(self) -> None:
        _integer(self.source_index, "source_index")
        if not isinstance(self.role, Role):
            raise TypeError("role must be a Role.")


class GroupedSpeakerAudioDataset:
    """Group a speaker-cartesian audio dataset back onto its text axis."""

    def __init__(
        self,
        dataset: MapDataset,
        speaker_ids: Sequence[str],
        text_ref: TextRef = (Role.DEFAULT, Modality.TEXT),
        audio_ref: tuple[Role, Modality] = (Role.DEFAULT, Modality.AUDIO),
        *,
        row_specs: Sequence[SpeakerAudioRow] | None = None,
    ) -> None:
        self._dataset: MapDataset = dataset
        self._speaker_ids: tuple[str, ...] = _speaker_ids(speaker_ids)
        self._text_ref: TextRef = _text_ref(text_ref)
        self._audio_ref: tuple[Role, Modality] = _audio_ref(audio_ref)
        if len(dataset) % len(self.speaker_ids) != 0:
            raise ValueError("flat speaker dataset length must be divisible by speaker count.")
        self._row_specs: tuple[SpeakerAudioRow, ...] = _speaker_audio_rows(
            row_specs,
            row_count=len(dataset) // len(self.speaker_ids),
            default_role=self.text_ref[0],
        )

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

    @property
    def row_specs(self) -> tuple[SpeakerAudioRow, ...]:
        return self._row_specs

    def __len__(self) -> int:
        return len(self.row_specs)

    def __getitem__(self, index: int) -> Sample:
        return self.load(index)

    def load(self, index: int, *, view: AudioView | None = None) -> Sample:
        """Load one grouped text row, optionally selecting an audio view."""

        _index(index, len(self))
        row = self.row_specs[index]
        start = index * len(self.speaker_ids)
        first_sample = self.dataset[start]
        first_text = _text_item(first_sample.get(self.text_ref), self.text_ref)
        source_text = _source_text(first_text)
        grouped: Sample = dict(first_sample)
        grouped[self.text_ref] = TextItem(
            views=source_text.views,
            meta={
                **source_text.meta,
                TextMeta.SOURCE_INDEX: row.source_index,
            },
        )

        audio_items: list[AudioItem] = []
        for offset, speaker_id in enumerate(self.speaker_ids):
            sample = first_sample if offset == 0 else self.dataset[start + offset]
            text_item = _text_item(sample.get(self.text_ref), self.text_ref)
            if _source_text(text_item) != source_text:
                raise ValueError(
                    f"flat sample {start + offset} text content differs within text row {index}."
                )
            source_index = text_item.meta.get(TextMeta.SOURCE_INDEX)
            if source_index != row.source_index:
                raise ValueError(
                    f"flat sample {start + offset} has source index {source_index!r}; expected {row.source_index!r} for text row {index}."
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
            audio_items.append(audio_item)

        resolved_view = _selected_audio_view(audio_items, view)
        value, lengths, sample_rate = _collate_audio_view(audio_items, resolved_view)
        views: dict[AudioView, object] = {
            resolved_view: _audio_value(value, sample_rate),
            AudioView.SPEAKERS: self.speaker_ids,
            AudioView.SPEAKER_LENGTHS: lengths,
        }
        meta = (
            {AudioMeta.DURATION: int(lengths.max().item()) / sample_rate}
            if sample_rate is not None
            else {}
        )
        grouped[self.audio_ref] = AudioItem(
            views=views,
            meta=meta,
        )
        return grouped


@dataclass(frozen=True)
class SpeakerAudioBlock:
    """One materialized rectangular selection from a speaker audio-view grid."""

    text_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    roles: tuple[Role, ...]
    texts: tuple[str, ...]
    speaker_ids: tuple[str, ...]
    audio_view: AudioView
    audio: AudioItem

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
        if not isinstance(self.source_indices, tuple) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.source_indices
        ):
            raise TypeError("source_indices must be a tuple of non-negative integers.")
        if not isinstance(self.roles, tuple) or any(
            not isinstance(role, Role) for role in self.roles
        ):
            raise TypeError("roles must be a tuple of Role values.")
        text_count = len(self.text_indices)
        if any(
            len(values) != text_count
            for values in (self.source_indices, self.roles, self.texts)
        ):
            raise ValueError(
                "text_indices, source_indices, roles, and texts must have the same length."
            )
        if not isinstance(self.speaker_ids, tuple):
            raise TypeError("speaker_ids must be a tuple of strings.")
        _unique_speaker_ids(self.speaker_ids)
        _audio_view(self.audio_view)
        if not isinstance(self.audio, AudioItem):
            raise TypeError("audio must be an AudioItem.")
        if self.audio_view not in self.audio.views:
            raise ValueError(f"audio does not contain selected view {self.audio_view!r}.")
        if self.audio.views.get(AudioView.SPEAKERS) != self.speaker_ids:
            raise ValueError("audio speaker axis does not match speaker_ids.")
        shape = (len(self.texts), len(self.speaker_ids))
        lengths = self.audio.views.get(AudioView.SPEAKER_LENGTHS)
        if not isinstance(lengths, torch.Tensor):
            raise TypeError("lengths must be a Tensor.")
        if lengths.dtype is not torch.int64:
            raise TypeError("lengths must have dtype torch.int64.")
        if tuple(lengths.shape) != shape:
            raise ValueError(f"lengths must have shape {shape}.")
        if bool(torch.any(lengths < 0).item()):
            raise ValueError("lengths must be non-negative.")
        _block_value(self.values, view=self.audio_view, shape=shape, lengths=lengths)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.texts), len(self.speaker_ids)

    @property
    def values(self) -> object:
        return self.audio.views[self.audio_view]

    @property
    def lengths(self) -> torch.Tensor:
        value = self.audio.views[AudioView.SPEAKER_LENGTHS]
        return cast(torch.Tensor, value)

    @property
    def waveforms(self) -> torch.Tensor:
        waveform, _ = _waveform_pair(self.values)
        return waveform

    @property
    def sample_rate(self) -> int:
        _, sample_rate = _waveform_pair(self.values)
        return sample_rate


class SpeakerAudioGrid:
    """Expose a flat speaker store as a logical source × text × speaker grid."""

    def __init__(
        self,
        dataset: MapDataset,
        speaker_ids: Sequence[str],
        text_ref: TextRef = (Role.DEFAULT, Modality.TEXT),
        audio_ref: tuple[Role, Modality] = (Role.DEFAULT, Modality.AUDIO),
        *,
        row_specs: Sequence[SpeakerAudioRow] | None = None,
    ) -> None:
        self._cells: MapDataset = dataset
        self._speaker_ids: tuple[str, ...] = _unique_speaker_ids(speaker_ids)
        self._rows = GroupedSpeakerAudioDataset(
            dataset,
            self.speaker_ids,
            text_ref=text_ref,
            audio_ref=audio_ref,
            row_specs=row_specs,
        )
        self._source_indices, self._text_roles, self._row_indices = _speaker_audio_axes(
            self.row_specs
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
    def row_specs(self) -> tuple[SpeakerAudioRow, ...]:
        return self.rows.row_specs

    @property
    def source_indices(self) -> tuple[int, ...]:
        return self._source_indices

    @property
    def text_roles(self) -> tuple[Role, ...]:
        return self._text_roles

    @property
    def text_ref(self) -> TextRef:
        return self.rows.text_ref

    @property
    def audio_ref(self) -> tuple[Role, Modality]:
        return self.rows.audio_ref

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.source_indices), len(self.text_roles), len(self.speaker_ids)

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int) -> SpeakerAudioBlock:
        source_index = self.source_indices[
            _axis_index(index, len(self), name="source position")
        ]
        return self.select(source=source_index).load()

    def select(
        self,
        *,
        source: int | None = None,
        text: int | Role | str | None = None,
        speaker: str | None = None,
    ) -> SpeakerAudioSelection:
        """Select one source sample, text slot, speaker, or their rectangular subset."""

        source_indices = (
            self.source_indices
            if source is None
            else (_source_index(source, self.source_indices),)
        )
        text_roles = self.text_roles if text is None else (self._selected_text(text),)
        text_indices = tuple(
            self._row_indices[(source_index, text_role)]
            for source_index in source_indices
            for text_role in text_roles
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

    def _selected_text(self, text: int | Role | str) -> Role:
        if isinstance(text, bool):
            raise TypeError("text must be a text slot index or role label.")
        if isinstance(text, int):
            return self.text_roles[
                _axis_index(text, len(self.text_roles), name="text position")
            ]
        role = _text_role(text)
        if role not in self.text_roles:
            raise ValueError(
                f"text role metadata {role.value!r} is not present in the grid."
            )
        return role

    def _load(
        self,
        text_indices: tuple[int, ...],
        speaker_indices: tuple[int, ...],
        view: AudioView | None,
    ) -> SpeakerAudioBlock:
        if not text_indices or not speaker_indices:
            raise ValueError("speaker audio selection must not be empty.")
        texts: list[str] = []
        audio_items: list[AudioItem] = []
        for text_index in text_indices:
            row = self.row_specs[text_index]
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
                        f"flat sample {flat_index} text content differs within text row {text_index}."
                    )
                source_index = text_item.meta.get(TextMeta.SOURCE_INDEX)
                if source_index != row.source_index:
                    raise ValueError(
                        f"flat sample {flat_index} has source index {source_index!r}; expected {row.source_index!r} for text row {text_index}."
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
                audio_items.append(audio_item)
            if source_text is None:
                raise ValueError("speaker audio selection must include a speaker.")
            texts.append(_text(source_text))
        resolved_view = _selected_audio_view(audio_items, view)
        value, lengths, sample_rate = _collate_audio_view(audio_items, resolved_view)
        grid_shape = (len(text_indices), len(speaker_indices))
        selected_speakers = tuple(
            self.speaker_ids[speaker_index] for speaker_index in speaker_indices
        )
        audio = AudioItem(
            views={
                resolved_view: _audio_value(
                    _reshape_audio_value(value, grid_shape, resolved_view),
                    sample_rate,
                ),
                AudioView.SPEAKERS: selected_speakers,
                AudioView.SPEAKER_LENGTHS: lengths.reshape(grid_shape),
            },
            meta=(
                {AudioMeta.DURATION: int(lengths.max().item()) / sample_rate}
                if sample_rate is not None
                else {}
            ),
        )
        return SpeakerAudioBlock(
            text_indices=text_indices,
            source_indices=tuple(
                self.row_specs[index].source_index for index in text_indices
            ),
            roles=tuple(self.row_specs[index].role for index in text_indices),
            texts=tuple(texts),
            speaker_ids=selected_speakers,
            audio_view=resolved_view,
            audio=audio,
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
            _axis_index(text_index, len(grid.row_specs), name="text row index")
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
    def source_indices(self) -> tuple[int, ...]:
        return tuple(
            self.grid.row_specs[index].source_index for index in self.text_indices
        )

    @property
    def roles(self) -> tuple[Role, ...]:
        return tuple(self.grid.row_specs[index].role for index in self.text_indices)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.text_indices), len(self.speaker_ids)

    def load(self, *, view: AudioView | None = None) -> SpeakerAudioBlock:
        """Read and pad only the selected rectangle."""

        return self.grid._load(self.text_indices, self._speaker_indices, view)


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


def _speaker_audio_rows(
    value: object,
    *,
    row_count: int,
    default_role: Role,
) -> tuple[SpeakerAudioRow, ...]:
    if value is None:
        return tuple(
            SpeakerAudioRow(source_index=index, role=default_role)
            for index in range(row_count)
        )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("row_specs must be a sequence of SpeakerAudioRow values.")
    rows = tuple(value)
    if any(not isinstance(row, SpeakerAudioRow) for row in rows):
        raise TypeError("row_specs must contain only SpeakerAudioRow values.")
    if len(rows) != row_count:
        raise ValueError(
            f"row_specs contains {len(rows)} rows, expected {row_count} from the flat dataset."
        )
    return cast(tuple[SpeakerAudioRow, ...], rows)


def _speaker_audio_axes(
    rows: tuple[SpeakerAudioRow, ...]
) -> tuple[tuple[int, ...], tuple[Role, ...], dict[tuple[int, Role], int]]:
    if not rows:
        return (), (), {}
    first_source = rows[0].source_index
    text_roles: list[Role] = []
    for row in rows:
        if row.source_index != first_source:
            break
        if row.role in text_roles:
            raise ValueError(
                f"speaker audio rows repeat text role {row.role.value!r} for source {first_source}."
            )
        text_roles.append(row.role)
    text_axis = tuple(text_roles)
    if len(rows) % len(text_axis) != 0:
        raise ValueError("speaker audio row count must be divisible by text count.")

    source_indices: list[int] = []
    seen_sources: set[int] = set()
    row_indices: dict[tuple[int, Role], int] = {}
    for start in range(0, len(rows), len(text_axis)):
        group = rows[start : start + len(text_axis)]
        source_index = group[0].source_index
        if source_index in seen_sources:
            raise ValueError(f"speaker audio rows repeat source index {source_index}.")
        seen_sources.add(source_index)
        if any(row.source_index != source_index for row in group):
            raise ValueError(
                f"speaker audio source block at row {start} mixes source indices."
            )
        actual_roles = tuple(row.role for row in group)
        if actual_roles != text_axis:
            raise ValueError(
                f"speaker audio source {source_index} has text role order {actual_roles!r}; expected {text_axis!r}."
            )
        source_indices.append(source_index)
        for offset, role in enumerate(actual_roles):
            row_indices[(source_index, role)] = start + offset
    return tuple(source_indices), text_axis, row_indices


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


def _audio_view(value: object) -> AudioView:
    if not isinstance(value, AudioView):
        raise TypeError("audio_view must be an AudioView.")
    if value is AudioView.WAVEFORM or value is AudioView.BICODEC:
        return value
    if value in _FRAME_CODEC_VIEWS:
        return value
    raise ValueError("speaker audio grids require a waveform or codec audio view.")


def _selected_audio_view(
    items: Sequence[AudioItem],
    requested: AudioView | None,
) -> AudioView:
    if not items:
        raise ValueError("speaker audio selection must not be empty.")
    if requested is not None:
        view = _audio_view(requested)
        if any(view not in item.views for item in items):
            raise ValueError(f"selected audio view {view!r} is missing from one or more cells.")
        return view

    available = [
        set(item.views).intersection(_FRAME_CODEC_VIEWS | {AudioView.WAVEFORM, AudioView.BICODEC})
        for item in items
    ]
    common = set.intersection(*available)
    if len(common) != 1:
        if not common:
            raise ValueError("selected cells do not share a collatable audio view.")
        raise ValueError(
            "selected cells contain multiple audio views; pass view=AudioView explicitly."
        )
    return _audio_view(next(iter(common)))


def _index(value: object, length: int) -> None:
    if _integer(value, "index") >= length:
        raise IndexError("grouped sample index out of range.")


def _axis_index(value: object, length: int, *, name: str) -> int:
    index = _integer(value, name)
    if index >= length:
        raise IndexError(f"{name} is out of range.")
    return index


def _source_index(value: object, source_indices: tuple[int, ...]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("source must be an integer source index.")
    if value not in source_indices:
        raise ValueError(f"source index {value} is not present in the grid.")
    return value


def _text_role(value: Role | str) -> Role:
    try:
        return Role(value)
    except ValueError as error:
        raise ValueError(
            f"text role metadata {value!r} is not present in the grid."
        ) from error


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
    if value == "":
        raise ValueError("TextView.TEXT must be a non-empty string.")
    return value


def _audio_item(value: object, ref: tuple[Role, Modality]) -> AudioItem:
    if not isinstance(value, AudioItem):
        raise TypeError(f"{ref!r} must contain an AudioItem.")
    return value


def _waveform(value: object) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = _waveform_pair(value)
    if waveform.ndim != 2:
        raise ValueError("AudioView.WAVEFORM waveform must have shape [channel, time].")
    return waveform, sample_rate


def _waveform_pair(value: object) -> tuple[torch.Tensor, int]:
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
    return waveform, sample_rate


def _codec(value: object, view: AudioView) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"AudioView.{view.name} must be a Tensor.")
    if value.ndim != 2:
        raise ValueError(f"AudioView.{view.name} must have shape [unit, codebook].")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"AudioView.{view.name} must not be empty.")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise TypeError(f"AudioView.{view.name} must contain integer ids.")
    return value


def _semantic_acoustic(value: object) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError("AudioView.BICODEC must be a semantic/acoustic mapping.")
    fields = cast(Mapping[object, object], value)
    if set(fields) != {"semantic", "acoustic"}:
        raise ValueError(
            "AudioView.BICODEC must contain exactly 'semantic' and 'acoustic'."
        )
    return (
        _codec(fields["semantic"], AudioView.BICODEC),
        _codec(fields["acoustic"], AudioView.BICODEC),
    )


def _collate_audio_view(
    items: Sequence[AudioItem],
    view: AudioView,
) -> tuple[object, torch.Tensor, int | None]:
    if not items:
        raise ValueError("speaker audio selection must not be empty.")
    if view is AudioView.WAVEFORM:
        pairs = tuple(_waveform(item.views.get(view)) for item in items)
        sample_rate = pairs[0][1]
        if any(current_rate != sample_rate for _, current_rate in pairs[1:]):
            raise ValueError("selected speaker waveforms must share one sample rate.")
        waveforms = tuple(waveform for waveform, _ in pairs)
        lengths = _speaker_lengths(tuple(int(value.shape[-1]) for value in waveforms))
        return _stack_waveforms(waveforms), lengths, sample_rate
    if view in _FRAME_CODEC_VIEWS:
        return (*_stack_unit_sequences(tuple(_codec(item.views.get(view), view) for item in items)), None)
    if view is AudioView.BICODEC:
        values = tuple(_semantic_acoustic(item.views.get(view)) for item in items)
        semantics, lengths = _stack_unit_sequences(
            tuple(semantic for semantic, _ in values)
        )
        acoustics = tuple(acoustic for _, acoustic in values)
        first_shape = tuple(acoustics[0].shape)
        if any(tuple(value.shape) != first_shape for value in acoustics[1:]):
            raise ValueError("AudioView.BICODEC acoustic units must share one fixed shape.")
        return {
            "semantic": semantics,
            "acoustic": torch.stack(acoustics),
        }, lengths, None
    raise ValueError(f"unsupported speaker grid audio view: {view!r}.")


def _audio_value(value: object, sample_rate: int | None) -> object:
    return value if sample_rate is None else (value, sample_rate)


def _reshape_audio_value(
    value: object,
    shape: tuple[int, int],
    view: AudioView,
) -> object:
    if view is AudioView.BICODEC:
        fields = cast(Mapping[str, torch.Tensor], value)
        return {
            name: tensor.reshape(*shape, *tensor.shape[1:])
            for name, tensor in fields.items()
        }
    tensor = cast(torch.Tensor, value)
    return tensor.reshape(*shape, *tensor.shape[1:])


def _block_value(
    value: object,
    *,
    view: AudioView,
    shape: tuple[int, int],
    lengths: torch.Tensor,
) -> None:
    if view is AudioView.WAVEFORM:
        tensor, _ = _waveform_pair(value)
        if tensor.ndim != 4 or tuple(tensor.shape[:2]) != shape:
            raise ValueError(
                "block waveform must have shape [text, speaker, channel, time]."
            )
        limit = tensor.shape[-1]
    elif view in _FRAME_CODEC_VIEWS:
        tensor = cast(torch.Tensor, value)
        if tensor.ndim != 4 or tuple(tensor.shape[:2]) != shape:
            raise ValueError(
                "block codec view must have shape [text, speaker, unit, codebook]."
            )
        limit = tensor.shape[2]
    elif view is AudioView.BICODEC:
        semantic, acoustic = _semantic_acoustic_block(value, shape)
        limit = semantic.shape[2]
        if tuple(acoustic.shape[:2]) != shape:
            raise ValueError("block acoustic units must preserve text and speaker axes.")
    else:
        raise ValueError(f"unsupported speaker grid audio view: {view!r}.")
    if bool(torch.any(lengths > limit).item()):
        raise ValueError("lengths must not exceed the selected audio view unit axis.")


def _semantic_acoustic_block(
    value: object,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError("AudioView.BICODEC block must be a semantic/acoustic mapping.")
    fields = cast(Mapping[object, object], value)
    if set(fields) != {"semantic", "acoustic"}:
        raise ValueError(
            "AudioView.BICODEC block must contain exactly 'semantic' and 'acoustic'."
        )
    semantic = fields["semantic"]
    acoustic = fields["acoustic"]
    if not isinstance(semantic, torch.Tensor) or not isinstance(acoustic, torch.Tensor):
        raise TypeError("AudioView.BICODEC block values must be Tensors.")
    if semantic.ndim != 4 or acoustic.ndim != 4:
        raise ValueError(
            "AudioView.BICODEC block values must have [text, speaker, unit, codebook] shape."
        )
    if tuple(semantic.shape[:2]) != shape:
        raise ValueError("block semantic units must preserve text and speaker axes.")
    return semantic, acoustic


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


def _stack_unit_sequences(
    values: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not values:
        raise ValueError("codec unit list must not be empty.")
    codebooks = values[0].shape[1]
    dtype = values[0].dtype
    device = values[0].device
    if any(value.shape[1] != codebooks for value in values[1:]):
        raise ValueError("codec units must share one codebook axis.")
    if any(value.dtype != dtype for value in values[1:]):
        raise TypeError("codec units must share one dtype.")
    if any(value.device != device for value in values[1:]):
        raise ValueError("codec units must share one device.")
    lengths = _speaker_lengths(tuple(int(value.shape[0]) for value in values))
    max_length = int(lengths.max().item())
    batch = values[0].new_zeros((len(values), max_length, codebooks))
    for index, value in enumerate(values):
        batch[index, : value.shape[0]] = value
    return batch, lengths


def _speaker_lengths(lengths: Sequence[int]) -> torch.Tensor:
    return torch.as_tensor(lengths, dtype=torch.int64)
