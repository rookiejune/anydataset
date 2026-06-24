from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from ..modalities import ModalityKey
from ..modalities.audio import AudioKey, AudioOptKey, AudioView
from ..modalities.text import TextKey, TextOptKey
from .base import DatasetAdapter, MissingModalityError, ModalityRole

if TYPE_CHECKING:
    from ..api.cache import CacheManifest
    from ..api.spec import DatasetSpec


@dataclass(frozen=True)
class LocalFilesAdapter(DatasetAdapter):
    waveform_field: str = AudioView.WAVEFORM
    sample_rate_field: str = AudioKey.SAMPLE_RATE
    text_field: str | None = None
    lang_field: str | None = None
    audio_field: str | None = None
    audio_waveform_field: str = "array"
    audio_sample_rate_field: str = "sampling_rate"
    duration_field: str | None = None
    label_field: str | None = None
    labels_field: str | None = None
    file_field: str | None = None
    lang_value: str | None = None
    labels_fields: Mapping[str, str] = field(default_factory=dict)
    text_fields: Mapping[ModalityRole, str] = field(default_factory=dict)
    lang_fields: Mapping[ModalityRole, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.text_field is not None and None in self.text_fields:
            raise ValueError("Use either text_field or text_fields[None], not both.")
        if self.lang_field is not None and None in self.lang_fields:
            raise ValueError("Use either lang_field or lang_fields[None], not both.")
        if self.lang_value is not None and (
            self.lang_field is not None or None in self.lang_fields
        ):
            raise ValueError("Use either lang_value or lang field mappings, not both.")
        if self.labels_field is not None and self.labels_fields:
            raise ValueError("Use either labels_field or labels_fields, not both.")

    def prepare(self, spec: "DatasetSpec", cache: "CacheManifest") -> dict[str, Any]:
        return {
            "path": Path(spec.path).expanduser(),
            "split": spec.split,
            "cache_path": cache.cache_path,
        }

    def iter_samples(self, manifest: dict[str, Any]) -> Iterator[dict]:
        path = manifest["path"]
        if not path.exists():
            raise FileNotFoundError(path)

        if path.is_file():
            yield from self._iter_file(path)
            return

        for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
            yield {"path": str(file_path)}

    def _iter_file(self, path: Path) -> Iterator[dict]:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return

        yield {"path": str(path)}

    def audio(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        if role is not None:
            raise MissingModalityError(ModalityKey.AUDIO, role)

        waveform, sample_rate = self._extract_waveform(row)
        views = {
            AudioView.WAVEFORM: waveform,
        }
        if self.file_field is not None:
            views[AudioView.FILE] = row[self.file_field]

        data: dict[str, Any] = {
            AudioKey.SAMPLE_RATE: sample_rate,
            AudioKey.VIEWS: views,
        }
        if self.duration_field is not None:
            data[AudioOptKey.DURATION] = row[self.duration_field]
        if self.label_field is not None:
            data[AudioOptKey.LABEL] = row[self.label_field]
        if self.labels_field is not None:
            data[AudioOptKey.LABELS] = row[self.labels_field]
        if self.labels_fields:
            data[AudioOptKey.LABELS] = {
                name: row[field] for name, field in self.labels_fields.items()
            }
        return data

    def text(self, row: Mapping[str, Any], role: ModalityRole = None) -> Mapping[str, Any]:
        text_field = _field_for(role, self.text_field, self.text_fields, ModalityKey.TEXT)
        lang_field = _optional_field_for(role, self.lang_field, self.lang_fields)

        data: dict[str, Any] = {
            TextKey.CONTENT: row[text_field],
        }
        if self.lang_value is not None:
            data[TextOptKey.LANG] = self.lang_value
        elif lang_field is not None:
            data[TextOptKey.LANG] = row[lang_field]
        return data

    def _extract_waveform(self, row: Mapping[str, Any]) -> tuple[Any, int | None]:
        if self.audio_field is None:
            return row[self.waveform_field], _maybe_int(row.get(self.sample_rate_field))

        audio = row[self.audio_field]
        if isinstance(audio, Mapping):
            return (
                audio[self.audio_waveform_field],
                _maybe_int(audio.get(self.audio_sample_rate_field)),
            )
        decoded = _maybe_decode_audio(audio)
        if decoded is not None:
            return decoded
        return audio, _maybe_int(row.get(self.sample_rate_field))


def _field_for(
    role: ModalityRole,
    default_field: str | None,
    fields: Mapping[ModalityRole, str],
    modality: str,
) -> str:
    if role is None and default_field is not None:
        return default_field
    if role in fields:
        return fields[role]
    raise MissingModalityError(modality, role)


def _optional_field_for(
    role: ModalityRole,
    default_field: str | None,
    fields: Mapping[ModalityRole, str],
) -> str | None:
    if role is None and default_field is not None:
        return default_field
    return fields.get(role)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _maybe_decode_audio(audio: Any) -> tuple[Any, int] | None:
    get_all_samples = getattr(audio, "get_all_samples", None)
    if get_all_samples is None:
        return None

    samples = get_all_samples()
    data = getattr(samples, "data")
    sample_rate = getattr(samples, "sample_rate")
    return data, int(sample_rate)
