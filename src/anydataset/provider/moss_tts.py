from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from ..dataset.collate import Batch, FieldGroup, FieldRef
from ..types.item import Modality, Role
from ..types.item import AudioView, TextView

try:
    import torchaudio
except ImportError as exc:
    raise ImportError(
        "MossTTSProvider requires `pip install anydataset[audio]`."
    ) from exc


class MossTTSProvider:
    output = AudioView.WAVEFORM

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        options: Any | None = None,
        reference_role: Role | None = None,
        max_reference_files: int = 1024,
        runtime_kwargs: Mapping[str, object] | None = None,
        **load_options: Any,
    ) -> None:
        try:
            from anytrain.tts.moss import MossTTS
        except ImportError as exc:
            raise ImportError(
                "MossTTSProvider requires `anytrain[moss-tts]`."
            ) from exc
        kwargs = {"runtime_kwargs": runtime_kwargs, **load_options}
        if model is None:
            self.tts = MossTTS.from_pretrained(**kwargs)
        else:
            self.tts = MossTTS.from_pretrained(model, **kwargs)
        self.options = options
        self.reference_role = reference_role
        self.max_reference_files = _positive_int(
            "max_reference_files",
            max_reference_files,
        )
        self._tempdir: TemporaryDirectory[str] | None = None
        self._reference_file_count = 0

    def __call__(self, views: Mapping[Any, Any]) -> Any:
        text = views[TextView.TEXT]
        if not isinstance(text, str):
            raise TypeError("MossTTSProvider expects a string TextView.TEXT input.")
        output = self.tts.synthesize(
            text,
            self.options,
            reference_audio_path=self._reference_path(views),
        )
        return _audio_output(output)

    def call_batch(
        self,
        batch: Batch,
    ) -> Sequence[Any] | Mapping[tuple[Role, Modality], Sequence[Any]]:
        refs = _text_refs(batch)
        outputs = {ref: self._synthesize_ref_batch(batch, ref) for ref in refs}
        if len(refs) == 1:
            return outputs[refs[0]]
        return outputs

    def _synthesize_ref_batch(
        self,
        batch: Batch,
        ref: tuple[Role, Modality],
    ) -> Sequence[Any]:
        texts = _text_batch(batch.sample[ref].views[TextView.TEXT])
        outputs = self.tts.synthesize(
            texts,
            self.options,
            reference_audio_paths=self._reference_paths(batch, len(texts)),
        )
        if not isinstance(outputs, Sequence):
            raise TypeError("MossTTS batch synthesize output must be a sequence.")
        return [_audio_output(output) for output in outputs]

    def _reference_path(self, views: Mapping[Any, Any]) -> str | None:
        if self.reference_role is None:
            return None
        if AudioView.FILE in views:
            return self._file_path(views[AudioView.FILE])
        if AudioView.WAVEFORM in views:
            waveform, sample_rate = views[AudioView.WAVEFORM]
            return self._waveform_path(torch.as_tensor(waveform), int(sample_rate))
        raise ValueError(
            "MossTTSProvider reference input requires AudioView.FILE or "
            "AudioView.WAVEFORM."
        )

    def _reference_paths(self, batch: Batch, count: int) -> Sequence[str] | None:
        if self.reference_role is None:
            return None
        ref = (self.reference_role, Modality.AUDIO)
        views = batch.sample[ref].views
        if AudioView.FILE in views:
            values = _file_batch(views[AudioView.FILE])
            if len(values) != count:
                raise ValueError("reference file batch size must match text batch size.")
            self._prepare_reference_files(sum(isinstance(value, bytes) for value in values))
            paths = [self._file_path(value) for value in values]
            if len(paths) != count:
                raise ValueError("reference file batch size must match text batch size.")
            return paths
        if AudioView.WAVEFORM in views:
            waveform, sample_rates = views[AudioView.WAVEFORM]
            lengths = batch.lengths(FieldRef(ref, FieldGroup.VIEWS, AudioView.WAVEFORM))
            if waveform.shape[0] != count:
                raise ValueError("reference audio batch size must match text batch size.")
            self._prepare_reference_files(count)
            return [
                self._waveform_path(
                    waveform[index, :, : int(length.item())],
                    int(sample_rates[index].item()),
                )
                for index, length in enumerate(lengths)
            ]
        raise ValueError(
            "MossTTSProvider reference batch requires AudioView.FILE or "
            "AudioView.WAVEFORM."
        )

    def _file_path(self, value: Any) -> str:
        if isinstance(value, (str, Path)):
            return str(Path(value).expanduser())
        if isinstance(value, bytes):
            return self._bytes_path(value)
        raise TypeError("reference file view must be a path or bytes.")

    def _bytes_path(self, value: bytes) -> str:
        self._prepare_reference_file()
        path = self._reference_dir / f"ref-{self._reference_file_count:08d}.wav"
        path.write_bytes(value)
        self._reference_file_count += 1
        return str(path)

    def _waveform_path(self, waveform: torch.Tensor, sample_rate: int) -> str:
        self._prepare_reference_file()
        path = self._reference_dir / f"ref-{self._reference_file_count:08d}.wav"
        torchaudio.save(str(path), waveform.detach().cpu(), sample_rate)
        self._reference_file_count += 1
        return str(path)

    def _prepare_reference_file(self) -> None:
        if self._tempdir is None or self._reference_file_count >= self.max_reference_files:
            self._reset_reference_dir()

    def _prepare_reference_files(self, count: int) -> None:
        if count <= 0:
            return
        if count > self.max_reference_files:
            raise ValueError(
                "reference batch size must not exceed max_reference_files."
            )
        if (
            self._tempdir is None
            or self._reference_file_count + count > self.max_reference_files
        ):
            self._reset_reference_dir()

    def _reset_reference_dir(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
        self._tempdir = TemporaryDirectory(prefix="anydataset-moss-ref-")
        self._reference_file_count = 0

    @property
    def _reference_dir(self) -> Path:
        if self._tempdir is None:
            self._reset_reference_dir()
        if self._tempdir is None:
            raise RuntimeError("failed to create reference audio tempdir.")
        return Path(self._tempdir.name)


def _text_refs(batch: Batch) -> tuple[tuple[Role, Modality], ...]:
    refs = tuple(
        ref
        for ref in batch.sample
        if ref[1] is Modality.TEXT
        and TextView.TEXT in batch.sample[ref].views
    )
    if not refs:
        raise ValueError("MossTTSProvider.call_batch expects at least one text input.")
    return refs


def _text_batch(value: Any) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("MossTTSProvider.call_batch expects a text sequence.")
    texts = list(value)
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("MossTTSProvider.call_batch expects string text inputs.")
    return texts


def _file_batch(value: Any) -> list[Any]:
    if isinstance(value, (str, Path, bytes)):
        return [value]
    if not isinstance(value, Sequence):
        raise TypeError("batched reference file view must be a sequence.")
    return list(value)


def _audio_output(output: Any) -> tuple[Any, int]:
    return output.waveform, output.sample_rate


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value
