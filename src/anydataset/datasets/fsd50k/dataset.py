from __future__ import annotations

import json
import os
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch

from anydataset.api.spec import DatasetSpec
from anydataset.datasets.base import DatasetAdapter

if TYPE_CHECKING:
    from anydataset.api.cache import CacheManifest
    from anydataset.datasets.task_adapters import TaskAdapterRegistry


_VALID_SPLITS = frozenset({"dev", "eval"})


def fsd50k_spec(split: str = "dev") -> DatasetSpec:
    return DatasetSpec(
        source="huggingface_audio_files",
        path="Fhrozen/FSD50k",
        name="fsd50k",
        split=split,
        adapter=FSD50KDataset(),
    )


def register_task_adapters(registry: "TaskAdapterRegistry") -> None:
    from anydataset.tasks import Task

    from .adapters.audio_codec import FSD50KAudioCodecAdapter

    registry.register("fsd50k", Task.AUDIO_CODEC, lambda spec: FSD50KAudioCodecAdapter())


class FSD50KDataset(DatasetAdapter):
    def prepare(self, spec: DatasetSpec, cache: CacheManifest) -> dict[str, Any]:
        split = spec.split or "dev"
        if split not in _VALID_SPLITS:
            raise ValueError("FSD50K split must be `dev` or `eval`.")

        manifest_path = cache.cache_path / f"{split}_files.json"
        if not manifest_path.exists():
            files = _list_fsd50k_files(spec.path, split)
            _write_json(manifest_path, files)
        else:
            files = json.loads(manifest_path.read_text(encoding="utf-8"))

        return {
            "repo_id": spec.path,
            "split": split,
            "files": files,
            "cache_path": cache.cache_path,
        }

    def iter_samples(self, manifest: dict[str, Any]) -> Iterator[dict]:
        yield from self._iter_files(manifest, range(len(manifest["files"])))

    def iter_indexed_samples(
        self,
        manifest: Any,
        num_shards: int = 1,
        shard_id: int = 0,
    ) -> Iterator[tuple[int, dict]]:
        indices = range(shard_id, len(manifest["files"]), num_shards)
        for index, row in zip(indices, self._iter_files(manifest, indices)):
            yield index, row

    def _iter_files(
        self,
        manifest: dict[str, Any],
        indices,
    ) -> Iterator[dict]:
        files = manifest["files"]
        for index in indices:
            file_name = files[index]
            local_path = _download_file(manifest, file_name)
            waveform, sample_rate = _load_audio(local_path)
            yield {
                "audio": {
                    "array": waveform,
                    "sampling_rate": sample_rate,
                },
                "path": file_name,
            }


def _list_fsd50k_files(repo_id: str, split: str) -> list[str]:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = (
        f"{endpoint}/api/datasets/{repo_id}/tree/main/clips/{split}"
        "?recursive=true&expand=false&limit=1000"
    )
    files: list[str] = []
    while url:
        request = Request(url, headers={"User-Agent": "anydataset"})
        with urlopen(request, timeout=60) as response:
            rows = json.loads(response.read().decode("utf-8"))
            for row in rows:
                path = row.get("path")
                if path and row.get("type") == "file" and path.endswith(".wav"):
                    files.append(path)
            next_url = response.headers.get("Link")
        url = _next_link_url(next_url, endpoint)
    return sorted(files)


def _next_link_url(link_header: str | None, endpoint: str) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' not in part:
            continue
        start = part.find("<")
        end = part.find(">")
        if start == -1 or end == -1:
            return None
        return _rewrite_endpoint(part[start + 1 : end], endpoint)
    return None


def _rewrite_endpoint(url: str, endpoint: str) -> str:
    target = urlsplit(url)
    replacement = urlsplit(endpoint)
    return urlunsplit(
        (
            replacement.scheme,
            replacement.netloc,
            target.path,
            target.query,
            target.fragment,
        )
    )


def _download_file(manifest: dict[str, Any], file_name: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "FSD50K support requires `pip install anydataset[huggingface]`."
        ) from exc

    return hf_hub_download(
        repo_id=manifest["repo_id"],
        repo_type="dataset",
        filename=file_name,
        cache_dir=str(Path(manifest["cache_path"]) / "hf"),
    )


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio
    except (ImportError, OSError):
        return _load_pcm_wave(path)

    waveform, sample_rate = torchaudio.load(str(path))
    return waveform.to(dtype=torch.float32), int(sample_rate)


def _load_pcm_wave(path: str | Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
        raw = handle.readframes(frames)

    return _pcm_bytes_to_waveform(raw, channels, sample_width), int(sample_rate)


def _pcm_bytes_to_waveform(
    raw: bytes,
    channels: int,
    sample_width: int,
) -> torch.Tensor:
    if channels <= 0:
        raise ValueError("WAV files must contain at least one channel.")
    if sample_width == 1:
        values = torch.tensor(list(raw), dtype=torch.float32).sub(128).div(128)
    elif sample_width == 2:
        values = torch.frombuffer(raw, dtype=torch.int16).clone().float().div(32768)
    elif sample_width == 3:
        values = _int24_pcm_to_float(raw)
    elif sample_width == 4:
        values = torch.frombuffer(raw, dtype=torch.int32).clone().float().div(2147483648)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}.")

    return values.reshape(-1, channels).transpose(0, 1).contiguous()


def _int24_pcm_to_float(raw: bytes) -> torch.Tensor:
    bytes_tensor = torch.tensor(list(raw), dtype=torch.int32).reshape(-1, 3)
    values = (
        bytes_tensor[:, 0]
        | (bytes_tensor[:, 1] << 8)
        | (bytes_tensor[:, 2] << 16)
    )
    sign_bit = 1 << 23
    values = (values ^ sign_bit) - sign_bit
    return values.to(dtype=torch.float32).div(8388608)


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)
