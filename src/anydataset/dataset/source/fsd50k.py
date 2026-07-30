from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import torch

from ..._runtime.sharding import validate_shard
from ...cache import FileLock
from ...types import Spec
from .protocol import validate_load_options


FSD50K_SOURCE = "fsd50k"

_VALID_SPLITS = frozenset({"dev", "eval"})
_HUB_PAGE_LIMIT = 1000
_MANIFEST_SCHEMA_VERSION = 1


class FSD50KSource:
    """Physical FSD50K source backed by Hugging Face Hub audio files."""

    def prepare(self, spec: Spec, cache_path: Path) -> FSD50KDataset:
        validate_load_options(spec, {"revision"}, source="FSD50K")
        split = spec.split or "dev"
        if split not in _VALID_SPLITS:
            raise ValueError("FSD50K split must be 'dev' or 'eval'.")
        revision = spec.load_options.get("revision", "main")
        if not isinstance(revision, str) or not revision:
            raise ValueError("FSD50K revision must be a non-empty string.")

        dataset = FSD50KDataset(
            repo_id=spec.path,
            split=split,
            revision=revision,
            cache_path=cache_path,
        )
        dataset.prepare()
        return dataset

    def iter_indexed_shard(
        self,
        dataset: FSD50KDataset,
        *,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Mapping[str, Any]]]:
        yield from dataset.iter_indexed_shard(num_shards, shard_id)


class FSD50KDataset:
    def __init__(
        self,
        *,
        repo_id: str,
        split: str,
        revision: str,
        cache_path: Path,
    ) -> None:
        self.repo_id = repo_id
        self.split = split
        self.revision = revision
        self.cache_path = cache_path
        self._files: tuple[str, ...] | None = None

    @property
    def files(self) -> tuple[str, ...]:
        if self._files is None:
            self.prepare()
        if self._files is None:
            raise RuntimeError("FSD50K file manifest was not prepared.")
        return self._files

    def prepare(self) -> None:
        manifest_path = self.cache_path / f"{self.split}_files.json"
        with FileLock(self.cache_path / ".prepare.lock", wait_timeout=3600.0):
            files: list[str] | None = None
            if manifest_path.exists():
                files = _read_manifest(manifest_path, self.split)
            if files is None:
                files = _manifest_files(
                    _list_files(self.repo_id, self.split, self.revision),
                    self.split,
                )
                _write_manifest(manifest_path, files)
        self._files = tuple(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("FSD50KDataset index out of range.")
        return _row_for(self, index)

    def iter_indexed_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Mapping[str, Any]]]:
        validate_shard(num_shards, shard_id)
        for index in range(shard_id, len(self), num_shards):
            yield index, self[index]


def _row_for(dataset: FSD50KDataset, index: int) -> dict[str, Any]:
    file_name = dataset.files[index]
    local_path = _download_file(dataset, file_name)
    waveform, sample_rate = _load_audio(local_path)
    return {
        "audio": {
            "array": waveform,
            "sampling_rate": sample_rate,
        },
        "path": file_name,
        "audio_path": str(local_path),
    }


def _list_files(repo_id: str, split: str, revision: str) -> list[str]:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = (
        f"{endpoint}/api/datasets/{repo_id}/tree/{quote(revision, safe='')}/clips/{split}"
        f"?recursive=true&expand=false&limit={_HUB_PAGE_LIMIT}"
    )
    files: list[str] = []
    seen_urls: set[str] = set()
    while url:
        if url in seen_urls:
            raise RuntimeError("FSD50K Hub pagination returned a repeated URL.")
        seen_urls.add(url)
        request = Request(url, headers={"User-Agent": "anydataset"})
        with urlopen(request, timeout=60) as response:
            rows = json.loads(response.read().decode("utf-8"))
            if not isinstance(rows, list) or any(
                not isinstance(row, Mapping) for row in rows
            ):
                raise ValueError("FSD50K Hub tree response must be a list of mappings.")
            for row in rows:
                path = row.get("path")
                if row.get("type") != "file":
                    continue
                if not isinstance(path, str):
                    raise ValueError("FSD50K Hub file entries must contain a path.")
                if path.endswith(".wav"):
                    files.append(path)
            next_url = response.headers.get("Link")
        url = _next_link_url(next_url, endpoint)
        if len(rows) >= _HUB_PAGE_LIMIT and url is None:
            raise RuntimeError(
                "FSD50K Hub listing truncated: full page without next Link."
            )
    return sorted(files)


def _manifest_files(value: object, split: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("FSD50K file manifest must be a non-empty list.")
    prefix = f"clips/{split}/"
    files = []
    for file_name in value:
        if (
            not isinstance(file_name, str)
            or not file_name.startswith(prefix)
            or not file_name.endswith(".wav")
        ):
            raise ValueError(
                f"FSD50K file manifest entries must be WAV paths under {prefix}."
            )
        files.append(file_name)
    if len(set(files)) != len(files):
        raise ValueError("FSD50K file manifest entries must be unique.")
    return files


def _read_manifest(path: Path, split: str) -> list[str] | None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    if (
        raw.get("schema_version") != _MANIFEST_SCHEMA_VERSION
        or raw.get("listed_complete") is not True
    ):
        return None
    return _manifest_files(raw.get("files"), split)


def _write_manifest(path: Path, files: list[str]) -> None:
    _write_json(
        path,
        {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "listed_complete": True,
            "files": files,
        },
    )


def _next_link_url(link_header: str | None, endpoint: str) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' not in part:
            continue
        start = part.find("<")
        end = part.find(">")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("FSD50K Hub Link header next URL is malformed.")
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


def _download_file(dataset: FSD50KDataset, file_name: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "FSD50K support requires 'pip install anydataset[huggingface]'."
        ) from exc

    return hf_hub_download(
        repo_id=dataset.repo_id,
        repo_type="dataset",
        revision=dataset.revision,
        filename=file_name,
        cache_dir=str(dataset.cache_path / "hf"),
    )


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio
    except (ImportError, OSError) as exc:
        raise ImportError(
            "FSD50K audio loading requires a working torchaudio installation "
            "('pip install anydataset[audio]')."
        ) from exc

    waveform, sample_rate = torchaudio.load(str(path))
    return waveform.to(dtype=torch.float32), int(sample_rate)


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)
