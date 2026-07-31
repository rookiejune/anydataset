from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from ..._runtime.sharding import validate_shard
from ...cache import FileLock
from ...types import Spec
from .protocol import _validate_load_options


_VALID_REPO_TYPES = frozenset({"dataset", "model", "space"})
_HUB_PAGE_LIMIT = 1000
_MANIFEST_SCHEMA_VERSION = 1


class HuggingFaceFilesSource:
    """Physical Hugging Face Hub file-tree source.

    The source discovers and downloads files from a Hub repository. It only
    returns physical row facts such as the repo-relative path and local cached
    path; dataset-specific field names and decoding belong in presets/parsers.
    """

    def prepare(self, spec: Spec, cache_path: Path) -> _HuggingFaceFilesDataset:
        _validate_load_options(
            spec,
            {"revision", "repo_type", "path_prefix", "path_template", "suffixes"},
            source="Hugging Face files",
        )
        revision = _revision(spec)
        repo_type = _string_option(
            spec.load_options.get("repo_type", "dataset"),
            "repo_type",
        )
        if repo_type not in _VALID_REPO_TYPES:
            raise ValueError(
                "Hugging Face files repo_type must be 'dataset', 'model', or 'space'."
            )

        path_prefix = _path_prefix(spec)
        suffixes = _suffixes(spec.load_options.get("suffixes", ()))
        dataset = _HuggingFaceFilesDataset(
            repo_id=spec.path,
            revision=revision,
            repo_type=repo_type,
            path_prefix=path_prefix,
            suffixes=suffixes,
            cache_path=cache_path,
        )
        dataset.prepare()
        return dataset

    def iter_indexed_shard(
        self,
        dataset: _HuggingFaceFilesDataset,
        *,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Mapping[str, Any]]]:
        yield from dataset.iter_indexed_shard(num_shards, shard_id)


class _HuggingFaceFilesDataset:
    def __init__(
        self,
        *,
        repo_id: str,
        revision: str,
        repo_type: str,
        path_prefix: str,
        suffixes: tuple[str, ...],
        cache_path: Path,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.repo_type = repo_type
        self.path_prefix = path_prefix
        self.suffixes = suffixes
        self.cache_path = cache_path
        self._files: tuple[str, ...] | None = None

    @property
    def files(self) -> tuple[str, ...]:
        if self._files is None:
            self.prepare()
        if self._files is None:
            raise RuntimeError("Hugging Face file manifest was not prepared.")
        return self._files

    def prepare(self) -> None:
        manifest_path = self.cache_path / "files.json"
        with FileLock(self.cache_path / ".prepare.lock", wait_timeout=3600.0):
            files: list[str] | None = None
            if manifest_path.exists():
                files = _read_manifest(
                    manifest_path,
                    path_prefix=self.path_prefix,
                    suffixes=self.suffixes,
                )
            if files is None:
                files = _manifest_files(
                    _list_files(
                        self.repo_id,
                        self.path_prefix,
                        self.revision,
                        self.repo_type,
                    ),
                    path_prefix=self.path_prefix,
                    suffixes=self.suffixes,
                )
                _write_manifest(
                    manifest_path,
                    files,
                    path_prefix=self.path_prefix,
                    suffixes=self.suffixes,
                )
        self._files = tuple(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("Hugging Face files dataset index out of range.")
        file_name = self.files[index]
        local_path = _download_file(self, file_name)
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "repo_type": self.repo_type,
            "path": file_name,
            "local_path": str(local_path),
        }

    def iter_indexed_shard(
        self,
        num_shards: int,
        shard_id: int,
    ) -> Iterator[tuple[int, Mapping[str, Any]]]:
        validate_shard(num_shards, shard_id)
        for index in range(shard_id, len(self), num_shards):
            yield index, self[index]


def _path_prefix(spec: Spec) -> str:
    path_prefix = spec.load_options.get("path_prefix")
    path_template = spec.load_options.get("path_template")
    if path_prefix is not None and path_template is not None:
        raise ValueError("Use either path_prefix or path_template, not both.")
    if path_template is not None:
        template = _string_option(path_template, "path_template")
        if "{split}" in template and spec.split is None:
            raise ValueError("Hugging Face files path_template requires Spec.split.")
        path_prefix = template.format(split=spec.split or "")
    if path_prefix is None:
        return ""
    if not isinstance(path_prefix, str):
        raise ValueError("Hugging Face files path_prefix must be a string.")
    return path_prefix.strip("/")


def _revision(spec: Spec) -> str:
    revision = spec.version
    load_revision = spec.load_options.get("revision")
    if revision is not None and load_revision is not None:
        load_revision = _string_option(load_revision, "revision")
        if revision != load_revision:
            raise ValueError("Use either matching Spec.version and revision, or only one.")
    if revision is None:
        revision = load_revision if load_revision is not None else "main"
    return _string_option(revision, "revision")


def _string_option(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Hugging Face files {name} must be a non-empty string.")
    return value


def _suffixes(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        suffixes = (value,)
    elif isinstance(value, Sequence):
        suffixes = tuple(value)
    else:
        raise TypeError("Hugging Face files suffixes must be a string or sequence.")
    for suffix in suffixes:
        if not isinstance(suffix, str) or not suffix:
            raise ValueError(
                "Hugging Face files suffixes must contain non-empty strings."
            )
    return suffixes


def _list_files(
    repo_id: str,
    path_prefix: str,
    revision: str,
    repo_type: str,
) -> list[str]:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    namespace = _api_namespace(repo_type)
    path_segment = quote(path_prefix.strip("/"), safe="/")
    if path_segment:
        path_segment = f"/{path_segment}"
    url = (
        f"{endpoint}/api/{namespace}/{repo_id}/tree/{quote(revision, safe='')}{path_segment}"
        f"?recursive=true&expand=false&limit={_HUB_PAGE_LIMIT}"
    )
    files: list[str] = []
    seen_urls: set[str] = set()
    while url:
        if url in seen_urls:
            raise RuntimeError("Hugging Face Hub pagination returned a repeated URL.")
        seen_urls.add(url)
        request = Request(url, headers={"User-Agent": "anydataset"})
        with urlopen(request, timeout=60) as response:
            rows = json.loads(response.read().decode("utf-8"))
            if not isinstance(rows, list) or any(
                not isinstance(row, Mapping) for row in rows
            ):
                raise ValueError(
                    "Hugging Face Hub tree response must be a list of mappings."
                )
            for row in rows:
                path = row.get("path")
                if row.get("type") != "file":
                    continue
                if not isinstance(path, str):
                    raise ValueError(
                        "Hugging Face Hub file entries must contain a path."
                    )
                files.append(path)
            next_url = response.headers.get("Link")
        url = _next_link_url(next_url, endpoint)
        if len(rows) >= _HUB_PAGE_LIMIT and url is None:
            raise RuntimeError(
                "Hugging Face Hub listing truncated: full page without next Link."
            )
    return sorted(files)


def _api_namespace(repo_type: str) -> str:
    if repo_type == "dataset":
        return "datasets"
    if repo_type == "model":
        return "models"
    if repo_type == "space":
        return "spaces"
    raise ValueError(
        "Hugging Face files repo_type must be 'dataset', 'model', or 'space'."
    )


def _manifest_files(
    value: object,
    *,
    path_prefix: str,
    suffixes: tuple[str, ...],
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("Hugging Face file manifest must be a non-empty list.")

    prefix = path_prefix.strip("/")
    if prefix:
        prefix = f"{prefix}/"
    files = []
    for file_name in value:
        if not isinstance(file_name, str):
            raise ValueError("Hugging Face file manifest entries must be strings.")
        if prefix and not file_name.startswith(prefix):
            raise ValueError(
                f"Hugging Face file manifest entries must be under {prefix}."
            )
        if suffixes and not file_name.endswith(suffixes):
            expected = ", ".join(suffixes)
            raise ValueError(
                f"Hugging Face file manifest entries must end with one of: {expected}."
            )
        files.append(file_name)
    if len(set(files)) != len(files):
        raise ValueError("Hugging Face file manifest entries must be unique.")
    return files


def _read_manifest(
    path: Path,
    *,
    path_prefix: str,
    suffixes: tuple[str, ...],
) -> list[str] | None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    if (
        raw.get("schema_version") != _MANIFEST_SCHEMA_VERSION
        or raw.get("listed_complete") is not True
        or raw.get("path_prefix") != path_prefix
        or tuple(raw.get("suffixes", ())) != suffixes
    ):
        return None
    return _manifest_files(
        raw.get("files"),
        path_prefix=path_prefix,
        suffixes=suffixes,
    )


def _write_manifest(
    path: Path,
    files: list[str],
    *,
    path_prefix: str,
    suffixes: tuple[str, ...],
) -> None:
    _write_json(
        path,
        {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "listed_complete": True,
            "path_prefix": path_prefix,
            "suffixes": list(suffixes),
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
            raise ValueError("Hugging Face Hub Link header next URL is malformed.")
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


def _download_file(dataset: _HuggingFaceFilesDataset, file_name: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Hugging Face files support requires 'pip install anydataset[huggingface]'."
        ) from exc

    return hf_hub_download(
        repo_id=dataset.repo_id,
        repo_type=dataset.repo_type,
        revision=dataset.revision,
        filename=file_name,
        cache_dir=str(dataset.cache_path / "hf"),
    )


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)
