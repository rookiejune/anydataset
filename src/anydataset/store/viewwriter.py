from __future__ import annotations

import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..types.item import Modality, Role, View
from .jsonio import write_json
from .manifest import (
    ViewManifestEntry,
)
from .manifestio import view_manifest_writer
from .paths import view_json_path, view_ready_path, view_shard_path
from .payload import add_payload, payload_for_view


class ViewWriter:
    def __init__(
        self,
        root: Path,
        view: tuple[Role, Modality, View],
        provider: Mapping[str, Any] | None = None,
        max_shard_samples: int | None = None,
        max_shard_bytes: int | None = None,
        shard_prefix: str = "",
    ) -> None:
        self.root = root
        self.view = view
        self.provider = dict(provider or {})
        self.max_shard_samples = max_shard_samples
        self.max_shard_bytes = max_shard_bytes
        self.shard_prefix = shard_prefix
        self.shard_index = 0
        self.shard = _shard_name(self.shard_index, self.shard_prefix)
        self.shard_samples = 0
        self.shard_bytes = 0
        self.manifest = view_manifest_writer(root, view)
        self.tar = self._open_shard(self.shard)
        self.closed = False

    def write(self, sample_id: str, value: Any) -> None:
        payload = payload_for_view(self.view, sample_id, value)
        if self._should_roll(len(payload.data)):
            self._roll_shard()
        add_payload(self.tar, payload)
        self.shard_samples += 1
        self.shard_bytes += len(payload.data)
        self.manifest.write(
            ViewManifestEntry(
                role=self.view[0],
                modality=self.view[1],
                view=self.view[2],
                sample_id=sample_id,
                shard=self.shard,
                key=payload.key,
            )
        )

    def close(self) -> None:
        self.close_payload()
        view_json = {
            "role": self.view[0],
            "modality": self.view[1],
            "view": self.view[2],
            "provider": self.provider,
        }
        write_json(view_json_path(self.root, self.view), view_json)
        self.manifest.close()
        view_ready_path(self.root, self.view).touch()

    def close_payload(self) -> None:
        if not self.closed:
            self.tar.close()
            self.closed = True

    def abort(self) -> None:
        self.close_payload()
        self.manifest.abort()

    def _should_roll(self, payload_bytes: int) -> bool:
        if self.shard_samples == 0:
            return False
        if (
            self.max_shard_samples is not None
            and self.shard_samples >= self.max_shard_samples
        ):
            return True
        return (
            self.max_shard_bytes is not None
            and self.shard_bytes + payload_bytes > self.max_shard_bytes
        )

    def _roll_shard(self) -> None:
        self.tar.close()
        self.shard_index += 1
        self.shard = _shard_name(self.shard_index, self.shard_prefix)
        self.shard_samples = 0
        self.shard_bytes = 0
        self.tar = self._open_shard(self.shard)

    def _open_shard(self, shard: str) -> tarfile.TarFile:
        path = view_shard_path(self.root, self.view, shard)
        path.parent.mkdir(parents=True, exist_ok=True)
        return tarfile.open(path, "w")


def _shard_name(index: int, prefix: str = "") -> str:
    return f"{prefix}{index:06d}.tar"
