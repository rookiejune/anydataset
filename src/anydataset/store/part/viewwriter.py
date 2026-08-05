from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

from ...types.item import Modality, Role, View
from ..manifest.schema import (
    ViewManifestEntry,
)
from ..manifest.io import view_manifest_writer
from ..paths import view_ready_path, view_shard_path
from ..payload.archive import (
    Payload,
    add_payload,
    payload_for_view,
    payload_tar_span,
    write_payload_index,
)


class ViewWriter:
    def __init__(
        self,
        root: Path,
        view: tuple[Role, Modality, View],
        max_shard_samples: int | None,
        max_shard_bytes: int | None = None,
        shard_prefix: str = "",
    ) -> None:
        self.root = root
        self.view = view
        self.max_shard_samples = max_shard_samples
        self.max_shard_bytes = max_shard_bytes
        self.shard_prefix = shard_prefix
        self.shard_index = 0
        self.shard = _shard_name(self.shard_index, self.shard_prefix)
        self.shard_samples = 0
        self.manifest = view_manifest_writer(root, view)
        self.tar = self._open_shard(self.shard)
        self.closed = False

    def write(self, sample_index: int, value: Any) -> None:
        self.write_payload(
            sample_index,
            payload_for_view(self.view, sample_index, value),
        )

    def write_payload(self, sample_index: int, payload: Payload) -> None:
        if self._should_roll(payload):
            self._roll_shard()
        add_payload(self.tar, payload)
        self.shard_samples += 1
        self.manifest.write(
            ViewManifestEntry(
                role=self.view[0],
                modality=self.view[1],
                view=self.view[2],
                sample_index=sample_index,
                shard=self.shard,
                key=payload.key,
            )
        )

    def close(self) -> None:
        self.close_payload()
        self.manifest.close()
        view_ready_path(self.root, self.view).touch()

    def close_payload(self) -> None:
        if not self.closed:
            self._close_shard(write_index=True)

    def abort(self) -> None:
        self._close_shard(write_index=False)
        self.manifest.abort()

    def _should_roll(self, payload: Payload) -> bool:
        if self.shard_samples == 0:
            return False
        if (
            self.max_shard_samples is not None
            and self.shard_samples >= self.max_shard_samples
        ):
            return True
        if self.max_shard_bytes is None:
            return False
        projected_offset = self.tar.offset + payload_tar_span(self.tar, payload)
        return _closed_tar_size(projected_offset) > self.max_shard_bytes

    def _roll_shard(self) -> None:
        self._close_shard(write_index=True)
        self.shard_index += 1
        self.shard = _shard_name(self.shard_index, self.shard_prefix)
        self.shard_samples = 0
        self.tar = self._open_shard(self.shard)
        self.closed = False

    def _close_shard(self, *, write_index: bool) -> None:
        if self.closed:
            return
        self.tar.close()
        self.closed = True
        if write_index:
            write_payload_index(self.root, self.view, self.shard)

    def _open_shard(self, shard: str) -> tarfile.TarFile:
        path = view_shard_path(self.root, self.view, shard)
        path.parent.mkdir(parents=True, exist_ok=True)
        return tarfile.open(path, "w")


def _shard_name(index: int, prefix: str = "") -> str:
    return f"{prefix}{index:06d}.tar"


def _closed_tar_size(offset: int) -> int:
    with_end_blocks = offset + tarfile.BLOCKSIZE * 2
    return (
        (with_end_blocks + tarfile.RECORDSIZE - 1)
        // tarfile.RECORDSIZE
        * tarfile.RECORDSIZE
    )
