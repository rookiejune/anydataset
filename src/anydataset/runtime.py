"""Shared process runtime configuration for dataset-wide workers.

The module owns process start-method choices and device ownership boundaries.
It does not own materializer, filter, provider, or cache semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from ._parallel import StartMethod

AutoStartMethod = Union[StartMethod, Literal["auto"]]


@dataclass(frozen=True)
class Runtime:
    process_start_method: StartMethod = "spawn"
    server_start_method: StartMethod | None = None
    reader_start_method: AutoStartMethod = "auto"
    writer_start_method: AutoStartMethod = "auto"

    @property
    def uses_local_device(self) -> bool:
        return self.server_start_method is None

    @property
    def reader_worker_start_method(self) -> StartMethod:
        return self._start_method(self.reader_start_method)

    @property
    def writer_worker_start_method(self) -> StartMethod:
        return self._start_method(self.writer_start_method)

    def _start_method(self, value: AutoStartMethod) -> StartMethod:
        if value != "auto":
            return value
        if self.server_start_method is not None:
            return "fork"
        return "spawn"
