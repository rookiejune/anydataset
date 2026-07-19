"""Shared process runtime configuration for dataset-wide workers.

The module owns process start-method choices and device ownership boundaries.
It does not own materializer, filter, provider, or cache semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from ._parallel import StartMethod, validate_start_method

AutoStartMethod = Union[StartMethod, Literal["auto"]]


@dataclass(frozen=True)
class Runtime:
    process_start_method: StartMethod = "spawn"
    server_start_method: StartMethod | None = None
    reader_start_method: AutoStartMethod = "auto"
    writer_start_method: AutoStartMethod = "auto"

    def __post_init__(self) -> None:
        validate_start_method("process_start_method", self.process_start_method)
        validate_start_method(
            "server_start_method",
            self.server_start_method,
            optional=True,
        )
        validate_start_method(
            "reader_start_method",
            self.reader_start_method,
            auto=True,
        )
        validate_start_method(
            "writer_start_method",
            self.writer_start_method,
            auto=True,
        )

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
