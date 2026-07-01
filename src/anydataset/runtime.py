from __future__ import annotations

"""Shared process runtime configuration for dataset-wide workers.

The module owns process start-method choices and device ownership boundaries.
It does not own materializer, filter, provider, or cache semantics.
"""

from dataclasses import dataclass
from typing import Literal

from ._parallel import StartMethod

type DeviceScope = Literal["local", "remote"]


@dataclass(frozen=True)
class Runtime:
    process_start_method: StartMethod = "spawn"
    loader_start_method: StartMethod = "spawn"
    device_scope: DeviceScope = "local"

    @property
    def uses_local_device(self) -> bool:
        return self.device_scope == "local"

