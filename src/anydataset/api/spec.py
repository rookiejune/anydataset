from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    path: str
    name: str
    split: str | None = None
    version: str | None = None
    adapter: Any | None = None
    load_options: Mapping[str, Any] = field(default_factory=dict)
    sample_metadata: Mapping[str, Any] = field(default_factory=dict)
    ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("DatasetSpec.name must be a non-empty unique dataset name.")

    @property
    def key(self) -> str:
        if self.ref:
            return self.ref
        if self.name and self.split:
            return f"{self.name}:{self.split}"
        return self.name
