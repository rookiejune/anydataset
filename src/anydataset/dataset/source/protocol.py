from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ...types import Spec


class DatasetSource(Protocol):
    def prepare(self, spec: Spec, cache_path: Path) -> Iterable[Any]:
        raise NotImplementedError
