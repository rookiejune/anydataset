from __future__ import annotations

from collections.abc import Collection, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ...types import Spec


class DatasetSource(Protocol):
    def prepare(self, spec: Spec, cache_path: Path) -> Iterable[Any]:
        raise NotImplementedError


def validate_load_options(
    spec: Spec,
    allowed: Collection[str],
    *,
    source: str,
) -> None:
    extra = set(spec.load_options) - set(allowed)
    if extra:
        name = min(extra)
        raise TypeError(f"Unexpected {source} load option: {name}.")
