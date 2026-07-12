from __future__ import annotations

from collections.abc import Iterable, Iterator
from enum import Enum
from itertools import zip_longest
from typing import TypeVar

try:
    from typing import NotRequired, Self
except ImportError:
    from typing_extensions import NotRequired, Self

__all__ = ["NotRequired", "Self", "StrEnum", "strict_zip"]


class StrEnum(str, Enum):
    def _generate_next_value_(
        name: str,
        start: int,
        count: int,
        last_values: list[str],
    ) -> str:
        return name.lower()


T = TypeVar("T")

_MISSING = object()


def strict_zip(*iterables: Iterable[T]) -> Iterator[tuple[T, ...]]:
    for values in zip_longest(*iterables, fillvalue=_MISSING):
        if any(value is _MISSING for value in values):
            raise ValueError("zip() argument lengths differ.")
        yield values
