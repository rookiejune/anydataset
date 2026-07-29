from __future__ import annotations

from collections.abc import Iterable, Iterator
from enum import Enum
from itertools import zip_longest
from typing import Any, TypeVar, overload

from typing_extensions import NotRequired, Self

__all__ = ["NotRequired", "Self", "StrEnum", "strict_zip"]


class StrEnum(str, Enum):
    # EnumMeta requires a plain function here on Python 3.9.
    def _generate_next_value_(  # pyright: ignore[reportIncompatibleMethodOverride]
        name: str,
        start: int,
        count: int,
        last_values: list[str],
    ) -> str:
        return name.lower()


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

_MISSING = object()


@overload
def strict_zip(
    first: Iterable[T],
    second: Iterable[U],
    /,
) -> Iterator[tuple[T, U]]: ...


@overload
def strict_zip(
    first: Iterable[T],
    second: Iterable[U],
    third: Iterable[V],
    /,
) -> Iterator[tuple[T, U, V]]: ...


def strict_zip(*iterables: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    for values in zip_longest(*iterables, fillvalue=_MISSING):
        if any(value is _MISSING for value in values):
            raise ValueError("zip() argument lengths differ.")
        yield values
