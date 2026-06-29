from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import Enum

from .types import FilterLabel


def label(value: FilterLabel) -> str:
    if isinstance(value, bool):
        return "accept" if value else "reject"
    if isinstance(value, Enum):
        enum_value = value.value
        output = enum_value if isinstance(enum_value, str) else value.name
        return validate_label(str(output))
    if isinstance(value, str):
        return validate_label(value)
    raise TypeError("filter predicate must return bool, str, or Enum.")


def validate_label(value: str) -> str:
    if value == "":
        raise ValueError("filter label must not be empty.")
    return value


def unique_labels(labels: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in labels:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)


def rule_cache_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def label_file_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_string(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value == "":
        raise ValueError(f"{name} must not be empty.")
