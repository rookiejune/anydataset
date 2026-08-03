"""Validation shared by versioned store runtime pickle states."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

PICKLE_SCHEMA_VERSION_FIELD = "pickle_schema_version"


def decode_pickle_state(
    state: object,
    *,
    kind: str,
    current_version: int,
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(state, dict) or any(
        not isinstance(name, str) for name in state
    ):
        raise TypeError(f"Invalid {kind} pickle state.")
    values = dict(state)
    if PICKLE_SCHEMA_VERSION_FIELD not in values:
        return True, values
    version = values.pop(PICKLE_SCHEMA_VERSION_FIELD)
    if type(version) is not int:
        raise TypeError(f"{kind} pickle_schema_version must be an integer.")
    if version != current_version:
        raise ValueError(
            f"Unsupported {kind} pickle_schema_version {version!r}; "
            f"expected {current_version}."
        )
    return False, values


def validate_pickle_fields(
    state: Mapping[str, Any],
    *,
    kind: str,
    required: Collection[str],
    optional: Collection[str] = (),
) -> None:
    fields = frozenset(state)
    missing = frozenset(required) - fields
    if missing:
        raise ValueError(
            f"{kind} pickle state is missing required field {min(missing)!r}."
        )
    unsupported = fields - frozenset(required) - frozenset(optional)
    if unsupported:
        raise ValueError(
            f"{kind} pickle state has unsupported field {min(unsupported)!r}."
        )
