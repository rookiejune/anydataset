from __future__ import annotations

import warnings
from enum import auto
from typing import Union

from ._compat import StrEnum


class _LegacyPolicy(StrEnum):
    """Policy for intentionally handling legacy data or runtime contracts."""

    WARN = auto()
    ALLOW = auto()
    REJECT = auto()


_LegacyPolicyValue = Union[_LegacyPolicy, str]


def _policy(value: _LegacyPolicyValue) -> _LegacyPolicy:
    try:
        return _LegacyPolicy(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in _LegacyPolicy)
        raise ValueError(f"legacy_policy must be one of: {choices}.") from exc


def legacy(
    subject: str,
    *,
    legacy_policy: _LegacyPolicyValue,
    warning: str,
    error: str | None = None,
    stacklevel: int = 2,
) -> None:
    resolved = _policy(legacy_policy)
    if resolved is _LegacyPolicy.ALLOW:
        return
    if resolved is _LegacyPolicy.WARN:
        warnings.warn(warning, RuntimeWarning, stacklevel=stacklevel)
        return
    raise ValueError(
        error or f"{subject} is legacy; set legacy_policy='allow' to use it."
    )


__all__ = ["legacy"]
