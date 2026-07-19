"""Shared validation helpers for public constructor arguments."""

from __future__ import annotations

from math import isfinite


def positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def non_negative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return positive_int(name, value)


def positive_float(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number.")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def optional_positive_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return positive_float(name, value)
