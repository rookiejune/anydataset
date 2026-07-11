from __future__ import annotations

"""Shared validation helpers for public constructor arguments."""


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


def optional_positive_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return float(value)
