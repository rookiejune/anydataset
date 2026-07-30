"""Helpers for validated immutable value objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError


class Immutable:
    """Allow construction assignments, then reject attribute mutation."""

    __slots__ = ()

    _immutable_sealed: bool

    def seal(self) -> None:
        self._immutable_sealed = True

    def __setstate__(self, state: object) -> None:
        if not isinstance(state, dict):
            raise TypeError("invalid immutable pickle state.")
        for name, value in state.items():
            if name != "_immutable_sealed":
                setattr(self, name, value)
        self.seal()

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_immutable_sealed", False):
            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_immutable_sealed", False):
            raise FrozenInstanceError(f"cannot delete field {name!r}")
        super().__delattr__(name)
