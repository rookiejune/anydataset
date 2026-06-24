from __future__ import annotations

from dataclasses import dataclass

from .schema import ModalityKey, ModalityRole


@dataclass(frozen=True)
class ViewRef:
    modality: ModalityKey
    view_key: str
    role: ModalityRole = None

    def __post_init__(self) -> None:
        if not isinstance(self.modality, ModalityKey):
            raise TypeError("modality must be a ModalityKey.")
        _validate_segment("view_key", self.view_key)
        if self.role is not None:
            _validate_segment("role", self.role)
            if self.role == "views":
                raise ValueError("role cannot be 'views'.")
            object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "view_key", str(self.view_key))

    def path_parts(self) -> tuple[str, ...]:
        if self.role is None:
            return self.modality.value, "views", self.view_key
        return self.modality.value, self.role, "views", self.view_key


def _validate_segment(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if value in {"", ".", ".."}:
        raise ValueError(f"{name} must be a non-empty path segment.")
    if "/" in value:
        raise ValueError(f"{name} cannot contain '/'.")
