from __future__ import annotations

from ..types import item

type RoleLike = item.Role | str | None


class MissingModalityError(KeyError):
    def __init__(self, modality: item.Modality | str, role: RoleLike = None):
        suffix = "" if role is None else f" role {role!r}"
        super().__init__(f"Dataset source does not provide {modality!r}{suffix}.")
        self.modality = modality
        self.role = role
