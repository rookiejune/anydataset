from __future__ import annotations

from ..types.item import Modality, Role, View


def view_path(
    view: tuple[Role, Modality, View],
) -> tuple[str, str, str]:
    role, modality, key = view
    return role.value, modality.value, key.value


def sample_ref_path(ref: tuple[Role, Modality]) -> tuple[str, str]:
    role, modality = ref
    return role.value, modality.value


def validate_entry_ref(
    entry: tuple[Role, Modality, View],
    path: tuple[Role, Modality, View],
) -> None:
    if entry != path:
        raise ValueError("View manifest entry ref must match its path.")
