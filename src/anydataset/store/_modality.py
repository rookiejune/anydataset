from __future__ import annotations

"""Materialize missing modalities from existing role-local inputs.

The module enforces the modality materializer contract: each role must have
exactly one non-output input modality, and generated items do not inherit meta.
"""

from collections.abc import Iterator, Mapping
from typing import Any

from ..types.item import (
    AudioItem,
    AudioView,
    ImageItem,
    ImageView,
    Item,
    Modality,
    Role,
    Sample,
    TextItem,
    TextView,
    View,
)
from ._types import ModalityProviderLike, output_modality, views


def with_modality_provider(
    sample: Sample,
    provider: ModalityProviderLike,
) -> Sample:
    output = provider.output
    out_modality = output_modality(output)
    reference_role = getattr(provider, "reference_role", None)
    roles = role_items(sample)
    return {
        (role, out_modality): with_modality_view(
            output,
            provider(input_views(roles, role, input_item, out_modality, reference_role)),
        )
        for role, input_item in modality_inputs(
            roles,
            out_modality,
            reference_role,
        )
    }


def role_items(
    sample: Sample,
) -> dict[Role, dict[Modality, Item]]:
    roles: dict[Role, dict[Modality, Item]] = {}
    for (role, modality), item in sample.items():
        items = roles.setdefault(role, {})
        items[modality] = item
    return roles


def modality_inputs(
    roles: Mapping[Role, Mapping[Modality, Item]],
    output: Modality,
    reference_role: Role | None = None,
) -> Iterator[tuple[Role, Item]]:
    if reference_role is not None:
        reference_items = roles.get(reference_role)
        if reference_items is None or output not in reference_items:
            raise ValueError(
                f"Reference role {reference_role.value!r} must already have "
                f"output modality {output.value!r}."
            )

    for role, items in roles.items():
        if output in items:
            if role is reference_role:
                continue
            raise ValueError(
                f"Role {role.value!r} already has output modality {output.value!r}."
            )
        inputs = tuple((modality, item) for modality, item in items.items())
        if len(inputs) != 1:
            names = ", ".join(sorted(modality.value for modality, _ in inputs))
            raise ValueError(
                f"Role {role.value!r} needs exactly one input modality when "
                f"materializing {output.value!r}; got {names or 'none'}."
            )
        yield role, inputs[0][1]


def input_views(
    roles: Mapping[Role, Mapping[Modality, Item]],
    role: Role,
    input_item: Item,
    output: Modality,
    reference_role: Role | None,
) -> Mapping[View, Any]:
    if reference_role is None:
        return input_item.views
    reference_item = roles[reference_role][output]
    return {**input_item.views, **reference_item.views}


def with_modality_view(
    view: View,
    value: Any,
) -> Item:
    if isinstance(view, AudioView):
        return AudioItem(views=views(view, value))
    if isinstance(view, ImageView):
        return ImageItem(views=views(view, value))
    if isinstance(view, TextView):
        return TextItem(views=views(view, value))
    raise TypeError(
        "modality materializer output must be an AudioView, ImageView, or TextView."
    )
