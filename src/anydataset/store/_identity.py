"""Stable logical identities shared by online and canonical store datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

from ..types.item import Modality, Role, View

_REQUIRED_PROVENANCE = ("input_id", "provider_id", "output_id")


def materialized_universe_id(
    dataset_id: str,
    split: str | None,
    provenance: Mapping[str, str],
    sample_count: int,
    views: Iterable[tuple[Role, Modality, View]],
) -> str | None:
    """Identify one complete derived universe across online and ready paths."""

    if any(not provenance.get(key) for key in _REQUIRED_PROVENANCE):
        return None
    payload = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "split": split,
        "provenance": {key: provenance[key] for key in sorted(provenance)},
        "sample_count": sample_count,
        "views": [
            [role.value, modality.value, view.value] for role, modality, view in views
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"materialized-v1:{hashlib.sha256(encoded).hexdigest()}"


__all__ = ["materialized_universe_id"]
