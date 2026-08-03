"""Source-owned policies for separating physical and operational Spec options."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceIdentityPolicy:
    operational_load_options: frozenset[str]

    def physical_load_options(
        self,
        load_options: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            key: value
            for key, value in load_options.items()
            if key not in self.operational_load_options
        }


_DEFAULT_POLICY = SourceIdentityPolicy(frozenset({"prepare_workers"}))
_POLICIES: dict[str, SourceIdentityPolicy] = {}


def register_source_identity(
    source: str,
    operational_load_options: Collection[str],
) -> SourceIdentityPolicy:
    if source in _POLICIES:
        raise ValueError(f"Dataset source identity {source!r} is already registered.")
    policy = SourceIdentityPolicy(
        _DEFAULT_POLICY.operational_load_options
        | _option_names(operational_load_options)
    )
    _POLICIES[source] = policy
    return policy


def physical_load_options(
    source: str,
    load_options: Mapping[str, Any],
) -> dict[str, Any]:
    return _POLICIES.get(source, _DEFAULT_POLICY).physical_load_options(load_options)


def _option_names(options: Collection[str]) -> frozenset[str]:
    if isinstance(options, str):
        raise TypeError("Operational load options must be a collection of strings.")
    for option in options:
        if not isinstance(option, str):
            raise TypeError("Operational load options must contain strings.")
        if not option:
            raise ValueError("Operational load options must not contain empty strings.")
    return frozenset(options)
