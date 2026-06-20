from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from anydataset.datasets.catalog import DEFAULT_DATASET_MAP

from .spec import DatasetSpec


class DatasetResolver:
    def __init__(self, dataset_map: Mapping[str, DatasetSpec] | None = None):
        self._dataset_map = dict(DEFAULT_DATASET_MAP)
        if dataset_map:
            _validate_dataset_map(dataset_map)
            self._dataset_map.update(dataset_map)

    def resolve(self, dataset_ref: str) -> DatasetSpec:
        source, body = _split_source_prefix(dataset_ref)
        if source == "hf":
            path, split = _split_name_and_split(body)
            return DatasetSpec(
                source="huggingface",
                path=path,
                name=path,
                split=split,
                ref=dataset_ref,
            )

        if source == "local":
            path, split = _split_name_and_split(body)
            if not path:
                raise ValueError("local dataset references must include a named path.")
            return DatasetSpec(
                source="local_files",
                path=path,
                name=path,
                split=split,
                ref=dataset_ref,
            )

        name, split = _split_name_and_split(dataset_ref)
        if name not in self._dataset_map:
            raise KeyError(
                f"Unknown dataset {name!r}. Add it to dataset_map or use an explicit "
                "`hf://` or `local://` dataset reference."
            )

        spec = self._dataset_map[name]
        if spec.name != name:
            raise ValueError(
                f"Dataset map key {name!r} must match DatasetSpec.name {spec.name!r}."
            )
        return replace(
            spec,
            split=split or spec.split,
            ref=dataset_ref,
        )


def _validate_dataset_map(dataset_map: Mapping[str, DatasetSpec]) -> None:
    for name, spec in dataset_map.items():
        if not isinstance(spec, DatasetSpec):
            raise TypeError("dataset_map values must be DatasetSpec instances.")
        if name != spec.name:
            raise ValueError(
                f"Dataset map key {name!r} must match DatasetSpec.name {spec.name!r}."
            )


def _split_source_prefix(dataset_ref: str) -> tuple[str | None, str]:
    if dataset_ref.startswith("hf://"):
        return "hf", dataset_ref[len("hf://") :]
    if dataset_ref.startswith("local://"):
        return "local", dataset_ref[len("local://") :]
    return None, dataset_ref


def _split_name_and_split(value: str) -> tuple[str, str | None]:
    if ":" not in value:
        return value, None
    name, split = value.rsplit(":", 1)
    return name, split or None
