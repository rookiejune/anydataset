from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..adapters.catalog import DEFAULT_DATASET_MAP

from .spec import DatasetSpec

DatasetRef = str | DatasetSpec


def resolve_dataset_spec(
    dataset: DatasetRef,
    dataset_map: Mapping[str, DatasetSpec] | None = None,
) -> DatasetSpec:
    if isinstance(dataset, DatasetSpec):
        return dataset
    if isinstance(dataset, str):
        return DatasetResolver(dataset_map).resolve(dataset)
    raise TypeError("dataset must be a string reference or DatasetSpec.")


def resolve_dataset_specs(
    datasets: DatasetRef | Sequence[DatasetRef],
    dataset_map: Mapping[str, DatasetSpec] | None = None,
) -> list[DatasetSpec]:
    dataset_refs = _dataset_refs(datasets)
    if not dataset_refs:
        raise ValueError("datasets must contain at least one dataset.")

    resolver = DatasetResolver(dataset_map)
    specs: list[DatasetSpec] = []
    for dataset in dataset_refs:
        if isinstance(dataset, DatasetSpec):
            specs.append(dataset)
        elif isinstance(dataset, str):
            specs.append(resolver.resolve(dataset))
        else:
            raise TypeError("datasets must contain only string references or DatasetSpec values.")
    return specs


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
            )

        if source == "unified":
            path, split = _split_name_and_split(body)
            if not path:
                raise ValueError("unified dataset references must include a named path.")
            return DatasetSpec(
                source="unified",
                path=path,
                name=path,
                split=split,
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
    if dataset_ref.startswith("unified://"):
        return "unified", dataset_ref[len("unified://") :]
    return None, dataset_ref


def _split_name_and_split(value: str) -> tuple[str, str | None]:
    if ":" not in value:
        return value, None
    name, split = value.rsplit(":", 1)
    return name, split or None


def _dataset_refs(datasets: DatasetRef | Sequence[DatasetRef]) -> list[DatasetRef]:
    if isinstance(datasets, (str, DatasetSpec)):
        return [datasets]
    return list(datasets)
