from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    path: str
    name: str | None = None
    split: str | None = None
    version: str | None = None
    adapter: Any | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    ref: str | None = None

    @property
    def key(self) -> str:
        if self.ref:
            return self.ref
        if self.name and self.split:
            return f"{self.name}:{self.split}"
        if self.name:
            return self.name
        return self.path


DEFAULT_DATASET_MAP: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(source="huggingface", path="ylecun/mnist", name="mnist"),
    "cifar10": DatasetSpec(source="huggingface", path="uoft-cs/cifar10", name="cifar10"),
}


class DatasetRegistry:
    def __init__(self, dataset_map: Mapping[str, DatasetSpec] | None = None):
        self._dataset_map = dict(DEFAULT_DATASET_MAP)
        if dataset_map:
            self._dataset_map.update(dataset_map)

    def resolve(self, dataset_ref: str) -> DatasetSpec:
        source, body = _split_source_prefix(dataset_ref)
        if source == "hf":
            path, split = _split_name_and_split(body)
            name = path.rsplit("/", 1)[-1]
            return DatasetSpec(
                source="huggingface",
                path=path,
                name=name,
                split=split,
                ref=dataset_ref,
            )

        if source == "local":
            path, split = _split_name_and_split(body)
            name = Path(path).name or path
            return DatasetSpec(
                source="local_files",
                path=path,
                name=name,
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
        return replace(
            spec,
            name=spec.name or name,
            split=split or spec.split,
            ref=dataset_ref,
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
