from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..cache.identity import FilterBase, filter_base
from ..cache.generations import GenerationLease
from ..types import DatasetFactory

if TYPE_CHECKING:
    from ..api import FilterRule


@dataclass(frozen=True)
class FilteredDatasetFactory:
    base: DatasetFactory
    rule_name: str
    labels: tuple[str, ...]
    cache_path: Path
    metrics_path: Path | None
    input_id: str | None
    rule_id: str
    lease: GenerationLease = field(repr=False, compare=False)
    version: str | None = None
    content_id: str | None = None

    def __call__(self) -> FilterBase:
        from ..api import FilteredDataset, FilterRule

        return FilteredDataset._from_generation(
            filter_base(self.base()),
            FilterRule(
                self.rule_name,
                _unavailable_filter_factory,
                rule_id=self.rule_id,
                version=self.version,
                content_id=self.content_id,
            ),
            self.cache_path,
            self.labels,
            dataset_factory=self.base,
            metrics_path=self.metrics_path,
            input_id=self.input_id,
        )

    def __reduce__(self):
        return (
            restore_filtered_dataset_factory,
            (
                self.base,
                self.rule_name,
                self.labels,
                self.cache_path,
                self.metrics_path,
                self.input_id,
                self.rule_id,
                self.version,
                self.content_id,
            ),
        )


def make_filtered_dataset_factory(
    base: DatasetFactory,
    rule: FilterRule,
    labels: tuple[str, ...],
    cache_path: Path,
    metrics_path: Path | None,
    input_id: str | None,
) -> FilteredDatasetFactory:
    return _make_filtered_dataset_factory(
        base,
        rule.name,
        labels,
        cache_path,
        metrics_path,
        input_id,
        rule.rule_id,
        rule.version,
        rule.content_id,
    )


def restore_filtered_dataset_factory(
    base: DatasetFactory,
    rule_name: str,
    labels: tuple[str, ...],
    cache_path: Path,
    metrics_path: Path | None,
    input_id: str | None,
    rule_id: str | None = None,
    version: str | None = None,
    content_id: str | None = None,
) -> FilteredDatasetFactory:
    return _make_filtered_dataset_factory(
        base,
        rule_name,
        labels,
        cache_path,
        metrics_path,
        input_id,
        rule_name if rule_id is None else rule_id,
        version,
        content_id,
    )


def _make_filtered_dataset_factory(
    base: DatasetFactory,
    rule_name: str,
    labels: tuple[str, ...],
    cache_path: Path,
    metrics_path: Path | None,
    input_id: str | None,
    rule_id: str,
    version: str | None,
    content_id: str | None,
) -> FilteredDatasetFactory:
    from ..cache.generations import lease_filter_generation

    generation = lease_filter_generation(cache_path)
    return FilteredDatasetFactory(
        base=base,
        rule_name=rule_name,
        labels=labels,
        cache_path=Path(cache_path),
        metrics_path=None if metrics_path is None else Path(metrics_path),
        input_id=input_id,
        rule_id=rule_id,
        version=version,
        content_id=content_id,
        lease=generation.lease,
    )


def _unavailable_filter_factory():
    raise RuntimeError("cached filtered-view factory cannot rebuild its upstream rule.")
