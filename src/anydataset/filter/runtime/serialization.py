from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..cache.generations import lease_filter_generation
from ..cache.identity import filter_base
from ..cache.storage import read_partitions
from ..types import DatasetFactory, FilterPredicate

if TYPE_CHECKING:
    from ..api import FilteredDataset, _FilterCache


def restore_filter_cache(
    dataset_factory: DatasetFactory,
    rule_name: str,
    cache_path: Path,
    metrics_path: Path | None,
    input_id: str | None,
    rule_id: str | None = None,
    version: str | None = None,
    content_id: str | None = None,
) -> _FilterCache:
    from ..api import FilterRule, _FilterCache

    generation = lease_filter_generation(cache_path)
    try:
        return _FilterCache(
            filter_base(dataset_factory()),
            read_partitions(generation.path),
            FilterRule(
                rule_name,
                unavailable_filter_factory,
                rule_id=rule_id,
                version=version,
                content_id=content_id,
            ),
            generation.path,
            lease=generation.lease,
            dataset_factory=dataset_factory,
            metrics_path=metrics_path,
            input_id=input_id,
        )
    except Exception:
        generation.lease.close()
        raise


def restore_filtered_dataset(
    cache: _FilterCache,
    labels: tuple[str, ...],
) -> FilteredDataset:
    from ..api import FilteredDataset

    return FilteredDataset._from_cache(cache, labels=labels)


def unavailable_filter_factory() -> FilterPredicate:
    raise RuntimeError("cached filtered dataset cannot rebuild its upstream rule.")
