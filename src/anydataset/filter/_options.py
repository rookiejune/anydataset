from __future__ import annotations

from typing import cast

from ..runtime import Runtime
from .rules import validate_string
from .types import FilterApplyKwargs

DEFAULT_MAX_SHARD_SAMPLES = 1_000_000
DEFAULT_COMMIT_SAMPLES = 100_000

_DEFAULTS: FilterApplyKwargs = {
    "input_id": None,
    "metrics": False,
    "device": "auto",
    "batch_size": 1,
    "num_workers": 0,
    "prefetch_factor": None,
    "commit_samples": DEFAULT_COMMIT_SAMPLES,
    "max_shard_samples": DEFAULT_MAX_SHARD_SAMPLES,
    "write_workers": 1,
    "write_prefetch": None,
    "worker_timeout": None,
    "runtime": Runtime(),
}


def options(values: FilterApplyKwargs) -> FilterApplyKwargs:
    extra = set(values) - set(_DEFAULTS)
    if extra:
        name = min(extra)
        raise TypeError(f"Unexpected filter apply option: {name}.")
    resolved = {key: values.get(key, default) for key, default in _DEFAULTS.items()}
    input_id = resolved["input_id"]
    if input_id is not None:
        validate_string("input_id", input_id)
    return cast(FilterApplyKwargs, resolved)
