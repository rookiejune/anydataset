from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Union

Devices = Union[Literal["auto"], str, Iterable[str]]


def resolve_devices(devices: Devices) -> tuple[str, ...]:
    if devices == "auto":
        count = _cuda_device_count()
        if count > 0:
            return tuple(f"cuda:{index}" for index in range(count))
        return ("cpu",)
    if isinstance(devices, str):
        resolved = (devices,)
    else:
        resolved = tuple(devices)
    if not resolved:
        raise ValueError("devices must not be empty.")
    return resolved


def _cuda_device_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()
