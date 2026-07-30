from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Union

Devices = Union[Literal["auto"], str, Iterable[str]]


def resolve_devices(devices: Devices) -> tuple[str, ...]:
    if isinstance(devices, str):
        if devices == "auto":
            count = _cuda_device_count()
            if count > 0:
                return tuple(f"cuda:{index}" for index in range(count))
            return ("cpu",)
        resolved = (devices,)
    else:
        try:
            resolved = tuple(devices)
        except TypeError as exc:
            raise TypeError("devices must be a string or iterable of strings.") from exc
    if not resolved:
        raise ValueError("devices must not be empty.")
    for device in resolved:
        if not isinstance(device, str):
            raise TypeError("devices must contain strings.")
        if not device:
            raise ValueError("devices must not contain empty strings.")
    return resolved


def _cuda_device_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()
