from __future__ import annotations

import hashlib
import json
import marshal
from collections.abc import Mapping, Sequence
from enum import Enum
from functools import partial
from pathlib import Path
from types import BuiltinFunctionType, CellType, FunctionType, MethodType

import torch


def optional_semantic_id(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return value


def metadata_value(
    value: object,
    active: set[int] | None = None,
) -> object:
    if isinstance(value, Enum):
        return {
            "type": type_id(value),
            "value": metadata_value(value.value, active),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return {"type": type_id(value), "value": str(value)}
    if isinstance(value, (bytes, bytearray)):
        payload = bytes(value)
        return {
            "type": type_id(value),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, torch.Tensor):
        return tensor_value(value)
    if isinstance(value, range):
        return {
            "type": type_id(value),
            "start": value.start,
            "stop": value.stop,
            "step": value.step,
        }
    if active is None:
        active = set()
    if callable(value):
        return callable_identity(value, active)

    marker = id(value)
    if marker in active:
        return {"type": type_id(value), "recursive": True}
    active.add(marker)
    try:
        return metadata_object(value, active)
    finally:
        active.remove(marker)


def callable_id(value: object) -> object:
    return callable_identity(value, set())


def metadata_object(value: object, active: set[int]) -> object:
    if isinstance(value, Mapping):
        items = [
            [
                metadata_value(key, active),
                metadata_value(item, active),
            ]
            for key, item in value.items()
        ]
        items.sort(key=metadata_sort_key)
        return {"type": type_id(value), "items": items}
    if isinstance(value, (set, frozenset)):
        items = [metadata_value(item, active) for item in value]
        return {
            "type": type_id(value),
            "items": sorted(items, key=metadata_sort_key),
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return {
            "type": type_id(value),
            "items": [metadata_value(item, active) for item in value],
        }
    state = instance_state(value)
    if state is not None:
        return {
            "type": type_id(value),
            "state": metadata_digest(state, active),
        }
    if type(value).__repr__ is object.__repr__:
        return type_id(value)
    return {"type": type_id(value), "value": repr(value)}


def closure_value(cell: CellType) -> object:
    try:
        return cell.cell_contents
    except ValueError:
        return "<empty>"


def callable_identity(value: object, active: set[int]) -> object:
    marker = id(value)
    if marker in active:
        return {"type": type_id(value), "recursive": True}
    active.add(marker)
    try:
        return callable_value(value, active)
    finally:
        active.remove(marker)


def callable_value(value: object, active: set[int]) -> object:
    if isinstance(value, partial):
        return {
            "type": "functools.partial",
            "function": callable_identity(value.func, active),
            "args": metadata_digest(value.args, active),
            "keywords": metadata_digest(value.keywords or {}, active),
        }
    if isinstance(value, MethodType):
        return {
            "function": callable_identity(value.__func__, active),
            "owner": metadata_digest(value.__self__, active),
        }
    if isinstance(value, (FunctionType, BuiltinFunctionType)):
        identity: dict[str, object] = {
            "function": f"{value.__module__}.{value.__qualname__}",
        }
        if isinstance(value, FunctionType):
            identity["code"] = hashlib.sha256(
                marshal.dumps(value.__code__)
            ).hexdigest()[:16]
            identity["defaults"] = metadata_digest(
                value.__defaults__ or (),
                active,
            )
            identity["kwdefaults"] = metadata_digest(
                value.__kwdefaults__ or {},
                active,
            )
            identity["closure"] = [
                metadata_digest(closure_value(cell), active)
                for cell in (value.__closure__ or ())
            ]
        return identity

    if isinstance(value, type):
        identity = {"class": f"{value.__module__}.{value.__qualname__}"}
        for name in ("__init__", "__call__"):
            member = value.__dict__.get(name)
            if callable(member):
                identity[name] = callable_identity(member, active)
        return identity

    value_type = type_id(value)
    identity = {
        "type": value_type,
        "call": callable_identity(type(value).__call__, active),
    }
    state = instance_state(value)
    if state is not None:
        identity["state"] = metadata_digest(state, active)
        return identity
    if type(value).__repr__ is object.__repr__:
        return identity
    identity["value"] = repr(value)
    return identity


def instance_state(value: object) -> dict[str, object] | None:
    try:
        state = dict(vars(value))
    except TypeError:
        state = {}
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in {"__dict__", "__weakref__"} or name in state:
                continue
            try:
                state[name] = getattr(value, name)
            except AttributeError:
                continue
    return state or None


def type_id(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def metadata_digest(value: object, active: set[int]) -> str:
    payload = json.dumps(
        metadata_value(value, active),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def metadata_sort_key(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def tensor_value(value: torch.Tensor) -> dict[str, object]:
    metadata: dict[str, object] = {
        "type": type_id(value),
        "dtype": str(value.dtype),
        "layout": str(value.layout),
        "shape": list(value.shape),
    }
    if value.device.type == "meta":
        metadata["content"] = "meta"
        return metadata
    if value.is_quantized:
        metadata["sha256"] = dense_tensor_digest(value.int_repr())
        metadata["qscheme"] = str(value.qscheme())
        if value.qscheme() in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            metadata["scale"] = value.q_scale()
            metadata["zero_point"] = value.q_zero_point()
        else:
            metadata["axis"] = value.q_per_channel_axis()
            metadata["scales"] = dense_tensor_digest(
                value.q_per_channel_scales()
            )
            metadata["zero_points"] = dense_tensor_digest(
                value.q_per_channel_zero_points()
            )
        return metadata
    if value.layout == torch.strided:
        metadata["sha256"] = dense_tensor_digest(value)
        return metadata
    try:
        sparse = value.detach().cpu().to_sparse_coo().coalesce()
    except (NotImplementedError, RuntimeError) as exc:
        raise TypeError(
            "Callable identity cannot fingerprint this Tensor layout."
        ) from exc
    metadata["indices"] = dense_tensor_digest(sparse.indices())
    metadata["values"] = dense_tensor_digest(sparse.values())
    return metadata


def dense_tensor_digest(value: torch.Tensor) -> str:
    try:
        tensor = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8)
    except (NotImplementedError, RuntimeError) as exc:
        raise TypeError(
            "Callable identity cannot fingerprint this Tensor dtype."
        ) from exc
    digest = hashlib.sha256()
    chunk_size = 1024 * 1024
    for start in range(0, raw.numel(), chunk_size):
        digest.update(raw[start : start + chunk_size].numpy().tobytes())
    return digest.hexdigest()
