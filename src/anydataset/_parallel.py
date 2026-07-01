"""Shared spawn/DataLoader runtime for dataset-wide parallel scans.

The module exposes device worker setup, runtime index sharding, and picklability
checks used by higher-level filter and materialization flows. It does not own
filter labels, view generation, cache layout, or store writing rules.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import socket
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, IterableDataset

from ._logging import run_logs_dir, set_run_logs_dir
from ._sharding import runtime_shard, validate_shard

type DatasetFactory = Callable[[], Any]
type StartMethod = Literal["fork", "spawn", "forkserver"]

_DEFAULT_LOADER_PREFETCH_FACTOR = 2


@dataclass(frozen=True)
class DeviceWorker:
    device: str
    rank: int
    world_size: int
    master_addr: str
    master_port: str


class RuntimeIndexedDataset(IterableDataset):
    def __init__(
        self,
        dataset_factory: DatasetFactory,
    ) -> None:
        self.dataset_factory = dataset_factory

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        dataset = self.dataset_factory()
        yield from iter_runtime_indexed(dataset)


def iter_runtime_indexed(dataset: Any) -> Iterator[tuple[int, Any]]:
    iter_indexed = getattr(dataset, "iter_indexed_runtime_shard", None)
    if callable(iter_indexed):
        yield from iter_indexed()
        return

    shard = runtime_shard()
    yield from iter_indexed_shard(dataset, shard.flat_count, shard.flat_index)


def iter_indexed_shard(
    dataset: Any,
    num_shards: int,
    shard_id: int,
) -> Iterator[tuple[int, Any]]:
    validate_shard(num_shards, shard_id)
    iter_indexed = getattr(dataset, "iter_indexed_shard", None)
    if callable(iter_indexed):
        yield from iter_indexed(num_shards, shard_id)
        return

    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        for index in range(shard_id, len(dataset), num_shards):
            yield index, dataset[index]
        return

    raise TypeError("dataset must provide iter_indexed_shard() or be map-style.")


def indexed_loader(
    dataset_factory: DatasetFactory,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None = None,
    start_method: StartMethod = "spawn",
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": indexed_collate,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        validate_process_value(
            "dataset_factory",
            dataset_factory,
            context="DataLoader workers",
            start_method=start_method,
        )
        kwargs["multiprocessing_context"] = multiprocessing_context(start_method)
        kwargs["prefetch_factor"] = (
            prefetch_factor or _DEFAULT_LOADER_PREFETCH_FACTOR
        )
        kwargs["worker_init_fn"] = _RunLogsWorkerInit(run_logs_dir())
    return DataLoader(
        RuntimeIndexedDataset(dataset_factory),
        **kwargs,
    )


def indexed_collate(batch: Sequence[tuple[int, Any]]) -> tuple[tuple[int, Any], ...]:
    return tuple(batch)


@dataclass(frozen=True)
class _RunLogsWorkerInit:
    logs_dir: Path

    def __call__(self, worker_id: int) -> None:
        set_run_logs_dir(self.logs_dir)


def worker_configs(
    devices: Sequence[str],
    *,
    master_addr: str | None = None,
    master_port: str | None = None,
) -> tuple[DeviceWorker, ...]:
    addr = master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = master_port or os.environ.get("MASTER_PORT", free_port())
    return tuple(
        DeviceWorker(
            device=device,
            rank=rank,
            world_size=len(devices),
            master_addr=addr,
            master_port=port,
        )
        for rank, device in enumerate(devices)
    )


def multiprocessing_context(start_method: StartMethod = "spawn"):
    return multiprocessing.get_context(start_method)


def validate_process_value(
    name: str,
    value: object,
    *,
    context: str,
    start_method: StartMethod,
) -> None:
    if start_method == "fork":
        return
    validate_spawn_value(name, value, context=context)


def validate_spawn_value(name: str, value: object, *, context: str) -> None:
    try:
        pickle.dumps(value)
    except Exception as exc:
        raise TypeError(f"{name} must be picklable for {context}.") from exc


def set_worker_environment(
    worker: DeviceWorker,
    *,
    device_env: str,
) -> dict[str, str | None]:
    previous = {
        name: os.environ.get(name)
        for name in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            device_env,
        )
    }
    os.environ["RANK"] = str(worker.rank)
    os.environ["LOCAL_RANK"] = local_rank(worker.device, worker.rank)
    os.environ["WORLD_SIZE"] = str(worker.world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(worker.world_size)
    os.environ["MASTER_ADDR"] = worker.master_addr
    os.environ["MASTER_PORT"] = worker.master_port
    os.environ[device_env] = worker.device
    return previous


def set_single_worker_environment(
    device: str,
    *,
    device_env: str,
) -> dict[str, str | None]:
    return set_worker_environment(
        DeviceWorker(
            device=device,
            rank=0,
            world_size=1,
            master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
            master_port=os.environ.get("MASTER_PORT", free_port()),
        ),
        device_env=device_env,
    )


def restore_environment(previous: Mapping[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def set_torch_device(device: str) -> None:
    cuda = cuda_device(device)
    if cuda is None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    torch.cuda.set_device(cuda)


def cuda_device(device: str) -> int | None:
    prefix = "cuda:"
    if not device.startswith(prefix):
        return None
    index = device.removeprefix(prefix)
    if not index.isdecimal():
        raise ValueError(f"CUDA device must use cuda:<index>: {device}")
    return int(index)


def local_rank(device: str, fallback: int) -> str:
    prefix = "cuda:"
    if device.startswith(prefix):
        return device.removeprefix(prefix)
    return str(fallback)


def free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])
