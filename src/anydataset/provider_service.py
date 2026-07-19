from __future__ import annotations

import gc
import os
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import auto
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import torch

from ._compat import StrEnum
from ._parallel import (
    StartMethod,
    multiprocessing_context,
    validate_process_value,
    validate_start_method,
)
from ._validation import optional_positive_float, positive_float
from .types.item import View
from .view import BatchOutput, ViewMap

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from .dataset.collate import Batch

ProviderAddress = Union[str, Path, tuple[str, int]]
ProviderFactory = Callable[[str], Any]


@dataclass(frozen=True)
class RemoteProvider:
    output: View
    address: ProviderAddress
    authkey: bytes | None = None

    def __call__(self, views: ViewMap) -> Any:
        return _request(self.address, self.authkey, _ProviderCommand.CALL, views)

    def call_batch(self, batch: Batch) -> BatchOutput:
        return _request(self.address, self.authkey, _ProviderCommand.CALL_BATCH, batch)

    def close(self) -> None:
        _request(self.address, self.authkey, _ProviderCommand.CLOSE, None)


@dataclass(frozen=True)
class RemoteProviderFactory:
    output: View
    addresses: Mapping[str, ProviderAddress]
    authkey: bytes | None = None

    def __call__(self, device: str) -> RemoteProvider:
        try:
            address = self.addresses[device]
        except KeyError as exc:
            raise KeyError(
                f"No remote provider address for device {device!r}."
            ) from exc
        return RemoteProvider(
            output=self.output,
            address=address,
            authkey=self.authkey,
        )


@dataclass(frozen=True)
class RemoteFilterPredicate:
    address: ProviderAddress
    authkey: bytes | None = None

    def __call__(self, sample: Any) -> Any:
        return _request(self.address, self.authkey, _ProviderCommand.CALL, sample)

    def close(self) -> None:
        _request(self.address, self.authkey, _ProviderCommand.CLOSE, None)


@dataclass(frozen=True)
class RemoteFilterFactory:
    addresses: Mapping[str, ProviderAddress]
    authkey: bytes | None = None
    device_env: str = "ANYDATASET_FILTER_DEVICE"

    def __call__(self) -> RemoteFilterPredicate:
        device = os.environ.get(self.device_env)
        if device is None:
            if len(self.addresses) != 1:
                raise RuntimeError(f"{self.device_env} is required for remote filter.")
            address = next(iter(self.addresses.values()))
        else:
            try:
                address = self.addresses[device]
            except KeyError as exc:
                raise KeyError(
                    f"No remote filter address for device {device!r}."
                ) from exc
        return RemoteFilterPredicate(address=address, authkey=self.authkey)


@dataclass
class ProviderServer:
    address: ProviderAddress
    provider_factory: ProviderFactory
    device: str
    authkey: bytes | None = None
    start_method: StartMethod = "spawn"
    startup_timeout: float | None = 120.0
    shutdown_timeout: float = 10.0
    _process: BaseProcess | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_start_method("start_method", self.start_method)
        self.startup_timeout = optional_positive_float(
            "startup_timeout",
            self.startup_timeout,
        )
        self.shutdown_timeout = positive_float(
            "shutdown_timeout",
            self.shutdown_timeout,
        )

    def start(self) -> ProviderServer:
        if self._process is not None:
            raise RuntimeError("Provider server is already started.")
        validate_process_value(
            "provider_factory",
            self.provider_factory,
            context="provider server",
            start_method=self.start_method,
        )
        address = _address(self.address)
        _unlink_address(address)
        context = multiprocessing_context(self.start_method)
        config = _ProviderServerConfig(
            address=self.address,
            device=self.device,
            authkey=self.authkey,
        )
        self._process = context.Process(
            target=_serve_provider,
            args=(config, self.provider_factory),
            name=f"anydataset-provider-{self.device}",
        )
        self._process.start()
        try:
            self._wait_ready()
        except Exception:
            self._cleanup_failed_start()
            raise
        return self

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            _request(self.address, self.authkey, _ProviderCommand.CLOSE, None)
        except (ConnectionError, EOFError, FileNotFoundError, OSError):
            pass
        process.join(self.shutdown_timeout)
        if process.is_alive():
            process.terminate()
            process.join()
        self._process = None

    def __enter__(self) -> ProviderServer:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _wait_ready(self) -> None:
        if self._process is None:
            raise RuntimeError("Provider server process has not been created.")
        deadline = (
            None
            if self.startup_timeout is None
            else time.monotonic() + self.startup_timeout
        )
        while True:
            if self._process.exitcode is not None:
                raise RuntimeError(
                    f"Provider server exited during startup: {self._process.exitcode}."
                )
            try:
                _request(self.address, self.authkey, _ProviderCommand.PING, None)
                return
            except (ConnectionError, EOFError, FileNotFoundError, OSError):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(
                        "Provider server did not become ready."
                    ) from None
                time.sleep(0.05)

    def _cleanup_failed_start(self) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
        process.join()
        _unlink_address(_address(self.address))
        self._process = None


class RemoteProviderError(RuntimeError):
    def __init__(self, error: _ProviderError) -> None:
        super().__init__(
            f"Remote provider raised {error.type_name}: {error.message}\n"
            f"{error.traceback}"
        )


class _ProviderCommand(StrEnum):
    PING = auto()
    CALL = auto()
    CALL_BATCH = auto()
    CLOSE = auto()


@dataclass(frozen=True)
class _ProviderServerConfig:
    address: ProviderAddress
    device: str
    authkey: bytes | None


@dataclass(frozen=True)
class _ProviderRequest:
    command: _ProviderCommand
    payload: Any


@dataclass(frozen=True)
class _ProviderError:
    type_name: str
    message: str
    traceback: str


@dataclass(frozen=True)
class _ProviderResponse:
    value: Any = None
    error: _ProviderError | None = None


def _serve_provider(
    config: _ProviderServerConfig,
    provider_factory: ProviderFactory,
) -> None:
    address = _address(config.address)
    _unlink_address(address)
    provider = provider_factory(config.device)
    listener = Listener(address, authkey=config.authkey)
    try:
        while True:
            conn = _accept_connection(listener)
            if conn is None:
                continue
            try:
                should_close = _serve_connection(provider, conn)
            finally:
                conn.close()
            if should_close:
                return
    finally:
        listener.close()
        _unlink_address(address)


def _accept_connection(listener: Listener):
    try:
        return listener.accept()
    except (AuthenticationError, ConnectionError, EOFError):
        return None


def _serve_connection(provider: Any, conn: Any) -> bool:
    try:
        request = conn.recv()
    except Exception:
        return False
    response = _handle_request(provider, request)
    try:
        conn.send(response)
    except Exception as exc:
        try:
            conn.send(_ProviderResponse(error=_provider_error(exc)))
        except Exception:
            pass
        return False
    return (
        isinstance(request, _ProviderRequest)
        and request.command is _ProviderCommand.CLOSE
    )


def _handle_request(provider: Any, request: object) -> _ProviderResponse:
    try:
        if not isinstance(request, _ProviderRequest):
            raise TypeError("Provider server received an invalid request.")
        if request.command is _ProviderCommand.PING:
            return _ProviderResponse()
        if request.command is _ProviderCommand.CALL:
            return _ProviderResponse(value=provider(request.payload))
        if request.command is _ProviderCommand.CALL_BATCH:
            return _ProviderResponse(value=provider.call_batch(request.payload))
        if request.command is _ProviderCommand.CLOSE:
            return _ProviderResponse()
        raise TypeError(f"Unsupported provider command: {request.command!r}.")
    except Exception as exc:
        error = _provider_error(exc)
        try:
            _clear_cuda_cache()
        except Exception as cleanup_exc:
            cleanup = _provider_error(cleanup_exc)
            error = _ProviderError(
                type_name=error.type_name,
                message=error.message,
                traceback=(
                    f"{error.traceback}\n"
                    f"Provider cleanup raised {cleanup.type_name}: {cleanup.message}\n"
                    f"{cleanup.traceback}"
                ),
            )
        return _ProviderResponse(error=error)


def _provider_error(exc: Exception) -> _ProviderError:
    return _ProviderError(
        type_name=type(exc).__name__,
        message=str(exc),
        traceback=traceback.format_exc(),
    )


def _request(
    address: ProviderAddress,
    authkey: bytes | None,
    command: _ProviderCommand,
    payload: Any,
) -> Any:
    conn = Client(_address(address), authkey=authkey)
    try:
        conn.send(_ProviderRequest(command=command, payload=payload))
        response = conn.recv()
    finally:
        conn.close()
    if not isinstance(response, _ProviderResponse):
        raise TypeError("Provider server returned an invalid response.")
    if response.error is not None:
        raise RemoteProviderError(response.error)
    return response.value


def _address(address: ProviderAddress) -> str | tuple[str, int]:
    if isinstance(address, Path):
        return str(address)
    return address


def _unlink_address(address: str | tuple[str, int]) -> None:
    if not isinstance(address, str):
        return
    try:
        os.unlink(address)
    except FileNotFoundError:
        pass


def _clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
