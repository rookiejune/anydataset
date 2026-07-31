from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._runtime.parallel import (
    StartMethod,
    multiprocessing_context,
    validate_process_value,
    validate_start_method,
)
from .._validation import optional_positive_float, positive_float
from ..types.item import View
from ..view import BatchOutput, ViewMap
from ._service_protocol import (
    ProviderAddress,
    _ProviderCommand,
    _ProviderError,
    _ProviderRequest,
    _ProviderResponse,
    _ProviderServerConfig,
    _accept_connection,
    _serve_connection,
)

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from ..dataset.collate import Batch

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
        process = context.Process(
            target=_serve_provider,
            args=(config, self.provider_factory),
            name=f"anydataset-provider-{self.device}",
        )
        self._process = process
        process.start()
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
    path = Path(address)
    if not path.exists():
        return
    if _socket_in_use(path):
        raise RuntimeError(f"Provider socket is already in use: {path}")
    try:
        os.unlink(address)
    except FileNotFoundError:
        pass


def _socket_in_use(path: Path) -> bool:
    try:
        conn = Client(str(path), authkey=b"anydataset-probe")
    except AuthenticationError:
        return True
    except (ConnectionRefusedError, FileNotFoundError, PermissionError, OSError):
        return False
    try:
        conn.close()
    except Exception:
        pass
    return True
