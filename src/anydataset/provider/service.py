from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .._runtime.logging import write_info, write_warning
from .._runtime.parallel import (
    StartMethod,
    multiprocessing_context,
    validate_process_value,
    validate_start_method,
)
from .._validation import optional_positive_float, positive_float
from ..types.item import View
from ..view import BatchOutput, ViewMap
from ._protocol import (
    ProviderAddress,
    _ProviderCommand,
    _ProviderError,
    _ProviderResponse,
    _ProviderServerConfig,
)
from ._server import serve_provider, unlink_address
from ._transport import request

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
        return _value(request(self.address, self.authkey, _ProviderCommand.CALL, views))

    def call_batch(self, batch: Batch) -> BatchOutput:
        return _value(
            request(self.address, self.authkey, _ProviderCommand.CALL_BATCH, batch)
        )

    def close(self) -> None:
        _value(request(self.address, self.authkey, _ProviderCommand.CLOSE, None))


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
        return _value(request(self.address, self.authkey, _ProviderCommand.CALL, sample))

    def close(self) -> None:
        _value(request(self.address, self.authkey, _ProviderCommand.CLOSE, None))


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
        unlink_address(self.address, self.authkey)
        context = multiprocessing_context(self.start_method)
        config = _ProviderServerConfig(
            address=self.address,
            device=self.device,
            authkey=self.authkey,
        )
        write_info(
            "provider",
            "starting provider server: "
            f"device={self.device!r} address={self.address!r} "
            f"start_method={self.start_method!r}",
            event="provider_server_starting",
            fields=self._log_fields(),
        )
        process = context.Process(
            target=serve_provider,
            args=(config, self.provider_factory),
            name=f"anydataset-provider-{self.device}",
        )
        self._process = process
        process.start()
        try:
            self._wait_ready()
        except Exception as exc:
            write_warning(
                "provider",
                "provider server failed during startup: "
                f"device={self.device!r} address={self.address!r} "
                f"exitcode={process.exitcode!r}",
                event="provider_server_failed",
                fields={
                    **self._log_fields(process=process),
                    "exitcode": process.exitcode,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            self._cleanup_failed_start()
            raise
        write_info(
            "provider",
            "provider server ready: "
            f"device={self.device!r} address={self.address!r} "
            f"pid={process.pid!r}",
            event="provider_server_ready",
            fields=self._log_fields(process=process),
        )
        return self

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        write_info(
            "provider",
            "stopping provider server: "
            f"device={self.device!r} address={self.address!r} "
            f"pid={process.pid!r}",
            event="provider_server_stopping",
            fields=self._log_fields(process=process),
        )
        try:
            _value(request(self.address, self.authkey, _ProviderCommand.CLOSE, None))
        except (ConnectionError, EOFError, FileNotFoundError, OSError):
            pass
        process.join(self.shutdown_timeout)
        if process.is_alive():
            write_warning(
                "provider",
                "terminating provider server after shutdown timeout: "
                f"device={self.device!r} address={self.address!r} "
                f"pid={process.pid!r}",
                event="provider_server_terminated",
                fields=self._log_fields(process=process),
            )
            process.terminate()
            process.join(self.shutdown_timeout)
        if process.is_alive():
            write_warning(
                "provider",
                "killing provider server after terminate timeout: "
                f"device={self.device!r} address={self.address!r} "
                f"pid={process.pid!r}",
                event="provider_server_killed",
                fields=self._log_fields(process=process),
            )
            process.kill()
            process.join(self.shutdown_timeout)
        unlink_address(self.address, self.authkey)
        write_info(
            "provider",
            "provider server stopped: "
            f"device={self.device!r} address={self.address!r} "
            f"exitcode={process.exitcode!r}",
            event="provider_server_stopped",
            fields={
                **self._log_fields(process=process),
                "exitcode": process.exitcode,
            },
        )
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
                _value(request(self.address, self.authkey, _ProviderCommand.PING, None))
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
            process.join(self.shutdown_timeout)
        if process.is_alive():
            process.kill()
            process.join(self.shutdown_timeout)
        unlink_address(self.address, self.authkey)
        self._process = None

    def _log_fields(self, *, process: BaseProcess | None = None) -> dict[str, object]:
        return {
            "address": self.address,
            "device": self.device,
            "pid": None if process is None else process.pid,
            "start_method": self.start_method,
            "startup_timeout": self.startup_timeout,
            "shutdown_timeout": self.shutdown_timeout,
        }


class RemoteProviderError(RuntimeError):
    def __init__(self, error: _ProviderError) -> None:
        super().__init__(
            f"Remote provider raised {error.type_name}: {error.message}\n"
            f"{error.traceback}"
        )


def _value(response: _ProviderResponse) -> Any:
    if response.error is not None:
        raise RemoteProviderError(response.error)
    return response.value
