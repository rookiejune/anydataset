from __future__ import annotations

import traceback
from dataclasses import dataclass
from enum import auto
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Union

from .._compat import StrEnum
from .._runtime.devices import clear_cuda_cache

ProviderAddress = Union[str, Path, tuple[str, int]]


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
            clear_cuda_cache()
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
