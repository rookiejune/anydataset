from __future__ import annotations

from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

from ._protocol import (
    ProviderAddress,
    _ProviderCommand,
    _ProviderRequest,
    _ProviderResponse,
)


def connection_address(address: ProviderAddress) -> str | tuple[str, int]:
    if isinstance(address, Path):
        return str(address)
    return address


def request(
    address: ProviderAddress,
    authkey: bytes | None,
    command: _ProviderCommand,
    payload: Any,
) -> _ProviderResponse:
    conn = Client(connection_address(address), authkey=_authkey_value(authkey))
    try:
        conn.send(_ProviderRequest(command=command, payload=payload))
        response = conn.recv()
    finally:
        conn.close()
    if not isinstance(response, _ProviderResponse):
        raise TypeError("Provider server returned an invalid response.")
    return response


def _authkey_value(authkey: bytes | None) -> bytes | None:
    return authkey
