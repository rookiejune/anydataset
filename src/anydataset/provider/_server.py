from __future__ import annotations

import os
from collections.abc import Callable
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any

from ._protocol import (
    ProviderAddress,
    _ProviderServerConfig,
    _accept_connection,
    _serve_connection,
)
from ._transport import connection_address

ProviderFactory = Callable[[str], Any]


def serve_provider(
    config: _ProviderServerConfig,
    provider_factory: ProviderFactory,
) -> None:
    address = connection_address(config.address)
    unlink_address(config.address)
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
        unlink_address(config.address)


def unlink_address(address: ProviderAddress) -> None:
    connection = connection_address(address)
    if not isinstance(connection, str):
        return
    path = Path(connection)
    if not path.exists():
        return
    if socket_in_use(path):
        raise RuntimeError(f"Provider socket is already in use: {path}")
    try:
        os.unlink(connection)
    except FileNotFoundError:
        pass


def socket_in_use(path: Path) -> bool:
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
