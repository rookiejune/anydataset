"""Compatibility export for the provider process server."""

from __future__ import annotations

from .provider import service as _service
from .provider.service import (
    ProviderServer,
    RemoteFilterFactory,
    RemoteFilterPredicate,
    RemoteProvider,
    RemoteProviderError,
    RemoteProviderFactory,
)

_ProviderCommand = _service._ProviderCommand
_ProviderRequest = _service._ProviderRequest
_serve_connection = _service._serve_connection

__all__ = [
    "ProviderServer",
    "RemoteFilterFactory",
    "RemoteFilterPredicate",
    "RemoteProvider",
    "RemoteProviderError",
    "RemoteProviderFactory",
]
