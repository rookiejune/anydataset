from __future__ import annotations

import tempfile
import unittest
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client
from pathlib import Path

from anydataset.provider_service import (
    ProviderServer,
    RemoteFilterPredicate,
    _ProviderCommand,
    _ProviderRequest,
    _serve_connection,
)


def _raise_on_start(_device: str):
    raise RuntimeError("provider failed during startup")


class _EchoProvider:
    def __call__(self, value):
        return value


def _echo_provider(_device: str):
    return _EchoProvider()


class _BrokenPipeConnection:
    def recv(self):
        return _ProviderRequest(_ProviderCommand.CALL, "value")

    def send(self, _response):
        raise BrokenPipeError


class ProviderServerTest(unittest.TestCase):
    def test_start_failure_clears_process_handle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            address = Path(tmpdir) / "provider.sock"
            server = ProviderServer(
                address=address,
                provider_factory=_raise_on_start,
                device="cpu",
                startup_timeout=5.0,
            )

            with self.assertRaises(RuntimeError):
                server.start()

            self.assertIsNone(server._process)
            self.assertFalse(address.exists())

    def test_client_receive_failures_do_not_stop_server(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            address = Path(tmpdir) / "provider.sock"
            server = ProviderServer(
                address=address,
                provider_factory=_echo_provider,
                device="cpu",
            )

            with server:
                disconnected = Client(str(address))
                disconnected.close()
                malformed = Client(str(address))
                malformed.send_bytes(b"not-a-pickle")
                malformed.close()

                predicate = RemoteFilterPredicate(address)
                self.assertEqual(predicate("still-running"), "still-running")

    def test_authentication_failure_does_not_stop_server(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            address = Path(tmpdir) / "provider.sock"
            server = ProviderServer(
                address=address,
                provider_factory=_echo_provider,
                device="cpu",
                authkey=b"correct-key",
            )

            with server:
                with self.assertRaises(AuthenticationError):
                    Client(str(address), authkey=b"wrong-key")

                predicate = RemoteFilterPredicate(address, authkey=b"correct-key")
                self.assertEqual(predicate("still-running"), "still-running")

    def test_broken_response_connection_is_isolated(self):
        self.assertFalse(
            _serve_connection(_EchoProvider(), _BrokenPipeConnection())
        )


if __name__ == "__main__":
    unittest.main()
