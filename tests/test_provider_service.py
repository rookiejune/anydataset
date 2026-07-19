from __future__ import annotations

import tempfile
import unittest
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client
from pathlib import Path

from anydataset.provider_service import (
    ProviderServer,
    RemoteFilterPredicate,
    RemoteProviderError,
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


class _UnpicklableValue:
    def __reduce__(self):
        raise TypeError("result cannot be pickled")


class _UnpicklableResultProvider:
    def __call__(self, value):
        if value == "unpicklable":
            return _UnpicklableValue()
        return value


def _unpicklable_result_provider(_device: str):
    return _UnpicklableResultProvider()


class _BrokenPipeConnection:
    def recv(self):
        return _ProviderRequest(_ProviderCommand.CALL, "value")

    def send(self, _response):
        raise BrokenPipeError


class ProviderServerTest(unittest.TestCase):
    def test_rejects_invalid_runtime_options(self):
        cases = (
            ({"start_method": "invalid"}, ValueError, "start_method"),
            ({"startup_timeout": float("nan")}, ValueError, "finite"),
            ({"startup_timeout": 0}, ValueError, "positive"),
            ({"shutdown_timeout": float("inf")}, ValueError, "finite"),
            ({"shutdown_timeout": None}, TypeError, "number"),
        )

        for kwargs, error, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(error, message):
                    ProviderServer(
                        address="unused.sock",
                        provider_factory=_echo_provider,
                        device="cpu",
                        **kwargs,
                    )

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

    def test_unsupported_command_does_not_stop_server(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            address = Path(tmpdir) / "provider.sock"
            server = ProviderServer(
                address=address,
                provider_factory=_echo_provider,
                device="cpu",
            )

            with server:
                malformed = Client(str(address))
                malformed.send(_ProviderRequest("unsupported", None))
                response = malformed.recv()
                malformed.close()

                self.assertIsNotNone(response.error)
                predicate = RemoteFilterPredicate(address)
                self.assertEqual(predicate("still-running"), "still-running")

    def test_unpicklable_response_does_not_stop_server(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            address = Path(tmpdir) / "provider.sock"
            server = ProviderServer(
                address=address,
                provider_factory=_unpicklable_result_provider,
                device="cpu",
            )

            with server:
                predicate = RemoteFilterPredicate(address)
                with self.assertRaisesRegex(RemoteProviderError, "cannot be pickled"):
                    predicate("unpicklable")
                self.assertEqual(predicate("still-running"), "still-running")


if __name__ == "__main__":
    unittest.main()
