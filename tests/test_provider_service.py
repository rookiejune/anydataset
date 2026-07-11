from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from anydataset.provider_service import ProviderServer


def _raise_on_start(_device: str):
    raise RuntimeError("provider failed during startup")


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


if __name__ == "__main__":
    unittest.main()
