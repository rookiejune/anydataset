from __future__ import annotations

import unittest

from anydataset import Runtime


class RuntimeTest(unittest.TestCase):
    def test_auto_start_methods_keep_local_spawn(self):
        runtime = Runtime()

        self.assertEqual(runtime.reader_worker_start_method, "spawn")
        self.assertEqual(runtime.writer_worker_start_method, "spawn")

    def test_auto_start_methods_use_server_fork(self):
        runtime = Runtime(server_start_method="spawn")

        self.assertEqual(runtime.reader_worker_start_method, "fork")
        self.assertEqual(runtime.writer_worker_start_method, "fork")

    def test_explicit_start_methods_override_auto(self):
        runtime = Runtime(
            server_start_method="spawn",
            reader_start_method="spawn",
            writer_start_method="spawn",
        )

        self.assertEqual(runtime.reader_worker_start_method, "spawn")
        self.assertEqual(runtime.writer_worker_start_method, "spawn")


if __name__ == "__main__":
    unittest.main()
