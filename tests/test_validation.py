from __future__ import annotations

import unittest

from anydataset._devices import resolve_devices
from anydataset._validation import optional_positive_float


class ValidationTest(unittest.TestCase):
    def test_devices_require_non_empty_strings(self):
        with self.assertRaisesRegex(TypeError, "contain strings"):
            resolve_devices(("cpu", 1))
        with self.assertRaisesRegex(ValueError, "empty strings"):
            resolve_devices(("cpu", ""))
        with self.assertRaisesRegex(TypeError, "string or iterable"):
            resolve_devices(1)

    def test_optional_positive_float_rejects_non_finite_values(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    optional_positive_float("timeout", value)

    def test_optional_positive_float_rejects_huge_integers_cleanly(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            optional_positive_float("timeout", 10**400)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            optional_positive_float("timeout", -(10**400))


if __name__ == "__main__":
    unittest.main()
