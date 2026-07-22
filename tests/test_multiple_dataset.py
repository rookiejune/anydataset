from __future__ import annotations

import unittest

from anydataset.dataset.multiple import WeightedRandomStrategy
from anydataset.dataset.multiple import _cumulative_weights


class WeightedRandomStrategyTest(unittest.TestCase):
    def test_rejects_non_finite_weights(self):
        datasets = (({"value": 0},), ({"value": 1},))

        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                strategy = WeightedRandomStrategy(weights=(value, 1.0))
                with self.assertRaisesRegex(ValueError, "weights must be finite"):
                    list(strategy.iter(datasets))

    def test_finite_large_weights_do_not_overflow_cumulative_weights(self):
        self.assertEqual(_cumulative_weights((1e308, 1e308)), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
