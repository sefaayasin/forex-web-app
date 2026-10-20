import unittest

import numpy as np
import pandas as pd

from research_pivot_stops import BACK, PASS, UNRESOLVED, outcomes


def path(highs, lows):
    times = pd.date_range("2026-10-20", periods=len(highs), freq="15min").to_numpy()
    return np.asarray(highs, float), np.asarray(lows, float), times


class OutcomeTests(unittest.TestCase):
    def run_one(self, highs, lows, sign=1, level=1.0, step=0.1):
        high, low, times = path(highs, lows)
        return outcomes(high, low, times, np.array([0]), np.array([level]), np.array([step]), sign)[0]

    def test_upward_touch_that_keeps_going_passes(self):
        self.assertEqual(self.run_one([1.0, 1.05, 1.11], [0.95, 0.98, 1.04]), PASS)

    def test_upward_touch_that_turns_back(self):
        self.assertEqual(self.run_one([1.0, 1.02, 0.95], [0.95, 0.96, 0.89]), BACK)

    def test_pass_on_touch_bar_counts_but_back_on_touch_bar_is_ignored(self):
        self.assertEqual(self.run_one([1.12, 1.0], [0.85, 0.99]), PASS)
        self.assertEqual(self.run_one([1.0, 1.0], [0.85, 0.99]), UNRESOLVED)

    def test_both_sides_on_one_later_bar_is_unresolved(self):
        self.assertEqual(self.run_one([1.0, 1.2], [0.95, 0.8]), UNRESOLVED)

    def test_downward_touch_mirrors(self):
        self.assertEqual(self.run_one([1.05, 1.02, 1.0], [1.0, 0.95, 0.89], sign=-1), PASS)
        self.assertEqual(self.run_one([1.05, 1.11], [1.0, 1.0], sign=-1), BACK)

    def test_moves_after_four_hours_are_ignored(self):
        highs = [1.0] * 16 + [1.5]
        self.assertEqual(self.run_one(highs, [0.95] * 17), UNRESOLVED)
        highs = [1.0] * 15 + [1.5]
        self.assertEqual(self.run_one(highs, [0.95] * 16), PASS)


if __name__ == "__main__":
    unittest.main()
