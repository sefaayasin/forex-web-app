import unittest

import numpy as np
import pandas as pd

from research_pivot_plan import simulate

LONG = {"side": "LONG", "entry": 1.00, "stop": 0.99, "target": 1.02}


def run(highs, lows, closes=None, plan=LONG):
    n = len(highs)
    times = pd.date_range("2026-10-20", periods=n, freq="15min").to_numpy()
    closes = np.asarray(closes if closes is not None else lows, float)
    return simulate(np.asarray(highs, float), np.asarray(lows, float), closes, times, np.full(n, n - 1), 0, plan)


class SimulateTests(unittest.TestCase):
    def test_fill_then_target(self):
        self.assertEqual(run([1.01, 1.005, 1.03], [1.005, 0.999, 1.00])["kind"], "target")

    def test_never_filled(self):
        self.assertIsNone(run([1.02, 1.03], [1.01, 1.02]))

    def test_target_on_fill_bar_does_not_count_but_stop_does(self):
        self.assertEqual(run([1.01, 1.05], [1.01, 0.995], closes=[1.01, 1.0])["kind"], "day_end")
        self.assertEqual(run([1.01, 1.05], [1.01, 0.98])["kind"], "stop")

    def test_bar_with_both_counts_as_stop(self):
        self.assertEqual(run([1.01, 1.00, 1.03], [1.01, 1.00, 0.98])["kind"], "stop")

    def test_day_end_exit_at_close(self):
        result = run([1.01, 1.00, 1.01], [1.01, 0.995, 1.00], closes=[1.01, 1.0, 1.008])
        self.assertEqual((result["kind"], result["price"]), ("day_end", 1.008))

    def test_short_mirrors(self):
        short = {"side": "SHORT", "entry": 1.00, "stop": 1.01, "target": 0.98}
        self.assertEqual(run([0.99, 1.001, 0.99], [0.98, 0.99, 0.975], plan=short)["kind"], "target")


if __name__ == "__main__":
    unittest.main()
