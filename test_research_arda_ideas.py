import unittest

import numpy as np
import pandas as pd

from research_arda_ideas import _paths, server_bars, struct_dir


def path_result(rows, side=1, close=1.0, sl=.99, tp=1.02, minutes=60):
    index = pd.date_range("2024-01-02 08:00", periods=len(rows), freq="5min", tz="UTC")
    rows = np.asarray(rows, float)
    deadline = (index[0] + pd.Timedelta(minutes=minutes)).value
    full, part = _paths(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], index.asi8, np.array([0]),
        np.array([side]), np.array([close]), np.array([sl]), np.array([tp]), np.array([abs(close - sl)]),
        np.array([deadline]))
    return full[0], part[0]


class ServerClockTests(unittest.TestCase):
    def test_daily_bar_closes_at_new_york_five_pm(self):
        index = pd.date_range("2024-07-01 20:00", periods=8, freq="15min", tz="UTC")  # 16:00-17:45 New York
        m15 = pd.DataFrame({"Open": 1., "High": 1., "Low": 1., "Close": np.arange(8.)}, index=index)
        days = server_bars(m15, "D")
        self.assertEqual(len(days), 2)
        self.assertEqual(days.Close.iloc[0], 3.)  # last bar before 17:00 New York
        self.assertEqual(days.avail.iloc[0], pd.Timestamp("2024-07-01 21:00", tz="UTC"))


class StructureTests(unittest.TestCase):
    def test_close_above_confirmed_pivot_high_turns_structure_up(self):
        h = np.array([1, 2, 3, 5, 3, 2, 1, 2, 3, 4, 6.])
        l = h - .5
        c = h - .2
        out = struct_dir(h, l, c, 3)
        self.assertEqual(out[9], 0)  # pivot 5 confirmed at bar 6, no close above it yet
        self.assertEqual(out[10], 1)


class PathTests(unittest.TestCase):
    def test_target_hits_full_and_partial(self):
        full, part = path_result([[1., 1.001, .999, 1.]] + [[1.01, 1.011, 1.009, 1.01]] + [[1.02, 1.021, 1.019, 1.02]] * 3)
        self.assertAlmostEqual(full, 2.0)
        self.assertAlmostEqual(part, 1.5)  # half at +1R, half at +2R

    def test_partial_returns_to_entry_after_first_target(self):
        full, part = path_result([[1., 1.001, .999, 1.], [1.01, 1.011, 1.009, 1.01], [1.005, 1.006, .995, .996]]
            + [[.99, .991, .989, .99]] * 3)
        self.assertAlmostEqual(full, -1.0)
        self.assertAlmostEqual(part, 0.5)

    def test_stop_wins_in_the_same_bar(self):
        full, part = path_result([[1., 1.03, .98, 1.]] + [[1.] * 4] * 3)
        self.assertAlmostEqual(full, -1.0)
        self.assertAlmostEqual(part, -1.0)


if __name__ == "__main__":
    unittest.main()
