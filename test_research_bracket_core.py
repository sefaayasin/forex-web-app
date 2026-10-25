import unittest

import numpy as np
import pandas as pd

from research_bracket_core import bracket_trades, summarize


def fills_from(rows, start="2024-01-02 08:00"):
    index = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    rows = np.asarray(rows, float)
    return {"o": rows[:, 0], "h": rows[:, 1], "l": rows[:, 2], "c": rows[:, 3], "t": index.asi8,
        "bad": np.zeros(len(rows) + 1, dtype=np.int64), "index": index}


class BracketCoreTests(unittest.TestCase):
    def run_one(self, rows, side=1, stop=.001, cost=2., hours=1):
        fills = fills_from(rows)
        when = fills["index"][:1]
        return bracket_trades("EURUSD", fills, when, np.array([side]), np.array([stop]),
            when + pd.Timedelta(hours=hours), np.array([cost]))

    def test_net_target_pays_thirty_after_cost(self):
        # Stop 10 pips, cost 2 -> gross target 2 + 0.3 * 12 = 5.6 pips.
        trades = self.run_one([[1., 1.0006, .9999, 1.0005]] + [[1.0005] * 4] * 11)
        self.assertEqual(trades.reason.iloc[0], "target")
        self.assertAlmostEqual(trades.target_pips.iloc[0], 5.6)
        self.assertAlmostEqual(trades.pnl_usd.iloc[0], 30)
        self.assertLess(trades.pnl_stress_usd.iloc[0], 30)

    def test_stop_wins_when_both_levels_touch_in_one_bar(self):
        trades = self.run_one([[1., 1.002, .998, 1.]] + [[1.] * 4] * 11)
        self.assertEqual(trades.reason.iloc[0], "stop")
        self.assertAlmostEqual(trades.pnl_usd.iloc[0], -100)

    def test_adverse_gap_fills_at_open_beyond_budget(self):
        trades = self.run_one([[1., 1.0001, .9999, 1.], [.998] * 4] + [[.998] * 4] * 10)
        self.assertEqual(trades.reason.iloc[0], "stop_gap")
        self.assertLess(trades.pnl_usd.iloc[0], -100)

    def test_time_exit_uses_last_close_before_deadline(self):
        rows = [[1., 1.0002, .9998, 1.0001]] * 12 + [[1.01] * 4] * 5
        trades = self.run_one(rows, side=-1)
        self.assertEqual(trades.reason.iloc[0], "time")
        self.assertAlmostEqual(trades.gross_pips.iloc[0], -1, places=6)

    def test_one_position_at_a_time(self):
        fills = fills_from([[1., 1.0002, .9998, 1.]] * 30)
        when = fills["index"][[0, 2, 15]]
        trades = bracket_trades("EURUSD", fills, when, np.array([1, 1, -1]), np.full(3, .001),
            when + pd.Timedelta(hours=1), np.full(3, 1.))
        self.assertEqual(list(trades.entry_time), [when[0], when[2]])

    def test_entries_without_a_bar_at_decision_time_are_skipped(self):
        fills = fills_from([[1.] * 4] * 12)
        when = pd.DatetimeIndex([fills["index"][0] + pd.Timedelta(minutes=2)])
        trades = bracket_trades("EURUSD", fills, when, np.array([1]), np.array([.001]),
            when + pd.Timedelta(hours=1), np.array([1.]))
        self.assertTrue(trades.empty)
        self.assertEqual(summarize(trades)["trades"], 0)

    def test_quarantined_bar_inside_path_drops_the_trade(self):
        fills = fills_from([[1., 1.0002, .9998, 1.]] * 12)
        fills["bad"][5:] = 1  # bar 4 is quarantined
        when = fills["index"][:1]
        trades = bracket_trades("EURUSD", fills, when, np.array([1]), np.array([.001]),
            when + pd.Timedelta(hours=1), np.array([1.]))
        self.assertTrue(trades.empty)


if __name__ == "__main__":
    unittest.main()
