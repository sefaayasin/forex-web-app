import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from research_simple_baseline import exit_fill, prepare_bars, simulate, repair_small_envelope_errors

RULES = json.loads((Path(__file__).parent / "research/simple_baseline/protocol.json").read_text(encoding="utf-8"))["rules"]


class BaselineTests(unittest.TestCase):
    def test_small_data_repair_is_bounded_and_preserves_source(self):
        raw = pd.DataFrame({"Open": [1.], "High": [1.001], "Low": [.999], "Close": [.99899]})
        repaired, count = repair_small_envelope_errors(raw)
        self.assertEqual(count, 1)
        self.assertEqual(repaired.Low.iloc[0], .99899)
        self.assertEqual(raw.Low.iloc[0], .999)
        raw.loc[0, "Close"] = .998
        with self.assertRaises(ValueError):
            repair_small_envelope_errors(raw)

    def bars(self):
        index = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        return pd.DataFrame({"Open": 1., "High": 1.001, "Low": .999, "Close": 1., "atr": .01, "signal": [1, 0, 0, 0, 0]}, index=index)

    def test_entry_uses_next_open_and_period_exit(self):
        bars = self.bars()
        bars.loc[bars.index[1], "Open"] = 1.0005
        trades, _, counts = simulate(bars, "2024-01-01", "2024-01-02", RULES, .0001, 1.5)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0].entry_time, str(bars.index[1]))
        self.assertEqual(trades.iloc[0].entry, 1.0005)
        self.assertEqual(trades.iloc[0].reason, "period_end")
        self.assertEqual(counts["entries"], 1)

    def test_stop_cost_is_included_in_risk(self):
        bars = self.bars()
        bars.loc[bars.index[1], "Low"] = .98
        trades, equity, _ = simulate(bars, "2024-01-01", "2024-01-02", RULES, .0001, 1.5)
        self.assertAlmostEqual(trades.iloc[0].net_r, -1.)
        self.assertAlmostEqual(equity[-1], 9950.)

    def test_same_bar_stop_and_target_is_stop_for_both_sides(self):
        for side in (1, -1):
            position = {"side": side, "stop": 1 - side * .01, "target": 1 + side * .02}
            fill, reason = exit_fill(position, 1., 1.05, .95, False)
            self.assertEqual(reason, "stop")
            self.assertEqual(fill, position["stop"])

    def test_gap_stop_uses_worse_open(self):
        for side in (1, -1):
            position = {"side": side, "stop": 1 - side * .01, "target": 1 + side * .02}
            opening = 1 - side * .03
            fill, reason = exit_fill(position, opening, 1.05, .95, False)
            self.assertEqual((fill, reason), (opening, "stop_gap"))

    def test_time_exit_precedes_later_intrabar_touch(self):
        self.assertEqual(exit_fill({"side": 1, "stop": .99, "target": 1.02}, 1., 1.05, .95, True), (1., "time"))

    def test_no_signal_from_previous_period(self):
        bars = self.bars()
        trades, _, _ = simulate(bars, "2024-01-01 01:00", "2024-01-02", RULES, .0001, 1.5)
        self.assertTrue(trades.empty)

    def test_stale_signal_skipped_after_gap(self):
        bars = self.bars().drop(self.bars().index[1:3])
        trades, _, counts = simulate(bars, "2024-01-01", "2024-01-02", RULES, .0001, 1.5)
        self.assertTrue(trades.empty)
        self.assertEqual(counts["gap_skips"], 1)

    def test_indicator_history_does_not_use_future_bars(self):
        index = pd.date_range("2020-01-01", periods=700, freq="h", tz="UTC")
        close = 1. + np.sin(np.arange(700) / 45.) * .03
        bars = pd.DataFrame({"Open": close, "High": close + .001, "Low": close - .001, "Close": close}, index=index)
        full = prepare_bars(bars, RULES)
        prefix = prepare_bars(bars.iloc[:500], RULES)
        pd.testing.assert_frame_equal(full.iloc[:500], prefix)


if __name__ == "__main__":
    unittest.main()
