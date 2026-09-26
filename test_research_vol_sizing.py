import unittest

import numpy as np
import pandas as pd

from research_vol_sizing import attach_probability, bootstrap_difference, max_drawdown, regime, sizing_comparison


def toy_trades():
    day1, day2 = pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")
    return pd.DataFrame({
        "regime": ["high", "high", "normal", "normal", "high", "normal"],
        "net_r": [-1.0, -1.0, 1.5, -1.0, -1.0, 1.5],
        "day": [day1, day1, day1, day1, day2, day2],
    })


class VolSizingTests(unittest.TestCase):
    def test_regime_uses_app_thresholds(self):
        self.assertEqual(regime(pd.Series([0.65, 0.5, 0.35, 0.9])).tolist(), ["high", "uncertain", "normal", "high"])

    def test_probability_comes_from_the_last_closed_hour(self):
        probabilities = pd.Series([0.1, 0.9], index=pd.to_datetime(["2024-01-02 10:00", "2024-01-02 11:00"], utc=True))
        trades = pd.DataFrame({"entry_time": pd.to_datetime(["2024-01-02 11:55", "2024-01-02 12:00", "2024-01-02 16:30"], utc=True)})
        attached = attach_probability(trades, probabilities)
        self.assertEqual(attached.iloc[0], 0.1)
        self.assertEqual(attached.iloc[1], 0.9)
        self.assertTrue(np.isnan(attached.iloc[2]))

    def test_bootstrap_difference_reports_observed_high_minus_normal(self):
        trades = toy_trades()
        result = bootstrap_difference(trades, trades.net_r, "mean")
        self.assertAlmostEqual(result["difference"], -1.0 - (1.5 - 1.0 + 1.5) / 3)
        self.assertLessEqual(result["ci_low"], result["ci_high"])

    def test_sizing_keeps_average_risk_and_halves_high_regime(self):
        out = sizing_comparison(toy_trades())
        self.assertAlmostEqual(out["multiplier_high"] * 3 + out["multiplier_other"] * 3, 6.0)
        self.assertAlmostEqual(out["multiplier_high"] / out["multiplier_other"], 0.5)

    def test_max_drawdown(self):
        self.assertAlmostEqual(max_drawdown(pd.Series([1.0, -2.0, 0.5, -1.0, 3.0])), -2.5)


if __name__ == "__main__":
    unittest.main()
