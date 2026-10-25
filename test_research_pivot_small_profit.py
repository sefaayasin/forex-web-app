import unittest
import pandas as pd

from research_pivot_small_profit import evaluate_scenario, target_distance


class SmallProfitTests(unittest.TestCase):
    def test_net_target_adds_cost_and_scales_by_cash_per_pip(self):
        self.assertAlmostEqual(target_distance(10, 1.5, 5), 3.5)
        self.assertAlmostEqual(target_distance(3, 1, 1), 4)

    def test_early_small_target_allows_later_original_signal(self):
        now = pd.Timestamp("2025-01-07 12:00Z")
        index = pd.date_range(now, periods=52, freq="5min")
        bars = pd.DataFrame({"Open": 1.1, "High": 1.1005, "Low": 1.0999, "Close": 1.1}, index=index)
        decisions = pd.DataFrame([dict(pair="EURUSD", time=t, side="LONG", signal_price=1.1,
                                        stop=1.099, cost_pips=1.) for t in (now, now+pd.Timedelta(minutes=10))])
        trades, counts = evaluate_scenario(decisions, {"EURUSD": bars}, "pip", 3)
        self.assertEqual(len(trades), 2)
        self.assertEqual(counts["overlaps"], 0)
        self.assertTrue(trades.kind.eq("target").all())
        for value in trades.net_result:
            self.assertAlmostEqual(value, 3)

    def test_cash_target_does_not_hide_stop_budget(self):
        now = pd.Timestamp("2025-01-07 12:00Z")
        index = pd.date_range(now, periods=50, freq="5min")
        bars = pd.DataFrame({"Open": 1.1, "High": 1.1, "Low": 1.098, "Close": 1.1}, index=index)
        decisions = pd.DataFrame([dict(pair="EURUSD", time=now, side="LONG", signal_price=1.1,
                                       stop=1.099, cost_pips=1.)])
        trades, _ = evaluate_scenario(decisions, {"EURUSD": bars}, "cash", 10)
        self.assertEqual(trades.kind.iloc[0], "stop")
        self.assertAlmostEqual(trades.net_result.iloc[0], -100)


if __name__ == "__main__":
    unittest.main()
