from types import SimpleNamespace
import unittest

import pandas as pd

from compare_entry_models import summarize


class ComparisonMetricsTests(unittest.TestCase):
    def test_no_trades_does_not_claim_positive_performance(self):
        result = summarize(SimpleNamespace(trades=pd.DataFrame()), 10000.)
        self.assertEqual(result["trades"], 0)
        self.assertIsNone(result["avg_r"])
        self.assertIsNone(result["profit_factor"])

    def test_drawdown_includes_loss_from_initial_balance(self):
        result = summarize(SimpleNamespace(
            trades=pd.DataFrame({"PnL": [-50.], "Risk Amount": [50.]}),
            equity=pd.DataFrame({"Balance": [9950.]}),
        ), 10000.)
        self.assertAlmostEqual(result["max_drawdown_pct"], .5)
        self.assertEqual(result["avg_r"], -1.)
        self.assertIsNone(result["last_30pct_avg_r"])

    def test_profit_factor_and_drawdown_after_peak(self):
        result = summarize(SimpleNamespace(
            trades=pd.DataFrame({"PnL": [100., -200., 50.], "Risk Amount": [50., 50., 50.]}),
            equity=pd.DataFrame({"Balance": [10100., 9900., 9950.]}),
        ), 10000.)
        self.assertEqual(result["net_pnl_usd"], -50.)
        self.assertEqual(result["profit_factor"], .75)
        self.assertAlmostEqual(result["max_drawdown_pct"], 200 / 10100 * 100)
        self.assertEqual(result["last_30pct_avg_r"], 1.)


if __name__ == "__main__":
    unittest.main()
