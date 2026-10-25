import unittest

import numpy as np
import pandas as pd

from research_cash_target import describe, rule_states, simulate, simulate_path


class CashTargetTests(unittest.TestCase):
    def frame(self, n=600):
        rng = np.random.default_rng(83)
        c = 1.1 + rng.normal(0, .0001, n).cumsum()
        d = pd.DataFrame({"Open": c, "High": c+.0001, "Low": c-.0001, "Close": c},
            index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"))
        d.attrs["symbol"] = "EURUSD"
        return d

    def test_all_entry_rules_are_causal(self):
        original = self.frame(1100)
        modified = original.copy()
        modified.iloc[777:] *= 2
        first, _ = rule_states(original)
        changed, _ = rule_states(modified)
        for name in first:
            np.testing.assert_array_equal(first[name][:777], changed[name][:777], err_msg=name)

    def test_net_cash_target_includes_the_cost_once(self):
        frame = self.frame()
        fills = frame.resample("5min").ffill()
        state = np.zeros(len(frame), dtype=int)
        state[511] = 1  # 08:00 UTC, in the declared liquid session
        when = frame.index[511] + pd.Timedelta(minutes=15)
        # Stop 10 pips, cost 2; cash target -> gross target 5.6 pips.
        j = fills.index.get_loc(when)
        fills.iloc[j, fills.columns.get_loc("High")] = fills.Open.iloc[j] + .001
        data = simulate(frame, fills, state, pd.Series(.001, index=frame.index),
            np.full(len(frame), 2.), "2024-01-01", "2024-02-01", 1., 1)
        self.assertEqual(len(data), 1)
        self.assertEqual(data.reason.iloc[0], "target")
        self.assertAlmostEqual(data.pnl_usd.iloc[0], 30)
        self.assertAlmostEqual(data.target_pips.iloc[0], 5.6)
        self.assertLess(data.pnl_stress_usd.iloc[0], 30)

    def test_stop_first_and_adverse_gap_can_exceed_cash_budget(self):
        values = np.tile([1., 1.001, .999, 1.], (12, 1))
        self.assertEqual(simulate_path(values, 0, 1, .001, .0005, 12)[2], "stop")
        values[:] = [1., 1.0001, .9999, 1.]
        values[1] = [.998] * 4
        price, _, reason = simulate_path(values, 0, 1, .001, .0005, 12)
        self.assertEqual(reason, "stop_gap")
        self.assertLess(price, .999)

    def test_missing_or_unfinished_path_is_not_labeled(self):
        frame = self.frame()
        fills = frame.resample("5min").ffill()
        state = np.zeros(len(frame), dtype=int)
        state[511] = 1
        fills = fills.drop(fills.index[511*3+6])
        data = simulate(frame, fills, state, pd.Series(.001, index=frame.index),
            np.ones(len(frame)), "2024-01-01", "2024-02-01", 1., 1)
        self.assertTrue(data.empty)

    def test_account_does_not_keep_trading_after_risk_becomes_unaffordable(self):
        trades = pd.DataFrame({"pnl_usd": [-100.] * 12, "pnl_stress_usd": [-110.] * 12,
            "reason": ["stop"] * 12})
        result = describe(trades)
        self.assertEqual(result["affordable_trades"], 10)
        self.assertEqual(result["fixed_budget_account_balance"], 0)
        self.assertTrue(result["account_exhausted"])


if __name__ == "__main__":
    unittest.main()
