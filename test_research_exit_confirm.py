import unittest

import pandas as pd

import research_exit_confirm as exit_confirm

HOUR = 3600 * 10**9


def frame(rows):
    return exit_confirm.trade_frame([
        {"pair": pair, "side": 1, "opened": opened * HOUR, "closed": closed * HOUR, "lot": .01, "gross_pips": 0.,
         "net_usd": r, "reason": "HEDEF", "trailing": False, "risk_usd": 1.} for pair, opened, closed, r in rows])


class ExitConfirmTests(unittest.TestCase):
    def test_reentries_count_per_pair_inside_the_window(self):
        rows = frame([("EURUSD", 0, 1, .3), ("EURUSD", 2, 3, .3), ("EURUSD", 5, 6, .3), ("GBPUSD", 3, 4, .3)])
        self.assertEqual(exit_confirm.reentries(rows, 60), 1)

    def test_paired_compares_only_common_entries_by_period(self):
        late = pd.Timestamp("2025-06-02", tz="UTC").value // HOUR
        base = frame([("EURUSD", 0, 1, -.5), ("EURUSD", late, late + 1, .3), ("GBPUSD", 3, 4, -1.)])
        other = frame([("EURUSD", 0, 1, -.2), ("EURUSD", late, late + 1, .3), ("USDJPY", 3, 4, 9.)])
        result = exit_confirm.paired(base, other)
        self.assertEqual((result["common"], result["changed"]), (2, 1))
        self.assertAlmostEqual(result["diff_r"], .15)
        self.assertAlmostEqual(result["diff_r_first"], .3)
        self.assertAlmostEqual(result["diff_r_second"], 0.)

    def test_big_loss_and_win_rate_use_r_not_dollars(self):
        rows = frame([("EURUSD", 0, 1, -.8), ("EURUSD", 2, 3, -.5), ("EURUSD", 4, 5, .3)])
        result = exit_confirm.describe(rows)
        self.assertEqual(result["big_losses"], 1)
        self.assertAlmostEqual(result["mean_loss_r"], -.65)
        self.assertAlmostEqual(result["win_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
