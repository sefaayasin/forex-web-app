import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forex_ml import features, load_bars, split_positions, strategy_metrics, targets


class MLTests(unittest.TestCase):
    def bars(self):
        idx = pd.date_range("2020-01-01", periods=500, freq="h", tz="UTC")
        c = 1 + np.arange(500) / 10000
        return pd.DataFrame({"Open": c, "High": c + .001, "Low": c - .001, "Close": c + .0001}, index=idx)

    def test_purge_and_holdout(self):
        for _, train, test in split_positions(10000, 24):
            self.assertLess(train[-1] + 24, test[0])
            self.assertLess(test[-1] + 24, 10000)

    def test_future_changes_do_not_change_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "news.csv"
            p.write_text("date,sentiment_score\n2020-01-05,0.5\n")
            bars = self.bars()
            x, _ = features(bars, p)
            changed = bars.copy()
            changed.iloc[300:] *= 2
            z, _ = features(changed, p)
            pd.testing.assert_frame_equal(x.iloc[:300], z.iloc[:300])
            self.assertTrue(x.loc[:"2020-01-05 22:00"].fomc_sentiment.isna().all())
            self.assertEqual(x.loc["2020-01-05 23:00", "fomc_sentiment"], .5)

    def test_next_open_target_and_unknown_tail(self):
        bars = self.bars()
        y = targets(bars, 24)
        self.assertAlmostEqual(y.iloc[0], bars.Close.iloc[24] / bars.Open.iloc[1] - 1)
        self.assertTrue(y.iloc[-24:].isna().all())

    def test_cost_and_nonoverlap(self):
        r = strategy_metrics(np.ones(100), np.zeros(100), np.arange(100), 24, np.full(100, .0001))
        self.assertEqual(r["trades"], 5)
        self.assertAlmostEqual(r["total_net_bps"], -5)

    def test_duplicate_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bars.csv"
            p.write_text("timestamp,open,high,low,close\n0,1,2,1,1\n0,1,2,1,1\n")
            with self.assertRaises(ValueError):
                load_bars(p)

    def test_small_bound_repair_and_large_error_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bars.csv"
            original = "timestamp,open,high,low,close\n0,1.1,1.2,1.09,1.08999\n"
            p.write_text(original)
            bars = load_bars(p)
            self.assertEqual(bars.attrs["repaired_bounds"], 1)
            self.assertEqual(bars.Low.iloc[0], bars.Close.iloc[0])
            self.assertEqual(p.read_text(), original)
            p.write_text(original.replace("1.08999", "1.08"))
            with self.assertRaises(ValueError):
                load_bars(p)


if __name__ == "__main__":
    unittest.main()
