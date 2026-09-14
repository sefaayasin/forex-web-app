import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forex_ml_tournament import (asof_values, build_features, load_hourly, make_targets,
    nonoverlap_positions, positions, score, selection_score, bootstrap_difference)


class TournamentTests(unittest.TestCase):
    def bars(self, n=1200):
        rng = np.random.default_rng(7)
        close = np.exp(np.cumsum(rng.normal(0, .001, n)))
        return pd.DataFrame({"Open": close, "Close": close, "High": close + .005,
            "Low": close - .005, "Volume": 100.},
            index=pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"))

    def test_features_cannot_see_future_price_or_news(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "news/cot").mkdir(parents=True)
            fomc = root / "news/fomc_statements.csv"
            cot = root / "news/cot/cot_currency_positioning.csv"
            fomc.write_text("date,sentiment_score\n2020-01-01,0.2\n2020-02-10,0.9\n")
            cot.write_text("date,currency,noncommercial_net,open_interest\n2020-01-01,EUR,10,100\n2020-02-10,EUR,90,100\n")
            b = self.bars()
            original, _ = build_features(b, "EURUSD", root)
            changed = b.copy()
            changed.iloc[900:] *= 1.1
            fomc.write_text(fomc.read_text().replace("0.9", "-100"))
            cot.write_text(cot.read_text().replace("90,100", "-900,100"))
            updated, _ = build_features(changed, "EURUSD", root)
            pd.testing.assert_frame_equal(original.iloc[:900], updated.iloc[:900])

    def test_asof_retains_weekend_release_and_never_backfills(self):
        index = pd.to_datetime(["2020-01-03 10:00Z", "2020-01-06 00:00Z"])
        available = pd.to_datetime(["2020-01-04 00:00Z"])
        result = asof_values(index, available, pd.DataFrame({"score": [.7]}), 10)
        self.assertTrue(np.isnan(result.score.iloc[0]))
        self.assertEqual(result.score.iloc[1], .7)

    def test_labels_end_before_split_boundary(self):
        b = self.bars()
        t = make_targets(b, 24)
        train = positions(b, t, "2020-01-01", "2020-02-01")
        test = positions(b, t, "2020-02-01", "2020-03-01")
        self.assertLess(t.label_end.iloc[train].max(), b.index[test[0]])
        self.assertFalse(t.valid.iloc[-24:].any())

    def test_missing_year_not_treated_as_next_bar(self):
        b = self.bars()
        idx = list(b.index)
        idx[800:] = [t + pd.Timedelta(days=365) for t in idx[800:]]
        b.index = pd.DatetimeIndex(idx)
        t = make_targets(b, 24)
        self.assertFalse(t.valid.iloc[776:800].any())

    def test_volatility_target_uses_future_window(self):
        b = self.bars()
        t = make_targets(b, 4)
        k = 600
        squared = b.Close.pct_change().pow(2)
        expected = squared.iloc[k+1:k+5].mean() > squared.iloc[k-479:k+1].mean()
        self.assertEqual(t.high_volatility.iloc[k], int(expected))

    def test_nonoverlap_and_baseline_imbalance(self):
        b = self.bars()
        t = make_targets(b, 24)
        ix = np.arange(500, 800, 4)
        take = nonoverlap_positions(b.index[ix], t.label_end.iloc[ix])
        for left, right in zip(take[:-1], take[1:]):
            self.assertGreaterEqual(b.index[ix[right]], t.label_end.iloc[ix[left]])
        y = np.r_[np.zeros(90), np.ones(10)]
        m = score(y, np.zeros(100), .1, np.zeros(100))
        self.assertEqual(m["accuracy"], .9)
        self.assertEqual(m["balanced_accuracy"], .5)

    def test_sort_in_memory_and_reject_corrupt_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "EURUSD.csv"
            content = "timestamp,open,high,low,close\n3600000,1,1.1,.9,1\n0,1,1.1,.9,1\n"
            p.write_text(content)
            d = load_hourly(p)
            self.assertTrue(d.attrs["reordered"])
            self.assertTrue(d.index.is_monotonic_increasing)
            self.assertEqual(p.read_text(), content)
            p.write_text(content.replace("1.1", "0.5"))
            with self.assertRaises(ValueError):
                load_hourly(p)

    def test_block_bootstrap_identical_predictions_has_zero_lift(self):
        y = np.tile([0, 1], 100)
        baseline = np.tile([0, 0], 100)
        lo, hi = bootstrap_difference(y, baseline.astype(float), baseline, repetitions=50)
        self.assertEqual((lo, hi), (0., 0.))

    def test_quarantine_excludes_affected_targets_and_feature_warmup(self):
        b = self.bars(n=2500)
        b.attrs["bad_positions"] = [700]
        t = make_targets(b, 24)
        self.assertFalse(t.valid.iloc[676:1701].any())
        self.assertTrue(t.valid.iloc[1701])

    def test_selection_penalizes_instability(self):
        rows = []
        for model, scores in [("stable", [.6, .6]), ("unstable", [.5, .72])]:
            for ba in scores:
                rows.append({"task": "direction", "horizon": 4, "bundle": "price", "model": model,
                    "accuracy": ba, "balanced_accuracy": ba, "auc": ba, "brier": .25,
                    "baseline_brier": .25, "majority_accuracy": .5,
                    "persistence_balanced_accuracy": .5, "n": 100})
        ranked = selection_score(pd.DataFrame(rows))
        self.assertEqual(ranked.iloc[0].model, "stable")


if __name__ == "__main__":
    unittest.main()
