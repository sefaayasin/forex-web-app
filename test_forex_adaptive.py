from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from forex_adaptive import (FIVE, FIFTEEN, before, build_dataset, entry_features,
    metrics, probabilities, resample_complete, select_model, trade_outcome, update_symbol)
from forex_indicators import compute_rsi


def bars(n=1600):
    rng = np.random.default_rng(10)
    close = 1 + rng.normal(0, .0002, n).cumsum()
    return pd.DataFrame({"Open": close, "High": close + .0003,
        "Low": close - .0003, "Close": close},
        index=pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"))


class AdaptiveTests(unittest.TestCase):
    def test_rsi_no_losses_and_flat_prices(self):
        for values, expected in ((np.arange(50.), 100.), (-np.arange(50.), 0.), (np.ones(50), 50.)):
            result = compute_rsi(pd.Series(values))
            self.assertTrue(result.iloc[:14].isna().all())
            self.assertAlmostEqual(result.iloc[-1], expected)

    def test_features_are_invariant_to_future_and_incomplete_higher_bars(self):
        original = bars()
        full, groups, _ = entry_features(original)
        changed = original.copy()
        changed.iloc[1111:] *= 2
        modified, _, _ = entry_features(changed)
        prefix, _, _ = entry_features(original.iloc[:1111])
        pd.testing.assert_frame_equal(full.iloc[:1111], modified.iloc[:1111])
        pd.testing.assert_frame_equal(full.iloc[:1111], prefix)
        self.assertTrue(set(groups["price"]) < set(groups["technical"]) < set(groups["extended"]))

    def test_resampling_rejects_partial_and_missing_bars(self):
        frame = bars(32).drop(bars(32).index[9])
        out = resample_complete(frame, 4)
        self.assertEqual(list(out.index), [frame.index[16-1]])

    def test_purge_uses_entire_outcome_window_even_early_exit(self):
        t = pd.Timestamp("2025-01-01", tz="UTC")
        data = pd.DataFrame({"entry_time": [t - pd.Timedelta(hours=2), t - pd.Timedelta(hours=6), t],
            "exit_time": [t - pd.Timedelta(hours=1), t - pd.Timedelta(hours=5), t + FIVE],
            "label_end": [t + pd.Timedelta(hours=2), t - pd.Timedelta(hours=2), t + pd.Timedelta(hours=4)]})
        selected = before(data, t - pd.Timedelta(days=1), t)
        self.assertEqual(selected.index.tolist(), [1])

    def test_stop_first_same_bar_and_gap_fill_both_sides(self):
        for side in (1, -1):
            values = np.tile([1., 1.2, .8, 1.], (48, 1))
            price, index, reason = trade_outcome(values, 0, side, .1)
            self.assertEqual(reason, "stop")
            self.assertEqual(index, 0)
            self.assertAlmostEqual(price, 1 - side * .1)
            values[:] = [1., 1.01, .99, 1.]
            values[1] = [1 - side * .2] * 4
            price, index, reason = trade_outcome(values, 0, side, .1)
            self.assertEqual(reason, "stop_gap")
            self.assertAlmostEqual(price, 1 - side * .2)

    def test_timeout_at_last_close_not_future_open(self):
        values = np.tile([1., 1.01, .99, 1.005], (49, 1))
        values[48] *= 2
        price, index, reason = trade_outcome(values, 0, 1, .1)
        self.assertEqual((price, index, reason), (1.005, 47, "time"))

    def test_cash_reward_matches_net_thirty_over_hundred_budget(self):
        values = np.tile([1., 1.0006, .9999, 1.], (48, 1))
        pip, stop_pips, cost = .0001, 10., 1.5
        target_distance = pip * (cost + .3 * (stop_pips + cost))
        price, _, reason = trade_outcome(values, 0, 1, stop_pips * pip, target_distance)
        self.assertEqual(reason, "target")
        net_r = ((price - 1) / pip - cost) / (stop_pips + cost)
        self.assertAlmostEqual(net_r * 100, 30)

    def training_data(self):
        rng = np.random.default_rng(17)
        times = pd.date_range("2010-01-01", "2025-01-01", freq="D", tz="UTC")
        feature = rng.normal(size=len(times))
        label = (feature + rng.normal(size=len(times)) > 0).astype(int)
        return pd.DataFrame({"entry_time": times, "label_end": times + pd.Timedelta(hours=4),
            "feature": feature, "label": label, "net_r": np.where(label, 1.5, -1.),
            "net_r_stress": np.where(label, 1.4, -1.1)})

    def test_future_outcomes_cannot_change_learned_weights_or_threshold(self):
        cutoff = pd.Timestamp("2023-01-01", tz="UTC")
        data = self.training_data()
        groups = {"price": ["feature"]}
        original = select_model(data, groups, cutoff)
        modified = data.copy()
        future = modified.entry_time >= cutoff
        modified.loc[future, "feature"] *= -100
        modified.loc[future, "label"] = 1 - modified.loc[future, "label"]
        modified.loc[future, ["net_r", "net_r_stress"]] = 999
        changed = select_model(modified, groups, cutoff)
        self.assertEqual(original["weights"], changed["weights"])
        self.assertEqual(original["threshold"], changed["threshold"])
        self.assertEqual(original["selection_trials"], changed["selection_trials"])
        p = probabilities(original, data.tail(10))
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertLess(pd.Timestamp(original["periods"]["train_last_label"]), cutoff - pd.DateOffset(months=12))
        self.assertLess(pd.Timestamp(original["periods"]["calibration_last_label"]), cutoff)

    def test_losing_validation_abstains(self):
        data = self.training_data()
        data["net_r"], data["net_r_stress"] = -1., -1.1
        bundle = select_model(data, {"price": ["feature"]}, pd.Timestamp("2023-01-01", tz="UTC"))
        self.assertIsNone(bundle["threshold"])
        self.assertIsNotNone(bundle["diagnostic_threshold"])

    def test_no_trades_and_first_loss_drawdown(self):
        self.assertIsNone(metrics(pd.DataFrame())["mean_r"])
        frame = pd.DataFrame({"label": [0, 1], "net_r": [-1., .5], "net_r_stress": [-1.1, .4]})
        self.assertEqual(metrics(frame)["max_drawdown_r"], 1.)

    def test_dataset_next_open_cost_and_unknown_tail(self):
        frame = bars(650)
        fills = frame.resample("5min").ffill()
        frame.attrs.update(bad_positions=[], repaired_bounds=0, quarantined_rows=0)
        fills.attrs.update(bad_positions=[], repaired_bounds=0, quarantined_rows=0)
        x, groups, atr = entry_features(frame)
        x["score15"] = 0.
        x.iloc[500, x.columns.get_loc("score15")] = .6
        x.iloc[-1, x.columns.get_loc("score15")] = .6
        with patch("forex_adaptive.load_hourly", side_effect=[frame, fills]), \
                patch("forex_adaptive.entry_features", return_value=(x, groups, atr)), \
                patch("forex_adaptive.digest", return_value="synthetic"):
            data, _, audit = build_dataset("EURUSD")
        self.assertEqual(len(data), 1)
        self.assertEqual(data.entry_time.iloc[0], frame.index[500] + FIFTEEN)
        self.assertAlmostEqual(data.net_r.iloc[0], (data.gross_pips.iloc[0] - 1.5) / data.stop_pips.iloc[0])
        self.assertEqual(audit["counters"]["missing_path"], 1)

    def test_cash_learner_does_not_count_small_timeout_profit_as_target_success(self):
        frame = bars(650)
        fills = frame.resample("5min").ffill()
        frame.attrs.update(bad_positions=[], repaired_bounds=0, quarantined_rows=0)
        fills.attrs.update(bad_positions=[], repaired_bounds=0, quarantined_rows=0)
        x, groups, atr = entry_features(frame)
        x["score15"] = 0.
        x.iloc[500, x.columns.get_loc("score15")] = .6
        def timeout(values, entry, side, distance, target_distance):
            return values[entry, 0] + .0003, entry+11, "time"
        with patch("forex_adaptive.load_hourly", side_effect=[frame, fills]), \
                patch("forex_adaptive.entry_features", return_value=(x, groups, atr)), \
                patch("forex_adaptive.trade_outcome", side_effect=timeout), \
                patch("forex_adaptive.digest", return_value="synthetic"):
            data, _, _ = build_dataset("EURUSD", net_target_ratio=.3)
        self.assertGreater(data.net_r.iloc[0], 0)
        self.assertEqual(data.label.iloc[0], 0)

    def test_symbol_and_interval_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            for symbol, interval in (("../EURUSD", 3), ("EURUSD", 2)):
                with self.assertRaises(ValueError):
                    update_symbol(symbol, Path(tmp), months=interval)


if __name__ == "__main__":
    unittest.main()
