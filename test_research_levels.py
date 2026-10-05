import unittest

import numpy as np
import pandas as pd

from research_levels import (
    BOTH, RESISTANCE, SUPPORT, SWING_BARS, WINDOW, classify, cluster_levels, epoch_day, evaluate_hypothesis,
    level_segments, pick_signals, round_rows, stationary_indices, swing_points, trade_stream, trendline_rows, trendlines,
)


def make_chart(high, low, close, buffer):
    high, low, close = (np.asarray(v, float) for v in (high, low, close))
    b = np.full(len(close), float(buffer))
    prev = np.concatenate([[np.nan], close[:-1]])
    return {"high": high, "low": low, "close": close, "prev": prev, "atr": b * 4, "b": b,
            "lo": np.minimum(low, prev) - b, "hi": np.maximum(high, prev) + b}


class SwingAndLevelTests(unittest.TestCase):
    def test_swing_needs_strict_left_and_weak_right(self):
        high = np.array([1, 2, 3, 3, 1, 2, 5, 2, 1, 1], float)
        low = high - 0.5
        swings = swing_points(high, low, k=2)
        highs = swings[swings.kind == 1]
        self.assertEqual(highs.bar.tolist(), [2, 6])
        self.assertEqual(highs.usable.tolist(), [5, 9])
        self.assertIn(4, swings[swings.kind == -1].bar.tolist())

    def test_cluster_keeps_groups_of_two_or_more(self):
        levels = cluster_levels(np.array([1.0200, 1.0100, 1.0004, 1.0, 1.0203]), 0.0005)
        np.testing.assert_allclose(levels, [1.0002, 1.02015])

    def test_level_exists_only_while_both_swings_are_usable_and_recent(self):
        swings = pd.DataFrame({"bar": [0, 20], "price": [1.1, 1.1001], "kind": [1, -1]})
        segments = level_segments(swings, np.ones(700), 700)
        starts = [s[0] for s in segments]
        self.assertEqual(starts[0], 20 + SWING_BARS + 1)
        self.assertEqual(segments[0][1], WINDOW + 1)
        np.testing.assert_allclose(segments[0][2], [1.10005])
        self.assertEqual(len(segments), 1)

    def test_round_levels_within_reach(self):
        chart = {"lo": np.array([np.nan, 1.0990, 1.0940]), "hi": np.array([np.nan, 1.1005, 1.1055])}
        t, y, _, role = round_rows(chart, 0.0001, 0.0)
        self.assertEqual(t.tolist(), [1, 2, 2, 2])
        np.testing.assert_allclose(y, [1.1000, 1.0950, 1.1000, 1.1050])
        self.assertTrue((role == BOTH).all())
        t, y, _, _ = round_rows(chart, 0.0001, 10.0)
        np.testing.assert_allclose(y, [1.0960, 1.1010])
        self.assertEqual(t.tolist(), [2, 2])


class SignalTests(unittest.TestCase):
    def test_bounce_and_break_rules_and_stops(self):
        chart = make_chart(high=[1.1012, 1.1006, 1.1012, 1.1008], low=[1.1008, 1.0999, 1.0990, 1.0994],
                           close=[1.1010, 1.1003, 1.0995, 1.1005], buffer=0.0002)
        y = np.full(3, 1.1000)
        out = classify(np.array([1, 2, 3]), y, y, np.full(3, BOTH), chart)
        rows = {(r.t, r.kind, r.side): r.stop_level for r in out.itertuples()}
        self.assertAlmostEqual(rows[(1, "bounce", 1)], 1.0997)
        self.assertAlmostEqual(rows[(2, "break", -1)], 1.1002)
        self.assertAlmostEqual(rows[(3, "break", 1)], 1.0998)
        self.assertEqual(len(rows), 3)

    def test_roles_limit_trendline_rules(self):
        chart = make_chart(high=[1.1012, 1.1006], low=[1.1008, 1.0999], close=[1.1010, 1.1003], buffer=0.0002)
        y = np.array([1.1000])
        self.assertEqual(len(classify(np.array([1]), y, y, np.array([RESISTANCE]), chart)), 0)
        self.assertEqual(len(classify(np.array([1]), y, y, np.array([SUPPORT]), chart)), 1)

    def test_pick_skips_conflicting_bars_and_uses_nearest_line(self):
        candidates = pd.DataFrame({"t": [5, 5, 6, 6], "kind": "bounce", "side": [1, -1, 1, 1],
                                   "stop_level": [1.0, 2.0, 3.0, 4.0], "dist": [0.1, 0.2, 0.5, 0.3]})
        picked = pick_signals(candidates, "bounce")
        self.assertEqual(picked.t.tolist(), [6])
        self.assertEqual(picked.stop_level.tolist(), [4.0])


class TrendlineTests(unittest.TestCase):
    def setUp(self):
        self.n = 100
        t = np.arange(self.n)
        self.line = 1.0 + 0.01 * (t - 10)
        self.swings = pd.DataFrame({"bar": [10, 30], "price": [1.0, 1.2], "kind": [-1, -1],
                                    "usable": [10 + SWING_BARS + 1, 30 + SWING_BARS + 1]})

    def chart(self, close):
        return make_chart(close + 0.001, close - 0.001, close, 0.01)

    def test_clean_rising_line_ends_at_its_break(self):
        close = self.line + 0.05
        close[60] = self.line[60] - 0.02
        chart = self.chart(close)
        lines = trendlines(self.swings, chart)
        self.assertEqual(len(lines), 1)
        start, end, _, _, slope, role, _ = lines[0]
        self.assertEqual((start, end, role), (43, self.n, SUPPORT))
        self.assertAlmostEqual(slope, 0.01)
        t, y, y_prev, _ = trendline_rows(lines, 0.0, chart)
        self.assertEqual(t.max(), 60)
        signals = classify(t, y, y_prev, np.full(len(t), SUPPORT), chart)
        self.assertEqual(signals[signals.kind == "break"].t.tolist(), [60])

    def test_line_crossed_between_anchors_is_not_drawn(self):
        close = self.line + 0.05
        close[20] = self.line[20] - 0.05
        self.assertEqual(trendlines(self.swings, self.chart(close)), [])


class TradeTests(unittest.TestCase):
    def context(self, start="2024-01-02 10:00", bars=60, target_bar=None):
        times = pd.date_range(start, periods=bars, freq="5min", tz="UTC")
        opens = np.full(bars, 1.1000)
        highs, lows = opens + 0.0002, opens - 0.0002
        if target_bar is not None:
            highs[target_bar] = 1.1020
        opens15 = pd.date_range(start, periods=bars // 3, freq="15min", tz="UTC")
        return {"open_ns": opens15.values.view("int64"), "m5_times": times.values, "m5_ns": times.values.view("int64"),
                "opens": opens, "highs": highs, "lows": lows, "closes": opens.copy(), "atr": np.full(bars // 3, 0.0010),
                "pip": 0.0001, "cost": np.full(24, 1.0)}

    def test_entry_at_next_open_target_and_position_lock(self):
        context = self.context(target_bar=6)
        signals = pd.DataFrame({"t": [0, 1], "side": [1, 1], "stop_level": [1.0990, 1.0990]})
        trades, skipped = trade_stream(signals, context)
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(pd.Timestamp(trade["entry_ns"], tz="UTC"), pd.Timestamp("2024-01-02 10:15", tz="UTC"))
        self.assertEqual(trade["reason"], "target")
        self.assertAlmostEqual(trade["stop_pips"], 10.0)
        self.assertAlmostEqual(trade["net_r"], (15.0 - 1.0) / 10.0)
        self.assertAlmostEqual(trade["stress_r"], (15.0 - 2.0) / 10.0)
        self.assertEqual(skipped["position_open"], 1)

    def test_minimum_stop_and_rollover_skip(self):
        context = self.context()
        trades, _ = trade_stream(pd.DataFrame({"t": [0], "side": [1], "stop_level": [1.09999]}), context)
        self.assertAlmostEqual(trades[0]["stop_pips"], 5.0)
        rollover = self.context(start="2024-01-02 21:45")  # entry 22:00 UTC = 17:00 New York
        trades, skipped = trade_stream(pd.DataFrame({"t": [0], "side": [1], "stop_level": [1.0990]}), rollover)
        self.assertEqual((len(trades), skipped["rollover_hour"]), (0, 1))


class BootstrapTests(unittest.TestCase):
    def test_indices_are_wrapping_blocks(self):
        indices = stationary_indices(10, simulations=50, block=1e9, seed=1)
        self.assertEqual(indices.shape, (50, 10))
        steps = np.diff(indices, axis=1) % 10
        self.assertTrue((steps == 1).all())
        mixed = stationary_indices(10, simulations=50, block=2, seed=1)
        self.assertTrue(((mixed >= 0) & (mixed < 10)).all())

    def test_evaluate_reports_means_differences_and_checks(self):
        rng = np.random.default_rng(0)
        days = np.arange(epoch_day("2010-01-04"), epoch_day("2024-01-01"), 3)

        def frame(mean, pairs):
            n = len(days) * 2
            net = rng.normal(mean, 0.5, n)
            return pd.DataFrame({
                "day": np.repeat(days, 2).astype(np.int32), "net_r": net, "gross_r": net + 0.1,
                "stress_r": net - 0.1, "stop_pips": 10.0, "cost_pips": 1.0,
                "reason": pd.Categorical(np.where(net > 0, "target", "stop")),
                "pair": pd.Categorical(np.resize(pairs, n)),
            })

        real, placebo = frame(0.3, [f"P{i}" for i in range(20)]), frame(0.0, ["P0"])
        day_index = np.unique(np.concatenate([real.day, placebo.day]))
        out = evaluate_hypothesis(real, placebo, day_index, stationary_indices(len(day_index), simulations=200))
        self.assertAlmostEqual(out["net_r_test"]["mean"], real.net_r.mean())
        self.assertAlmostEqual(out["net_r_minus_placebo"]["difference"], real.net_r.mean() - placebo.net_r.mean())
        self.assertLess(out["net_r_test"]["ci_low"], out["net_r_test"]["mean"])
        self.assertEqual(out["pairs"], 20)
        self.assertTrue(out["passes"])
        self.assertEqual(set(out["net_r_by_period"]), {"2008-2013", "2014-2019", "2020-2026"})
        out = evaluate_hypothesis(placebo, real, day_index, stationary_indices(len(day_index), simulations=200))
        self.assertFalse(out["checks"]["beats_placebo_significant"])


if __name__ == "__main__":
    unittest.main()
