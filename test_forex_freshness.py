import unittest

import numpy as np
import pandas as pd

from forex_freshness import reversal_level, reversal_warnings, signal_freshness


def bars_and_score(closes, scores, ema20=None, atr=0.0010):
    index = pd.date_range("2026-09-01", periods=len(closes), freq="15min", tz="UTC")
    closes = np.asarray(closes, dtype=float)
    bars = pd.DataFrame(
        {"Close": closes, "EMA20": closes if ema20 is None else ema20, "ATR14": atr}, index=index
    )
    return bars, pd.Series(scores, index=index, dtype=float)


class SignalFreshnessTests(unittest.TestCase):
    def test_age_and_move_count_from_last_threshold_cross(self):
        bars, score = bars_and_score([1.1000, 1.1000, 1.1010, 1.1020, 1.1030], [10, 30, 10, 40, 60])
        result = signal_freshness(bars, score, "LONG", pip=0.0001, typical_move_pips=100)
        self.assertEqual(result["state"], "FRESH")
        self.assertEqual(result["bars"], 2)
        self.assertEqual(result["start_time"], bars.index[3])
        self.assertAlmostEqual(result["moved_pips"], 10.0)
        self.assertAlmostEqual(result["moved_ratio"], 0.10)

    def test_move_beyond_typical_range_is_late(self):
        bars, score = bars_and_score([1.1000, 1.1050, 1.1120], [0, 50, 100])
        result = signal_freshness(bars, score, "LONG", pip=0.0001, typical_move_pips=60)
        self.assertEqual(result["state"], "LATE")

    def test_half_of_typical_range_is_mature(self):
        bars, score = bars_and_score([1.1000, 1.1010, 1.1045], [0, 50, 90])
        result = signal_freshness(bars, score, "LONG", pip=0.0001, typical_move_pips=60)
        self.assertEqual(result["state"], "MATURE")

    def test_far_from_ema20_is_late_even_with_small_move(self):
        closes = [1.1000, 1.1000, 1.1005]
        bars, score = bars_and_score(closes, [0, 50, 90], ema20=[1.1000, 1.1000, 1.0980], atr=0.0010)
        result = signal_freshness(bars, score, "LONG", pip=0.0001, typical_move_pips=100)
        self.assertAlmostEqual(result["stretch_atr"], 2.5)
        self.assertEqual(result["state"], "LATE")

    def test_short_side_measures_downward_move(self):
        bars, score = bars_and_score([1.1000, 1.0990, 1.0960], [0, -40, -80])
        result = signal_freshness(bars, score, "SHORT", pip=0.0001, typical_move_pips=100)
        self.assertAlmostEqual(result["moved_pips"], 30.0)
        self.assertEqual(result["state"], "FRESH")

    def test_score_not_on_side_now(self):
        bars, score = bars_and_score([1.1, 1.1, 1.1], [50, 50, 10])
        self.assertEqual(signal_freshness(bars, score, "LONG", 0.0001, 50)["state"], "NOT_ON_SIDE")

    def test_streak_older_than_data_is_flagged(self):
        bars, score = bars_and_score([1.1, 1.1, 1.1], [50, 60, 70])
        result = signal_freshness(bars, score, "LONG", 0.0001, 50)
        self.assertTrue(result["started_before_data"])
        self.assertEqual(result["bars"], 3)

    def test_no_side_returns_unknown(self):
        bars, score = bars_and_score([1.1, 1.1], [50, 50])
        self.assertEqual(signal_freshness(bars, score, "NONE", 0.0001, 50)["state"], "UNKNOWN")


def structure_frame(rows):
    defaults = {"RSIDivergence": "NONE", "MACDDivergence": "NONE", "BBPattern": "NONE",
                "MACDMomentumState": "MIXED", "RSI14": 55.0, "Score": 60.0}
    return pd.DataFrame([{**defaults, **row} for row in rows])


class ReversalWarningTests(unittest.TestCase):
    def test_quiet_frame_has_no_warning(self):
        self.assertEqual(reversal_warnings(structure_frame([{}, {}, {}]), "LONG"), [])

    def test_opposite_divergence_within_lookback_is_reported(self):
        frame = structure_frame([{}, {"RSIDivergence": "BEARISH"}, {}, {}])
        self.assertEqual(reversal_warnings(frame, "LONG", lookback=3), ["RSI uyumsuzluğu"])
        self.assertEqual(reversal_warnings(frame, "LONG", lookback=2), [])

    def test_same_side_divergence_is_not_a_warning(self):
        frame = structure_frame([{"RSIDivergence": "BULLISH", "MACDDivergence": "BULLISH"}])
        self.assertEqual(reversal_warnings(frame, "LONG"), [])

    def test_short_side_warnings(self):
        frame = structure_frame([{"MACDDivergence": "BULLISH", "BBPattern": "W_CONFIRMED",
                                  "MACDMomentumState": "BEARISH_WEAKENING", "RSI14": 25.0, "Score": 40.0}])
        self.assertEqual(
            reversal_warnings(frame, "SHORT"),
            ["MACD uyumsuzluğu", "W dip formasyonu", "MACD momentumu zayıflıyor", "RSI aşırı satım (25)",
             "skor ters yöne döndü (+40)"],
        )

    def test_long_rsi_extreme_and_score_flip(self):
        frame = structure_frame([{"RSI14": 74.0, "Score": -30.0}])
        self.assertEqual(reversal_warnings(frame, "LONG"), ["RSI aşırı alım (74)", "skor ters yöne döndü (-30)"])


class ReversalLevelTests(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(reversal_level(["a", "b"], ["c"]), "STRONG")
        self.assertEqual(reversal_level(["a", "b"], []), "WEAK")
        self.assertEqual(reversal_level(["a"], ["c", "d"]), "WEAK")
        self.assertEqual(reversal_level([], []), "NONE")


if __name__ == "__main__":
    unittest.main()
