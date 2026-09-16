import unittest

import numpy as np
import pandas as pd

from forex_ml_live import build_research_prediction, research_signal_alignment


def synthetic_hourly_bars(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return pd.DataFrame(
        {"Open": close, "Close": close, "High": close + 0.005, "Low": close - 0.005},
        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
    )


class ForexMlLiveTests(unittest.TestCase):
    def test_unavailable_for_symbol_without_a_saved_model(self):
        result = build_research_prediction("ZZZXXX", bars=synthetic_hourly_bars())
        self.assertEqual(result["status"], "unavailable")

    def test_insufficient_data_below_minimum_bar_count(self):
        result = build_research_prediction("EURUSD", bars=synthetic_hourly_bars(n=100))
        self.assertEqual(result["status"], "insufficient_data")

    def test_no_data_when_bars_missing(self):
        result = build_research_prediction("EURUSD", bars=None)
        self.assertEqual(result["status"], "no_data")

    def test_ready_prediction_has_a_valid_probability_and_disclaimer(self):
        result = build_research_prediction("EURUSD=X", bars=synthetic_hourly_bars())
        self.assertEqual(result["status"], "ready")
        self.assertTrue(0.0 <= result["probability_up"] <= 1.0)
        self.assertIn("işlem sinyali değildir", result["note"])

    def test_alignment_reports_agreement_with_requested_side(self):
        prediction = {"status": "ready", "task": "direction", "probability_up": 0.7}
        long_view = research_signal_alignment(prediction, "LONG")
        short_view = research_signal_alignment(prediction, "SHORT")
        self.assertTrue(long_view["aligned"])
        self.assertFalse(short_view["aligned"])
        self.assertAlmostEqual(long_view["aligned_probability_pct"], 70.0)
        self.assertAlmostEqual(short_view["aligned_probability_pct"], 30.0)

    def test_alignment_is_none_when_model_not_ready(self):
        self.assertIsNone(research_signal_alignment({"status": "unavailable"}, "LONG"))


if __name__ == "__main__":
    unittest.main()
