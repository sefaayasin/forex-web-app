import unittest

import numpy as np
import pandas as pd

from forex_ml_live import (
    build_research_prediction,
    direction_cost_verdict,
    has_research_model,
    research_signal_alignment,
    volatility_risk_view,
)


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

    def test_has_research_model_accepts_yahoo_suffix(self):
        self.assertTrue(has_research_model("EURUSD=X", "high_volatility"))
        self.assertFalse(has_research_model("ZZZXXX", "high_volatility"))

    def test_ready_volatility_prediction_carries_frozen_threshold(self):
        result = build_research_prediction("EURUSD", "high_volatility", bars=synthetic_hourly_bars())
        self.assertEqual(result["status"], "ready")
        self.assertAlmostEqual(result["confidence_threshold"], 0.65)
        view = volatility_risk_view(result)
        self.assertIn(view["level"], {"high", "normal", "uncertain"})
        self.assertEqual(view["horizon_bars"], 72)

    def test_volatility_view_is_uncertain_between_thresholds(self):
        def view(p):
            return volatility_risk_view(
                {"status": "ready", "task": "high_volatility", "probability_up": p, "confidence_threshold": 0.65}
            )

        self.assertEqual(view(0.65)["level"], "high")
        self.assertEqual(view(0.35)["level"], "normal")
        self.assertEqual(view(0.5)["level"], "uncertain")
        self.assertFalse(view(0.5)["confident"])
        self.assertAlmostEqual(view(0.8)["probability_high_pct"], 80.0)

    def test_volatility_view_ignores_direction_and_unready_predictions(self):
        self.assertIsNone(volatility_risk_view({"status": "ready", "task": "direction", "probability_up": 0.9}))
        self.assertIsNone(volatility_risk_view({"status": "no_data"}))

    def test_direction_cost_verdict(self):
        self.assertEqual(direction_cost_verdict(-0.4, -1.2), "Maliyet sonrası zararda")
        self.assertEqual(direction_cost_verdict(0.0, 0.5), "Maliyet sonrası zararda")
        self.assertEqual(direction_cost_verdict(0.15, -0.75), "Sadece düşük maliyette artı")
        self.assertEqual(direction_cost_verdict(0.3, 0.1), "Maliyet sonrası artı")
        self.assertEqual(direction_cost_verdict(None, None), "-")
        self.assertEqual(direction_cost_verdict(float("nan"), 1.0), "-")


if __name__ == "__main__":
    unittest.main()
