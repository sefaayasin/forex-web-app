import unittest
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from forex_analysis import bias_score_frame, evaluate_bias
from forex_indicators import compute_atr, compute_bbands, compute_macd, compute_rsi, wilder_average
from research_news_scan import load_app_score


def reference_wilder(values, period):
    result = np.full(len(values), np.nan)
    first = next((i for i in range(period-1, len(values)) if np.isfinite(values[i-period+1:i+1]).all()), None)
    if first is None:
        return result
    result[first] = sum(values[first-period+1:first+1]) / period
    for i in range(first+1, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result


class IndicatorAccuracyTests(unittest.TestCase):
    def frame(self):
        rng = np.random.default_rng(43)
        close = 1 + rng.normal(0, .0003, 1300).cumsum()
        return pd.DataFrame({"Open": close, "High": close + .0002,
            "Low": close - .0002, "Close": close},
            index=pd.date_range("2024-01-01", periods=len(close), freq="15min", tz="UTC"))

    def test_rsi_against_independent_wilder_recurrence(self):
        close = self.frame().Close
        delta = np.r_[np.nan, np.diff(close)]
        gain, loss = reference_wilder(np.maximum(delta, 0), 14), reference_wilder(np.maximum(-delta, 0), 14)
        expected = 100 * gain / (gain + loss)
        np.testing.assert_allclose(compute_rsi(close).to_numpy(), expected, equal_nan=True, atol=1e-12)

    def test_atr_against_independent_true_range_and_wilder_seed(self):
        frame = self.frame()
        tr = []
        for i in range(len(frame)):
            high, low = frame.High.iloc[i], frame.Low.iloc[i]
            previous = frame.Close.iloc[i-1] if i else frame.Close.iloc[i]
            tr.append(max(high-low, abs(high-previous), abs(low-previous)))
        np.testing.assert_allclose(compute_atr(frame), reference_wilder(np.array(tr), 14), equal_nan=True)

    def test_macd_and_bollinger_against_direct_formulas(self):
        close = self.frame().Close
        def ema(span):
            out, value = [], close.iloc[0]
            for price in close:
                value += 2 / (span+1) * (price-value)
                out.append(value)
            return np.array(out)
        line, signal, hist = compute_macd(close)
        np.testing.assert_allclose(line, ema(12)-ema(26), atol=1e-12)
        low, mid, high = compute_bbands(close)
        window = close.iloc[-20:].to_numpy()
        self.assertAlmostEqual(mid.iloc[-1], window.mean())
        self.assertAlmostEqual(high.iloc[-1], window.mean()+2*window.std(ddof=0))
        self.assertAlmostEqual(low.iloc[-1], window.mean()-2*window.std(ddof=0))

    def test_live_and_actual_app_backtest_match_at_every_sampled_close(self):
        frame = self.frame()
        scores = load_app_score()(frame)
        for end in range(100, len(frame), 23):
            live = evaluate_bias(frame.iloc[:end+1])
            self.assertEqual(live.score, scores.iloc[end])
            components = bias_score_frame(frame.iloc[:end+1]).iloc[-1]
            self.assertEqual(live.trend_score, components.trend)
            self.assertEqual(live.momentum_score, components.momentum)

    def test_future_changes_cannot_rewrite_scores(self):
        frame = self.frame()
        changed = frame.copy()
        changed.iloc[777:] *= 1.05
        pd.testing.assert_frame_equal(bias_score_frame(frame).iloc[:777], bias_score_frame(changed).iloc[:777])

    def test_seed_requires_full_window(self):
        self.assertTrue(wilder_average(pd.Series([1., 2.]), 3).isna().all())
        self.assertAlmostEqual(wilder_average(pd.Series([1., 2., 6., 9.]), 3).iloc[3], 5.)

    def test_long_backtest_fetches_longer_higher_timeframe_context(self):
        source = Path("forex_web_app_streamlit_v14_alert_decision.py").read_text(encoding="utf-8")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_filter_period_for_tf")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "app-period-test", "exec"), namespace)
        choose = namespace["_filter_period_for_tf"]
        self.assertEqual(choose("4 Saat", "365d"), "425d")
        self.assertEqual(choose("1 Saat", "365d"), "385d")
        self.assertEqual(choose("15 Dakika", "365d"), "375d")
        self.assertEqual(choose("4 Saat", "60d"), "120d")


if __name__ == "__main__":
    unittest.main()
