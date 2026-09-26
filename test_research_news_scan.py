import unittest
from datetime import date

import numpy as np
import pandas as pd

from research_news_scan import (
    load_app_score,
    new_york_to_utc,
    nfp_release_date,
    simulate_trade,
    state_from_scores,
)


def five_minute_bars(rows, start="2024-03-08 13:30"):
    index = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=index)


class NewsScanTests(unittest.TestCase):
    def test_nfp_rule_matches_published_release_dates(self):
        self.assertEqual(nfp_release_date(2023, 7), date(2023, 8, 4))
        self.assertEqual(nfp_release_date(2023, 11), date(2023, 12, 8))
        self.assertEqual(nfp_release_date(2021, 6), date(2021, 7, 2))
        self.assertEqual(nfp_release_date(2026, 8), date(2026, 9, 4))
        self.assertEqual(nfp_release_date(2024, 2).weekday(), 4)

    def test_new_york_release_time_follows_daylight_saving(self):
        self.assertEqual(new_york_to_utc(date(2024, 1, 5), 8, 30), pd.Timestamp("2024-01-05 13:30", tz="UTC"))
        self.assertEqual(new_york_to_utc(date(2024, 7, 5), 8, 30), pd.Timestamp("2024-07-05 12:30", tz="UTC"))

    def test_state_needs_both_timeframes_past_the_label_threshold(self):
        self.assertEqual(state_from_scores(30, 25), 1)
        self.assertEqual(state_from_scores(-40, -26), -1)
        self.assertEqual(state_from_scores(30, 10), 0)
        self.assertEqual(state_from_scores(30, -30), 0)
        self.assertEqual(state_from_scores(np.nan, 30), 0)

    def test_long_trade_reaches_target(self):
        bars = five_minute_bars([[1.0, 1.001, 0.9995, 1.0008], [1.0008, 1.0020, 1.0005, 1.0018]])
        entry, exit_price, reason = simulate_trade(bars, bars.index[0], 1, 0.001)
        self.assertEqual((entry, reason), (1.0, "target"))
        self.assertAlmostEqual(exit_price, 1.0015)

    def test_stop_is_taken_first_when_both_levels_sit_in_one_bar(self):
        bars = five_minute_bars([[1.0, 1.002, 0.998, 1.001]])
        _, exit_price, reason = simulate_trade(bars, bars.index[0], 1, 0.001)
        self.assertEqual(reason, "stop")
        self.assertAlmostEqual(exit_price, 0.999)

    def test_short_trade_gap_through_stop_fills_at_open(self):
        bars = five_minute_bars([[1.0, 1.0005, 0.9995, 1.0], [1.003, 1.004, 1.002, 1.003]])
        _, exit_price, reason = simulate_trade(bars, bars.index[0], -1, 0.001)
        self.assertEqual((exit_price, reason), (1.003, "stop_gap"))

    def test_timeout_exits_at_last_close_inside_holding_window(self):
        rows = [[1.0, 1.0002, 0.9998, 1.0001]] * 60
        bars = five_minute_bars(rows)
        _, exit_price, reason = simulate_trade(bars, bars.index[0], 1, 0.01)
        self.assertEqual((exit_price, reason), (1.0001, "time"))

    def test_app_score_function_compiles_without_streamlit(self):
        close = 1 + np.cumsum(np.random.default_rng(3).normal(0, 0.0005, 400))
        frame = pd.DataFrame(
            {"Open": close, "High": close + 0.0005, "Low": close - 0.0005, "Close": close},
            index=pd.date_range("2024-01-01", periods=400, freq="15min", tz="UTC"),
        )
        score = load_app_score()(frame)
        self.assertTrue(score.between(-100, 100).all())
        self.assertGreater(len(score), 0)


if __name__ == "__main__":
    unittest.main()
