import unittest

import numpy as np
import pandas as pd

from research_meta_label import (
    alignment_state,
    app_rule_mask,
    assign_split,
    closed_values,
    near_news_flags,
    primary_signals,
    resample_bars,
    simulate,
)


def five_minute_arrays(rows, start="2024-03-08 13:30"):
    times = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    o, h, l, c = (np.array(col, dtype=float) for col in zip(*rows))
    return o, h, l, c, times.values


class MetaLabelTests(unittest.TestCase):
    def test_state_needs_both_timeframes_aligned(self):
        state = alignment_state(np.array([30, 30, -40, 10]), np.array([25, -30, -25, 50]))
        self.assertEqual(state.tolist(), [1, 0, -1, 0])

    def test_signal_fires_on_new_alignment_or_flip_only(self):
        self.assertEqual(primary_signals(np.array([0, 1, 1, 0, -1, 1, 1])).tolist(), [1, 4, 5])
        self.assertEqual(primary_signals(np.array([1, 1])).tolist(), [0])

    def test_closed_values_use_only_bars_that_have_closed(self):
        index = pd.date_range("2024-01-01 00:00", periods=3, freq="15min", tz="UTC")
        series = pd.Series([1.0, 2.0, 3.0], index=index)
        times = pd.DatetimeIndex(pd.to_datetime(["2024-01-01 00:10", "2024-01-01 00:15", "2024-01-01 00:29"], utc=True))
        values = closed_values(times, series, pd.Timedelta(minutes=15), pd.Timedelta(minutes=30))
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(values[1:].tolist(), [1.0, 1.0])

    def test_closed_values_go_stale_after_a_gap(self):
        index = pd.DatetimeIndex(pd.to_datetime(["2024-01-01 00:00"], utc=True))
        times = pd.DatetimeIndex(pd.to_datetime(["2024-01-01 01:00"], utc=True))
        values = closed_values(times, pd.Series([5.0], index=index), pd.Timedelta(minutes=15), pd.Timedelta(minutes=30))
        self.assertTrue(np.isnan(values[0]))

    def test_resampled_hour_bars_are_utc_aligned_ohlc(self):
        index = pd.date_range("2024-01-01 00:00", periods=8, freq="15min", tz="UTC")
        bars = pd.DataFrame({"Open": range(1, 9), "High": range(11, 19), "Low": range(0, 8), "Close": range(2, 10)},
                            index=index, dtype=float)
        hourly = resample_bars(bars.drop(index[5]), "1h")
        self.assertEqual(hourly.index.tolist(), [index[0], index[4]])
        self.assertEqual(hourly.iloc[0].tolist(), [1.0, 14.0, 0.0, 5.0])
        self.assertEqual(hourly.iloc[1].tolist(), [5.0, 18.0, 4.0, 9.0])

    def test_long_trade_hits_target_and_stop_wins_ties(self):
        o, h, l, c, t = five_minute_arrays([[1.0, 1.001, .9995, 1.0008], [1.0008, 1.002, 1.0005, 1.0018]])
        self.assertEqual(simulate(o, h, l, c, t, 0, 1, .001)[1:], (1, "target"))
        o, h, l, c, t = five_minute_arrays([[1.0, 1.002, .998, 1.001]])
        price, _, reason = simulate(o, h, l, c, t, 0, 1, .001)
        self.assertEqual(reason, "stop")
        self.assertAlmostEqual(price, .999)

    def test_timeout_stays_inside_four_hours(self):
        o, h, l, c, t = five_minute_arrays([[1.0, 1.0002, .9998, 1.0001]] * 60)
        _, exit_pos, reason = simulate(o, h, l, c, t, 0, 1, .01)
        self.assertEqual((exit_pos, reason), (47, "time"))

    def test_news_flag_only_for_usd_pairs_within_an_hour(self):
        releases = pd.DatetimeIndex(pd.to_datetime(["2024-03-08 13:30"], utc=True))
        times = pd.DatetimeIndex(pd.to_datetime(["2024-03-08 12:45", "2024-03-08 14:20", "2024-03-08 15:00"], utc=True))
        self.assertEqual(near_news_flags("EURUSD", times, releases).tolist(), [1, 1, 0])
        self.assertEqual(near_news_flags("EURGBP", times, releases).tolist(), [0, 0, 0])

    def test_trades_crossing_a_split_boundary_are_purged(self):
        frame = pd.DataFrame({
            "entry_time": pd.to_datetime(["2018-12-31 22:00", "2018-12-31 23:00", "2019-06-01 10:00"], utc=True),
            "exit_time": pd.to_datetime(["2018-12-31 23:30", "2019-01-01 01:00", "2019-06-01 12:00"], utc=True),
        })
        self.assertEqual(assign_split(frame).tolist(), ["train", "purged", "validation"])

    def test_app_rule_needs_higher_timeframes_and_entry_threshold(self):
        frame = pd.DataFrame({"s4h": [30, 30, 10], "s1h": [30, 30, 30], "s5": [60, 59, 80]})
        self.assertEqual(app_rule_mask(frame).tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
