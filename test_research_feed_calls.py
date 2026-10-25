import unittest

import pandas as pd

from feed_replay import ReplayFeed, resample_4h_like_app
from research_feed_calls import decision_times, mark_new_calls, signed_moves


def five_minute(start="2026-08-03 08:00", periods=400, step=0.0001):
    index = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    opens = [1.1 + i * step for i in range(periods)]
    return pd.DataFrame({"Open": opens, "High": opens, "Low": opens, "Close": opens, "Volume": 1.0}, index=index)


class FeedCallTests(unittest.TestCase):
    def test_decision_times_are_open_quarter_hours(self):
        bars = five_minute(periods=12)
        times = decision_times(bars, bars.index[0], bars.index[-1] + pd.Timedelta(minutes=5))
        self.assertTrue(all(t.minute % 15 == 0 for t in times))
        self.assertEqual(len(times), 4)

    def test_signed_move_follows_the_card_side(self):
        bars = five_minute()
        up = signed_moves(bars, bars.index[0], 1, 0.0001)
        down = signed_moves(bars, bars.index[0], -1, 0.0001)
        self.assertAlmostEqual(up["move_1h"], 12.0)
        self.assertAlmostEqual(down["move_1h"], -12.0)
        self.assertAlmostEqual(up["move_15m"], 3.0)

    def test_repeated_scans_of_the_same_card_count_once(self):
        t = pd.date_range("2026-08-03 08:00", periods=5, freq="15min", tz="UTC")
        rows = pd.DataFrame({"pair": "EURUSD", "tf": "15 Dakika", "time": [t[0], t[1], t[2], t[3], t[4]],
                             "side": [1, 1, 0, 1, -1]})
        self.assertEqual(mark_new_calls(rows).new_call.tolist(), [True, False, False, True, True])
        gap = rows.drop(index=1)
        self.assertEqual(mark_new_calls(gap).new_call.tolist(), [True, False, True, True])

    def test_feed_never_returns_bars_that_open_after_now(self):
        bars = five_minute()
        feed = ReplayFeed(bars, bars, bars)
        feed.now = bars.index[100]
        window = feed.fetch_ohlc("EURUSD=X", "5m", "5d")
        self.assertEqual(window.index[-1], feed.now)
        self.assertTrue((window.index <= feed.now).all())

    def test_app_style_4h_bins_end_at_the_label(self):
        hourly = five_minute(periods=10).resample("1h").first().dropna()
        hourly.index = pd.date_range("2026-08-03 13:00", periods=len(hourly), freq="1h", tz="UTC")
        bins = resample_4h_like_app(hourly)
        self.assertIn(pd.Timestamp("2026-08-03 16:00", tz="UTC"), bins.index)


if __name__ == "__main__":
    unittest.main()
