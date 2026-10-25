import unittest
import pandas as pd

from research_pivot_advisor import History, direction_after, simulate, news_at


NOW = pd.Timestamp("2025-01-07 12:00Z")


def bars():
    index = pd.date_range(NOW, periods=50, freq="5min")
    return pd.DataFrame({"Open": 100., "High": 100.5, "Low": 99.5, "Close": 100.}, index=index)


class ExitTests(unittest.TestCase):
    def test_long_target_and_short_target(self):
        for side in ("LONG", "SHORT"):
            data = bars()
            if side == "LONG":
                data.iloc[1, data.columns.get_loc("High")] = 103
                stop, target = 99, 102
            else:
                data.iloc[1, data.columns.get_loc("Low")] = 97
                stop, target = 101, 98
            result = simulate(data, NOW, side, 100, stop, target, 1)
            self.assertEqual(result["kind"], "target")
            self.assertEqual(result["gross_pips"], 2)

    def test_both_touched_uses_stop(self):
        data = bars()
        data.iloc[1, data.columns.get_loc("High")] = 103
        data.iloc[1, data.columns.get_loc("Low")] = 98
        result = simulate(data, NOW, "LONG", 100, 99, 102, 1)
        self.assertEqual(result["kind"], "stop")
        self.assertTrue(result["ambiguous"])
        self.assertEqual(result["gross_pips"], -1)

    def test_stop_gap_uses_adverse_open(self):
        data = bars()
        data.iloc[2] = [97, 98, 96, 97]
        result = simulate(data, NOW, "LONG", 100, 99, 102, 1)
        self.assertEqual(result["kind"], "stop_gap")
        self.assertEqual(result["gross_pips"], -3)

    def test_incomplete_or_missing_paths_are_not_invented(self):
        self.assertIsNone(simulate(bars().iloc[:20], NOW, "LONG", 100, 99, 102, 1))
        data = bars().drop(NOW + pd.Timedelta(minutes=15))
        self.assertIsNone(simulate(data, NOW, "LONG", 100, 99, 102, 1))

    def test_next_open_price_not_last_known_close_is_actual_fill(self):
        data = bars()
        data.iloc[0, data.columns.get_loc("Open")] = 100.2
        result = simulate(data, NOW, "LONG", 100, 99, 102, 1)
        self.assertAlmostEqual(result["actual_entry"], 100.2)
        self.assertAlmostEqual(result["entry_gap_pips"], .2)
        self.assertAlmostEqual(result["gross_pips"], -.2)

    def test_four_hour_timeout(self):
        result = simulate(bars(), NOW, "LONG", 100, 99, 102, 1)
        self.assertEqual(result["kind"], "time")
        self.assertEqual(result["exit_time"], NOW + pd.Timedelta(hours=4))

    def test_direction_endpoint_is_separate_from_stop_result(self):
        data = bars()
        data.loc[NOW + pd.Timedelta(hours=1), "Open"] = 101
        self.assertEqual(direction_after(data, NOW, 1, "LONG", 1), 1)
        self.assertEqual(direction_after(data, NOW, 1, "SHORT", 1), -1)


class CausalityTests(unittest.TestCase):
    def test_breakout_prefilter_is_a_necessary_condition_of_actual_response_rule(self):
        import numpy as np
        from forex_analysis import market_structure_frame
        rng = np.random.default_rng(20261020)
        close = 1.1 + np.cumsum(rng.normal(0, .0003, 3000))
        open_price = np.r_[close[0], close[:-1]]
        data = pd.DataFrame({"Open": open_price, "High": np.maximum(open_price, close) + .0001,
                             "Low": np.minimum(open_price, close) - .0001, "Close": close},
                            index=pd.date_range("2025-01-01", periods=3000, freq="15min", tz="UTC"))
        frame = market_structure_frame(data)
        for side in ("LONG", "SHORT"):
            response = frame.ResponseSide.eq(side)
            necessary = ((data.Close > data.High.shift()) & (data.Close > data.Open)) if side == "LONG" else (
                (data.Close < data.Low.shift()) & (data.Close < data.Open))
            self.assertGreater(response.sum(), 0)
            self.assertFalse((response & ~necessary).any())

    def test_forming_fifteen_minute_price_uses_only_completed_five_minute_data(self):
        history = History.__new__(History)
        history.five = pd.DataFrame({"Open": [1., 1.1, 999.], "High": [1.2, 1.3, 999.],
                                     "Low": [.9, 1., 999.], "Close": [1.1, 1.2, 999.], "Volume": [1., 1., 999.]},
                                    index=pd.date_range(NOW, periods=3, freq="5min"))
        history.fifteen = pd.DataFrame({"Open": [1.], "High": [999.], "Low": [.9],
                                        "Close": [999.], "Volume": [999.]}, index=pd.DatetimeIndex([NOW]))
        history.sources = {"15m": history.fifteen}
        history.now = NOW + pd.Timedelta(minutes=10)
        visible = history.raw("EURUSD", "15m", "10d")
        self.assertEqual(visible.Close.iloc[-1], 1.2)
        self.assertEqual(visible.High.iloc[-1], 1.3)
        self.assertEqual(visible.Volume.iloc[-1], 2.)

    def test_news_uses_pair_currency_and_preserves_post_release_blackout(self):
        events = pd.DataFrame({"time": [NOW - pd.Timedelta(minutes=10), NOW + pd.Timedelta(minutes=20)],
                               "currency": ["USD", "JPY"], "title": ["CPI", "BoJ"]})
        minutes, title = news_at(events, "EURUSD", NOW)
        self.assertEqual(minutes, -10)
        self.assertEqual(title, "USD CPI")
        self.assertEqual(news_at(events, "AUDCAD", NOW), (None, ""))


if __name__ == "__main__":
    unittest.main()
