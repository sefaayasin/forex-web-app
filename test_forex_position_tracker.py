import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import forex_storage as storage
from forex_position_chart import make_plan, observe_plan, position_figure

NOW = pd.Timestamp("2026-10-23 10:00", tz="UTC")


def plan(side="SHORT", **changes):
    args = dict(symbol="EURUSD", provider="MetaTrader 5", side=side, entry=1.1,
                stop=1.101 if side == "SHORT" else 1.099,
                target=1.0995 if side == "SHORT" else 1.1005, now=NOW, lot=.02)
    args.update(changes)
    return make_plan(**args)


def candles():
    index = pd.date_range(NOW - pd.Timedelta(hours=3), periods=37, freq="5min")
    c = np.linspace(1.101, 1.0998, len(index))
    return pd.DataFrame({"Open": c + .00003, "High": c + .00012,
                         "Low": c - .00012, "Close": c}, index=index)


class TrackingMathTests(unittest.TestCase):
    def test_reversed_or_equal_levels_are_rejected(self):
        for side in ("LONG", "SHORT"):
            for changes in ({"stop": 1.1}, {"target": 1.1}, {"entry": 0.}, {"stop": float("nan")}):
                with self.subTest(side=side, changes=changes), self.assertRaises(ValueError):
                    plan(side, **changes)
        with self.assertRaises(ValueError):
            plan("SHORT", target=1.102)

    def test_directional_pips_and_target_distance_for_both_sides(self):
        for side, price in (("LONG", 1.1002), ("SHORT", 1.0998)):
            watched = plan(side)
            view = observe_plan(watched, {"price": price, "time": NOW+pd.Timedelta(seconds=10)},
                                now=NOW+pd.Timedelta(seconds=20))
            self.assertTrue(view["fresh"])
            self.assertAlmostEqual(view["move_pips"], 2.)
            self.assertAlmostEqual(view["target_distance_pips"], 3.)
            self.assertEqual(view["status"], "İzleniyor")
            self.assertEqual(watched["entry"], 1.1)

    def test_stale_future_pre_tracking_or_other_source_cannot_show_live_progress(self):
        for quote in ({"price": 1.0998, "time": NOW - pd.Timedelta(minutes=5)},
                      {"price": 1.0998, "time": NOW + pd.Timedelta(minutes=5)},
                      {"price": 1.0998, "time": NOW - pd.Timedelta(seconds=1)},
                      {"price": 1.0998, "time": NOW, "provider": "Yahoo Finance"},
                      {"price": 1.0998}, {"price": float("inf"), "time": NOW}):
            view = observe_plan(plan(), quote, now=NOW + pd.Timedelta(seconds=20))
            self.assertFalse(view["fresh"])
            self.assertIsNone(view["move_pips"])

    def test_price_in_stop_target_zone_never_claims_fill(self):
        for price, expected in ((1.1012, "Fiyat stop bölgesinde"), (1.0994, "Fiyat hedef bölgesinde")):
            view = observe_plan(plan(), {"price": price, "time": NOW}, now=NOW)
            self.assertEqual(view["status"], expected)
            self.assertNotIn("realized_pnl", view)

    def test_four_hour_timeout_remains_a_watch_plan(self):
        later = NOW + pd.Timedelta(hours=4)
        view = observe_plan(plan(), {"price": 1.1, "time": later}, now=later)
        self.assertIn("4 saat doldu", view["status"])

    def test_jpy_pip_convention(self):
        watched = plan(symbol="USDJPY", entry=150., stop=150.2, target=149.9)
        view = observe_plan(watched, {"price": 149.95, "time": NOW}, now=NOW)
        self.assertAlmostEqual(view["move_pips"], 5.)


class FigureTests(unittest.TestCase):
    def test_short_and_long_boxes_use_true_level_geometry(self):
        for side in ("LONG", "SHORT"):
            watched = plan(side)
            fig = position_figure(candles(), watched, now=NOW)
            boxes = [s for s in fig.layout.shapes if s.type == "rect"]
            self.assertEqual(len(boxes), 2)
            self.assertEqual((boxes[0].y0, boxes[0].y1), tuple(sorted([watched["entry"], watched["stop"]])))
            self.assertEqual((boxes[1].y0, boxes[1].y1), tuple(sorted([watched["entry"], watched["target"]])))
            self.assertAlmostEqual((boxes[0].y1-boxes[0].y0)/(boxes[1].y1-boxes[1].y0), 2.)
            self.assertEqual(fig.data[0].type, "candlestick")

    def test_future_or_corrupt_candles_are_not_plotted(self):
        bars = candles()
        bars.loc[NOW+pd.Timedelta(minutes=5)] = [9., 9., 9., 9.]
        bars.loc[NOW-pd.Timedelta(minutes=5), "High"] = .5
        fig = position_figure(bars, plan(), now=NOW)
        self.assertEqual(len(fig.data[0].x), 36)
        self.assertTrue(all(pd.Timestamp(x).tz_convert("UTC")<=NOW for x in fig.data[0].x))

    def test_missing_bars_preserve_levels_without_inventing_candles(self):
        fig = position_figure(pd.DataFrame(), plan(), now=NOW)
        self.assertEqual(len(fig.data), 0)
        self.assertEqual(len([s for s in fig.layout.shapes if s.type=="rect"]), 2)
        self.assertTrue(any("alınamadı" in a.text for a in fig.layout.annotations))


class StorageTests(unittest.TestCase):
    def test_refresh_does_not_move_saved_levels_and_archive_allows_new_plan(self):
        with TemporaryDirectory() as tmp, patch.object(storage, "APP_DB_PATH", Path(tmp)/"watch.db"):
            initial = plan()
            record_id = storage.save_tracked_plan(initial)
            duplicate = plan(entry=1.1001, stop=1.1012, target=1.0996)
            self.assertEqual(storage.save_tracked_plan(duplicate), record_id)
            saved = storage.tracked_plans("MetaTrader 5")
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["entry"], initial["entry"])
            self.assertEqual(saved[0]["stop"], initial["stop"])
            self.assertEqual(storage.tracked_plans("Yahoo Finance"), [])
            storage.archive_tracked_plan(record_id)
            self.assertEqual(storage.tracked_plans(), [])
            self.assertNotEqual(storage.save_tracked_plan(duplicate), record_id)
            self.assertEqual(storage.tracked_plans()[0]["entry"], duplicate["entry"])
            self.assertEqual(json.loads(json.dumps(storage.tracked_plans()))[0]["entry"], duplicate["entry"])


class PanelTests(unittest.TestCase):
    def test_manual_save_refresh_and_archive_without_market_data(self):
        from streamlit.testing.v1 import AppTest
        script = '''
import pandas as pd
from forex_position_panel import render_position_tracker
render_position_tracker(readings=[], provider="MetaTrader 5",
    get_bars=lambda *args: pd.DataFrame(), get_quote=lambda *args: {})
'''
        with TemporaryDirectory() as tmp, patch.object(storage, "APP_DB_PATH", Path(tmp)/"watch.db"):
            app = AppTest.from_string(script).run(timeout=15)
            self.assertFalse(list(app.exception))
            app.number_input[0].set_value(1.1)
            app.number_input[1].set_value(1.099)
            app.number_input[2].set_value(1.101)
            next(b for b in app.button if b.label=="Seviyeleri kaydet ve takip et").click().run(timeout=15)
            self.assertFalse(list(app.exception), list(app.exception))
            self.assertEqual(len(storage.tracked_plans()), 1)
            self.assertTrue(any(m.value=="—" for m in app.metric))
            app.run(timeout=15)
            self.assertEqual(storage.tracked_plans()[0]["entry"], 1.1)
            next(b for b in app.button if b.label=="Takibi bitir").click().run(timeout=15)
            self.assertFalse(list(app.exception), list(app.exception))
            self.assertEqual(storage.tracked_plans(), [])

    def test_ready_plan_button_freezes_last_displayed_levels(self):
        from streamlit.testing.v1 import AppTest
        script = '''
import pandas as pd
from forex_position_panel import render_position_tracker
reading=dict(symbol="EURUSD=X",side="SHORT",entry=1.1,stop=1.101,target=1.0995,
             technical_ready=True,action="SHORT GİR",lot=.02)
render_position_tracker(readings=[reading],provider="MetaTrader 5",
    get_bars=lambda *args: pd.DataFrame(),get_quote=lambda *args: {})
'''
        with TemporaryDirectory() as tmp, patch.object(storage, "APP_DB_PATH", Path(tmp)/"watch.db"):
            app = AppTest.from_string(script).run(timeout=15)
            self.assertFalse(list(app.exception), list(app.exception))
            next(b for b in app.button if b.label=="Bu planı takibe al").click().run(timeout=15)
            self.assertFalse(list(app.exception), list(app.exception))
            saved = storage.tracked_plans()
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["side"], "SHORT")
            self.assertEqual(saved[0]["target"], 1.0995)
            self.assertTrue(next(b for b in app.button if b.label=="Bu planı takibe al").disabled)


if __name__ == "__main__":
    unittest.main()
