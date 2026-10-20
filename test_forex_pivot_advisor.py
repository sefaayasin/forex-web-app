import ast
from pathlib import Path
import unittest
from unittest.mock import Mock

import pandas as pd

from forex_pivot_advisor import advise, calendar_available, closed_bars, recent_bar

NOW = pd.Timestamp("2026-10-20 12:37", tz="UTC")


def snapshot(side="LONG"):
    sign = 1 if side == "LONG" else -1
    return dict(symbol="EURUSD=X", side=side, entry=1.1, price=1.1, stop=1.1 - sign * .001,
                target=1.1 + sign * .002, pip=.0001, fresh=True, pivot_fresh=True,
                scores=dict.fromkeys(("4H", "1H", "15M", "5M"), sign * 60),
                ml={"status": "ready", "probability_up": .8 if sign == 1 else .2,
                    "as_of": NOW.floor("h") - pd.Timedelta(hours=1)},
                distance_atr=.1, response_side=side, reversal_warnings=[])


class AdvisorTests(unittest.TestCase):
    def read(self, value=None, **kwargs):
        options = dict(now=NOW, news_known=True, cost_pips=1)
        options.update(kwargs)
        return advise(snapshot() if value is None else value, **options)

    def test_full_agreement_releases_long_and_short_without_probability_claim(self):
        for side in ("LONG", "SHORT"):
            with self.subTest(side=side):
                result = self.read(snapshot(side))
                self.assertTrue(result["technical_ready"])
                self.assertEqual(result["alignment"], 100)
                self.assertEqual(result["action"], f"{side} GİR")
                self.assertIsNone(result["probability"])
                self.assertEqual(result["blockers"], [])
                self.assertAlmostEqual(result["net_rr"], 19 / 11)

    def test_high_ml_probability_never_overrides_opposite_hourly_trend(self):
        value = snapshot()
        value["scores"]["1H"] = -60
        value["ml"]["probability_up"] = .99
        result = self.read(value)
        self.assertFalse(result["technical_ready"])
        self.assertGreaterEqual(result["alignment"], 70)
        self.assertTrue(any("1H yön" in b for b in result["blockers"]))

    def test_missing_optional_ml_and_five_minute_direction_allow_technical_candidate(self):
        value = snapshot()
        value["ml"] = {}
        value["scores"]["5M"] = 0
        result = self.read(value)
        self.assertEqual(result["alignment"], 75)
        self.assertTrue(result["technical_ready"])
        self.assertEqual(result["action"], "LONG GİR")

    def test_strong_opposite_ml_blocks_even_above_alignment_threshold(self):
        for side, probability in (("LONG", .30), ("SHORT", .70)):
            value = snapshot(side)
            value["ml"]["probability_up"] = probability
            result = self.read(value)
            self.assertGreaterEqual(result["alignment"], 70)
            self.assertEqual(result["action"], "BEKLE")
            self.assertTrue(any("ML güçlü" in b for b in result["blockers"]))

    def test_market_entry_and_net_reward_are_recomputed_at_current_price(self):
        result = self.read({**snapshot(), "price": 1.1001})
        self.assertEqual(result["action"], "LONG GİR")
        self.assertEqual(result["entry"], 1.1001)
        self.assertEqual(result["pivot_entry"], 1.1)
        self.assertAlmostEqual(result["net_rr"], 18 / 12)
        too_late = self.read({**snapshot(), "price": 1.1019})
        self.assertEqual(too_late["action"], "BEKLE")
        self.assertLess(too_late["net_rr"], 1)

    def test_unavailable_news_and_costs_are_not_assumed_safe(self):
        result = self.read(news_known=False, cost_pips=None)
        self.assertFalse(result["technical_ready"])
        self.assertTrue(any("takvimi" in b for b in result["blockers"]))
        self.assertTrue(any("komisyon" in b for b in result["blockers"]))

    def test_news_before_and_after_blocks(self):
        for minutes in (-15, 0, 60):
            with self.subTest(minutes=minutes):
                result = self.read(news_minutes=minutes, news_title="USD CPI")
                self.assertFalse(result["technical_ready"])
                self.assertTrue(any("CPI" in b for b in result["blockers"]))

    def test_stale_inputs_pivot_and_reversal_each_block(self):
        for change in ({"fresh": False}, {"pivot_fresh": False},
                       {"reversal_warnings": ["15M: ters momentum"]},
                       {"reversal_warnings": None}, {"response_side": "NONE"},
                       {"distance_atr": .3}, {"distance_atr": float("nan")}):
            with self.subTest(change=change):
                self.assertFalse(self.read({**snapshot(), **change})["technical_ready"])

    def test_bad_price_geometry_nan_and_infinite_cost_are_rejected(self):
        for change in ({"stop": 1.101}, {"entry": float("nan")}, {"price": None}, {"price": float("nan")},
                       {"pip": 0}, {"target": 1.099}):
            self.assertFalse(self.read({**snapshot(), **change})["technical_ready"])
        for cost in (float("nan"), float("inf"), -1, 15):
            self.assertFalse(self.read(cost_pips=cost)["technical_ready"])

    def test_gross_reward_risk_can_pass_while_net_fails(self):
        result = self.read({**snapshot(), "target": 1.1011})
        self.assertLess(result["net_rr"], 1)
        self.assertFalse(result["technical_ready"])

    def test_weekend_and_rollover_block_full_agreement(self):
        for now in (pd.Timestamp("2026-10-24 12:00Z"), pd.Timestamp("2026-10-20 21:15Z")):
            self.assertFalse(self.read(now=now)["technical_ready"])

    def test_missing_data_does_not_crash_or_invent_direction(self):
        result = self.read({"symbol": "EURUSD=X", "error": "Kaynak yok"})
        self.assertEqual(result["alignment"], 0)
        self.assertEqual(result["action"], "BEKLE")
        self.assertFalse(result["technical_ready"])

    def test_stale_ml_is_excluded_from_alignment(self):
        value = snapshot()
        value["ml"]["as_of"] -= pd.Timedelta(days=1)
        self.assertFalse(self.read(value)["checks"][4]["ok"])


class TimestampTests(unittest.TestCase):
    def test_hourly_closed_bar_remains_fresh_between_hour_boundaries(self):
        self.assertTrue(recent_bar(pd.Timestamp("2026-10-20 11:00Z"), NOW, 60))
        self.assertFalse(recent_bar(pd.Timestamp("2026-10-20 12:00Z"), NOW, 60))
        self.assertFalse(recent_bar(pd.Timestamp("2026-10-20 09:00Z"), NOW, 60))
        self.assertFalse(recent_bar(NOW + pd.Timedelta(minutes=1), NOW, 15, closed=False))

    def test_closed_bars_keep_last_completed_and_drop_forming_and_future(self):
        bars = pd.DataFrame({"Close": [1, 2, 3]}, index=pd.date_range("2026-10-20 12:15", periods=3, freq="15min"))
        result = closed_bars(bars, NOW, 15)
        self.assertEqual(result["Close"].tolist(), [1])
        self.assertEqual(len(closed_bars(result, NOW, 15)), 1)

    def test_calendar_rejects_missing_impact_and_outdated_events(self):
        events = pd.DataFrame({"time": [NOW + pd.Timedelta(hours=1)], "currency": ["USD"], "impact": [3]})
        self.assertTrue(calendar_available(events, NOW))
        self.assertFalse(calendar_available(events.drop(columns="impact"), NOW))
        events["time"] -= pd.Timedelta(days=10)
        self.assertFalse(calendar_available(events, NOW))
        self.assertFalse(calendar_available(None, NOW))


class RefreshIntegrationTests(unittest.TestCase):
    def test_snapshot_builds_timeframes_and_keeps_failed_symbols_visible(self):
        import numpy as np
        from forex_analysis import market_structure_frame, bias_score_frame
        from forex_config import TIMEFRAMES
        from forex_freshness import reversal_warnings
        from forex_pivot_plan import atr14, pivot_plan

        tree = ast.parse(Path("forex_web_app_streamlit_v14_alert_decision.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_pivot_advisor_snapshot")
        now = pd.Timestamp.now(tz="UTC")

        def fetch(symbol, interval, period):
            minutes = {"15m": 15, "5m": 5, "60m": 60, "4h": 240}[interval]
            index = pd.date_range(end=now.floor(f"{minutes}min"), periods=300, freq=f"{minutes}min")
            close = 1.102 + np.sin(np.arange(300) / 20) * .0001
            return pd.DataFrame({"Open": close - .00001, "High": close + .0001,
                                 "Low": close - .0001, "Close": close}, index=index)

        levels = {"S2": 1.09, "S1": 1.095, "P": 1.1, "R1": 1.105, "R2": 1.11,
                  "source_day": (now.tz_localize(None).normalize() - pd.offsets.BDay(1))}
        namespace = dict(pd=pd, TIMEFRAMES=TIMEFRAMES, get_pip_size=lambda sym: .0001,
                         summary_pivots=lambda sym: levels, fetch_ohlc=fetch, closed_bars=closed_bars,
                         recent_bar=recent_bar, atr14=atr14, pivot_plan=pivot_plan,
                         _utc_index_df=lambda frame: frame, market_structure_frame=market_structure_frame,
                         score_series_for_backtest=lambda frame: bias_score_frame(frame)["score"],
                         reversal_warnings=reversal_warnings, has_research_model=lambda *args: False)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "snapshot", "exec"), namespace)
        result = namespace[fn.name]("EURUSD=X", "Yahoo Finance")
        self.assertNotIn("error", result)
        self.assertTrue(result["fresh"])
        self.assertTrue(result["pivot_fresh"])
        self.assertEqual(set(result["scores"]), {"4H", "1H", "15M", "5M"})
        self.assertEqual(result["side"], "LONG")
        namespace["fetch_ohlc"] = Mock(side_effect=RuntimeError("provider unavailable"))
        failed = namespace[fn.name]("EURUSD=X", "Yahoo Finance")
        self.assertEqual(failed["symbol"], "EURUSD=X")
        self.assertFalse(failed["fresh"])
        self.assertIn("provider unavailable", failed["error"])

    def test_refresh_clears_analysis_and_all_price_and_calendar_caches(self):
        tree = ast.parse(Path("forex_web_app_streamlit_v14_alert_decision.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "refresh_pivot_advisor")
        names = ("_fetch_ohlc_yahoo", "_fetch_ohlc_mt5", "_fetch_intraday_history_yahoo", "fetch_mt5_quote",
                 "fetch_last_price", "fetch_price_change", "scan_pivot_advisor", "cached_calendar")
        namespace = {name: Mock() for name in names}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "refresh", "exec"), namespace)
        namespace["refresh_pivot_advisor"]()
        for name in names:
            namespace[name].clear.assert_called_once_with()

    def test_summary_wrapper_wires_small_profit_panel(self):
        from types import SimpleNamespace
        from typing import Optional
        from unittest.mock import patch
        tree = ast.parse(Path("forex_web_app_streamlit_v14_alert_decision.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_summary_pivot_opportunities")
        self.assertTrue(fn.decorator_list)
        fn.decorator_list = []
        names = ("scan_pivot_advisor", "build_pivot_advisor_snapshot", "pivot_live_quote", "cached_calendar",
                 "pair_news", "pivot_live_cost", "estimate_pip_value_per_lot_usd", "refresh_pivot_advisor")
        namespace = {name: Mock() for name in names}
        namespace.update(pd=pd, Optional=Optional, st=SimpleNamespace(session_state={}))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "wrapper", "exec"), namespace)
        with patch("forex_pivot_panel.render_pivot_panel") as render:
            namespace[fn.name](None, account_usd=1234.)
            kwargs = render.call_args.kwargs
            self.assertEqual(kwargs["account_usd"], 1234.)
            self.assertEqual(kwargs["provider"], "Yahoo Finance")
            self.assertIs(kwargs["get_quote"], namespace["pivot_live_quote"])
            kwargs["scan"]("EURUSD=X", "Yahoo Finance")
            namespace["scan_pivot_advisor"].assert_called_once()
            kwargs["scan"]("EURUSD=X", "MetaTrader 5")
            namespace["build_pivot_advisor_snapshot"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
