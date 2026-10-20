import unittest
import pandas as pd

from forex_pivot_live import live_plan
from test_forex_pivot_advisor import snapshot, NOW


def live_snapshot(side="LONG"):
    sign = 1 if side == "LONG" else -1
    return {**snapshot(side), "entry": 1.1 - sign * .0001, "atr": .001,
            "provider": "Yahoo Finance", "as_of": NOW, "signal_bar": NOW.floor("15min")}


class LivePlanTests(unittest.TestCase):
    def read(self, side="LONG", snap=None, quote=None, **kwargs):
        options = dict(now=NOW, news_known=True, cost_pips=1., account_usd=1000.,
                       max_loss_usd=5., pip_value_per_lot=10.)
        options.update(kwargs)
        return live_plan(live_snapshot(side) if snap is None else snap,
                         {"price": 1.1, "time": NOW} if quote is None else quote, **options)

    def test_three_net_pips_for_both_sides_keeps_original_selector(self):
        for side in ("LONG", "SHORT"):
            with self.subTest(side=side):
                result = self.read(side)
                sign = 1 if side == "LONG" else -1
                self.assertEqual(result["action"], side + " GİR")
                self.assertAlmostEqual(sign * (result["target"] - result["entry"]) / .0001 - result["cost_pips"], 3)
                self.assertLessEqual(result["planned_loss_usd"], 5)
                self.assertGreater(result["expected_profit_usd"], 0)
                self.assertLess(result["small_target_rr"], 1)
                self.assertNotEqual(result["target"], result["pivot_target"])

    def test_old_or_future_quote_old_analysis_and_csv_cannot_release(self):
        for stamp in (NOW - pd.Timedelta(seconds=121), NOW + pd.Timedelta(seconds=1), None):
            self.assertEqual(self.read(quote={"price": 1.1, "time": stamp})["action"], "BEKLE")
        self.assertEqual(self.read(snap={**live_snapshot(), "as_of": NOW-pd.Timedelta(seconds=91)})["action"], "BEKLE")
        self.assertEqual(self.read(snap={**live_snapshot(), "provider": "Broker CSV"})["action"], "BEKLE")

    def test_late_price_is_not_chased_even_inside_one_atr(self):
        result = self.read(quote={"price": 1.1002, "time": NOW})
        self.assertEqual(result["action"], "BEKLE")
        self.assertTrue(any("kovalama" in b for b in result["blockers"]))

    def test_anchor_survives_new_analysis_of_same_response_bar(self):
        result = self.read(snap={**live_snapshot(), "price": 1.1002},
                           quote={"price": 1.1002, "time": NOW}, anchor_price=1.1)
        self.assertEqual(result["action"], "BEKLE")

    def test_crossing_pivot_entry_invalidates_old_plan(self):
        result = self.read(quote={"price": 1.09985, "time": NOW})
        self.assertEqual(result["action"], "BEKLE")
        self.assertTrue(any("diğer taraf" in b for b in result["blockers"]))

    def test_zero_risk_missing_pip_value_and_min_lot_budget_block(self):
        for change in ({"max_loss_usd": 0}, {"max_loss_usd": .01}, {"pip_value_per_lot": None},
                       {"max_loss_usd": 1001}, {"cost_pips": None}):
            self.assertEqual(self.read(**change)["action"], "BEKLE")

    def test_technical_news_and_cost_gates_are_preserved(self):
        self.assertEqual(self.read(news_known=False)["action"], "BEKLE")
        self.assertEqual(self.read(news_minutes=10)["action"], "BEKLE")
        self.assertEqual(self.read(cost_pips=4)["action"], "BEKLE")
        value = live_snapshot()
        value["scores"]["1H"] = -60
        self.assertEqual(self.read(snap=value)["action"], "BEKLE")

    def test_mt5_short_ask_levels_do_not_double_count_spread(self):
        for side in ("LONG", "SHORT"):
            r = self.read(side, quote={"price": 1.1, "bid": 1.1, "ask": 1.10005, "time": NOW})
            spread = .00005 if side == "SHORT" else 0
            self.assertAlmostEqual(r["broker_target"], r["target"] + spread)
            self.assertAlmostEqual(r["broker_stop"], r["stop"] + spread)
            actual_move = (r["broker_target"] - r["broker_entry"]) * (1 if side == "LONG" else -1) / .0001
            # 1 pip total cost = .5 spread + .5 commission.
            self.assertAlmostEqual(actual_move - .5, 3)

    def test_expiry_uses_earliest_data_deadline(self):
        result = self.read(quote={"price": 1.1, "time": NOW-pd.Timedelta(seconds=100)})
        self.assertEqual(result["valid_until"], NOW+pd.Timedelta(seconds=20))


class PanelTests(unittest.TestCase):
    def test_live_panel_releases_and_then_waits_for_news_and_zero_budget(self):
        from streamlit.testing.v1 import AppTest
        script = '''
import pandas as pd
import streamlit as st
from unittest.mock import Mock
from forex_pivot_panel import render_pivot_panel
from test_forex_pivot_live import live_snapshot
st.checkbox("Test haber", key="test_news")
now = pd.Timestamp.now(tz="UTC")
def scan(sym, provider):
    s = live_snapshot("SHORT" if sym == "USDJPY=X" else "LONG")
    s.update(symbol=sym, as_of=now, signal_bar=now.floor("15min"))
    s["ml"]["as_of"] = now.floor("h")-pd.Timedelta(hours=1)
    return s
events = pd.DataFrame({"time": [now+pd.Timedelta(hours=2)], "currency": ["USD"], "impact": [3]})
render_pivot_panel(scan=scan, get_quote=lambda *args: {"price": 1.1, "time": now},
    get_calendar=lambda: (events, True, now, {}),
    get_news=lambda *args: (10,"CPI") if st.session_state.get("test_news") else (None,""),
    get_cost=lambda *args: 1., get_pip_value=lambda *args: 10., refresh=Mock(),
    provider="Yahoo Finance", account_usd=1000.)
'''
        # Fix session-hour checks to a deterministic open-market time.
        script = script.replace('from forex_pivot_panel import render_pivot_panel',
                                'from forex_pivot_panel import render_pivot_panel\n'
                                'import forex_pivot_panel as panel\n'
                                'from forex_pivot_live import live_plan as real_plan\n'
                                'from test_forex_pivot_advisor import NOW\n'
                                'def fixed_plan(snapshot, quote, **kwargs):\n'
                                '    kwargs["now"] = NOW\n'
                                '    snapshot["as_of"] = NOW\n'
                                '    snapshot["ml"]["as_of"] = NOW.floor("h")-pd.Timedelta(hours=1)\n'
                                '    quote["time"] = NOW\n'
                                '    return real_plan(snapshot, quote, **kwargs)\n'
                                'panel.live_plan = fixed_plan')
        script += '\npanel.live_plan = real_plan\n'
        app = AppTest.from_string(script).run(timeout=15)
        self.assertFalse(list(app.exception), list(app.exception))
        self.assertTrue(app.success)
        self.assertTrue(app.dataframe[0].value["Karar"].str.contains("GİR").all())
        self.assertEqual(len(app.metric), 3)
        app.checkbox[0].check().run(timeout=15)
        self.assertFalse(list(app.exception), list(app.exception))
        self.assertTrue(app.dataframe[0].value["Karar"].eq("BEKLE").all())
        app.checkbox[0].uncheck().run(timeout=15)
        app.number_input[0].set_value(0).run(timeout=15)
        self.assertFalse(list(app.exception), list(app.exception))
        self.assertTrue(app.dataframe[0].value["Karar"].eq("BEKLE").all())


if __name__ == "__main__":
    unittest.main()
