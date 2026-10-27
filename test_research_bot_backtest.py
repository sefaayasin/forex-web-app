import unittest

import numpy as np
import pandas as pd

import research_bot_backtest as bt

J = bt.PAIRS.index("EURUSD")
STATS = {"pairs": {}, "pooled": {"ratios": {
    "0.25": {"ml": {"target_rate": 0.8}, "opposite": {"target_rate": 0.8}},
    "3.0": {"ml": {"target_rate": 0.7}, "opposite": {"target_rate": 0.7}},
}}}


def market(opens, highs=None, lows=None, prob=0.6, atr=0.0010, start="2023-01-02 10:00"):
    """Yalnız EURUSD'nin dolu olduğu yapay 5M piyasa; pazartesi sabahı (New York 05:00)."""
    n = len(opens)
    index = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    arrays = {k: np.full((n, len(bt.PAIRS)), np.nan) for k in ("Open", "High", "Low", "ATR", "Prob")}
    opens = np.asarray(opens, dtype=float)
    arrays["Open"][:, J] = opens
    arrays["High"][:, J] = opens + 0.00002 if highs is None else highs
    arrays["Low"][:, J] = opens - 0.00002 if lows is None else lows
    arrays["ATR"][:, J] = atr
    arrays["Prob"][:, J] = prob
    ny = index.tz_convert("America/New_York")
    usd = {c: np.ones(n) for c in ("USD", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "JPY")}
    return {"index": index, "ns": index.asi8, "ny_minutes": (ny.hour * 60 + ny.minute).to_numpy(),
            "ny_hour": ny.hour.to_numpy(), "ny_weekday": ny.weekday.to_numpy(), "usd": usd,
            "spread": np.zeros((24, len(bt.PAIRS))), **arrays}


class BotBacktestTests(unittest.TestCase):
    def test_cost_share_cap_blocks_expensive_entry_but_keeps_cheap_one(self):
        m = market(list(np.linspace(1.1, 1.102, 30)))
        m["spread"][:] = 1.5  # 2.2 pip including commission, versus 15 pip stop: >10%, <15%.
        self.assertTrue(bt.run(bt.Variant("allow", max_cost_share=.15), m, {}, STATS)["trades"])
        self.assertFalse(bt.run(bt.Variant("block", max_cost_share=.10), m, {}, STATS)["trades"])
        m["spread"][:] = 0.0
        self.assertTrue(bt.run(bt.Variant("cheap", max_cost_share=.10), m, {}, STATS)["trades"])

    def test_target_starts_trailing_and_exit_never_loses(self):
        # stop 15 pip; 12,5 $ risk ve 0,01 lot adımıyla hedef ≈ 5 pip; fiyat 20 pip çıkıp geri düşer
        path = list(np.linspace(1.1000, 1.1020, 30)) + list(np.linspace(1.1020, 1.0990, 30))
        trades = bt.run(bt.VARIANTS[0], market(path, prob=0.6), {}, STATS)["trades"]
        first = trades[0]
        self.assertEqual(first["side"], 1)
        self.assertTrue(first["trailing"])
        self.assertEqual(first["reason"], "TAKİP STOPU")
        self.assertGreater(first["net_usd"], 0)

    def test_half_stop_exits_at_half_the_loss(self):
        path = list(np.linspace(1.1000, 1.0980, 40))  # 20 pip düşüş; stop 15 pip, yarısı 7,5 pip
        trades = bt.run(bt.VARIANTS[0], market(path), {}, STATS)["trades"]
        first = trades[0]
        self.assertEqual(first["reason"], "YARIM STOP")
        self.assertAlmostEqual(first["gross_pips"], -7.5, delta=0.6)  # mumlar arası boşlukta açılıştan çıkar
        full = bt.run(bt.VARIANTS[2], market(path), {}, STATS)["trades"][0]  # yarım stop yok
        self.assertEqual(full["reason"], "STOP")
        self.assertAlmostEqual(full["gross_pips"], -15.0, delta=0.6)

    def test_basket_trail_closes_after_peak_turns(self):
        # 0,08 lot ≈ 0,8 $/pip; arm %2 = 10 $ ≈ 12,5 pip; fiyat 20 pip çıkıp geri iner
        trail = bt.Variant("t", basket_pct=0.02, basket_trail=True, **bt._STOP_ONLY)
        path = list(np.linspace(1.1000, 1.1020, 21)) + list(np.linspace(1.1019, 1.0980, 40))
        first = bt.run(trail, market(path), {}, STATS, fixed_risk=True)["trades"][0]
        self.assertEqual(first["reason"], "SEPET TAKİP")
        self.assertGreater(first["gross_pips"], 12.5)
        # tepe arm'a gelmezse sepet karışmaz, tam stop çalışır
        path = list(np.linspace(1.1000, 1.1005, 6)) + list(np.linspace(1.1004, 1.0970, 40))
        first = bt.run(trail, market(path), {}, STATS, fixed_risk=True)["trades"][0]
        self.assertEqual(first["reason"], "STOP")

    def test_support_needs_15m_score_on_entry_side(self):
        path = list(np.linspace(1.1000, 1.1020, 30))
        supported = bt.Variant("s", need_15m=True)
        for score, expected in ((50.0, True), (-50.0, False), (10.0, False)):
            m = market(path, prob=0.6)
            m["S15"] = np.full_like(m["Open"], score)
            self.assertEqual(bool(bt.run(supported, m, {}, STATS)["trades"]), expected)

    def test_news_blocks_entry(self):
        m = market([1.1000] * 24)
        event = {"USD": np.array([int(m["ns"][0] + 30 * 60 * 10**9)])}  # 30 dk sonra USD haberi
        trades = bt.run(bt.VARIANTS[0], m, event, STATS)["trades"]
        opened = [t for t in trades if t["opened"] < m["ns"][0] + 45 * 60 * 10**9]
        self.assertEqual(opened, [])  # haber 15 dk geçene kadar giriş yok

    def test_ml_flip_closes_at_the_new_hour(self):
        m = market([1.1000] * 30, prob=0.6)
        m["Prob"][12:, J] = 0.4  # 11:00'da ML SHORT der
        trades = bt.run(bt.VARIANTS[0], m, {}, STATS)["trades"]
        self.assertEqual(trades[0]["reason"], "ML YÖN DÖNDÜ")
        self.assertEqual(pd.Timestamp(trades[0]["closed"], tz="UTC"), pd.Timestamp("2023-01-02 11:00", tz="UTC"))
        self.assertNotIn("ML YÖN DÖNDÜ", [t["reason"] for t in bt.run(bt.VARIANTS[3], m, {}, STATS)["trades"]])

    def test_basket_closes_everything_at_the_total_target(self):
        path = list(np.linspace(1.1000, 1.1040, 40))  # 40 pip yükseliş; tek pozisyonun kârı 15 $'ı geçer
        stop_only = bt.BASKET_VARIANTS[1]
        first = bt.run(stop_only, market(path), {}, STATS, fixed_risk=True)["trades"][0]
        self.assertEqual(first["reason"], "TOPLAM HEDEF")
        self.assertGreaterEqual(first["net_usd"], 15.0)
        self.assertLess(first["net_usd"], 17.0)  # açılışlarda kontrol: hedefi az geçer
        # hedef olmayan yolda tam stop
        down = bt.run(stop_only, market(list(np.linspace(1.1000, 1.0970, 40))), {}, STATS, fixed_risk=True)["trades"][0]
        self.assertEqual(down["reason"], "STOP")


if __name__ == "__main__":
    unittest.main()
