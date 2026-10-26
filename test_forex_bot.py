import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import forex_bot

FAKE_MT5 = SimpleNamespace(ACCOUNT_TRADE_MODE_DEMO=0, ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1,
                           ORDER_FILLING_RETURN=2)


class ForexBotTests(unittest.TestCase):
    def test_volume_is_floored_to_broker_step_and_limits(self):
        info = SimpleNamespace(volume_step=0.01, volume_min=0.01, volume_max=50.0)
        self.assertEqual(forex_bot.volume_for(info, 1.079), 1.07)
        self.assertEqual(forex_bot.volume_for(info, 0.001), 0.01)
        self.assertEqual(forex_bot.volume_for(info, 80), 50.0)
        self.assertEqual(forex_bot.volume_for(SimpleNamespace(volume_step=0.1, volume_min=0.1, volume_max=10), 1.29), 1.2)

    def test_filling_follows_symbol_flags(self):
        self.assertEqual(forex_bot.filling(FAKE_MT5, SimpleNamespace(filling_mode=1)), 0)
        self.assertEqual(forex_bot.filling(FAKE_MT5, SimpleNamespace(filling_mode=2)), 1)
        self.assertEqual(forex_bot.filling(FAKE_MT5, SimpleNamespace(filling_mode=0)), 2)

    def test_refuses_real_account(self):
        calls = []
        mt5 = SimpleNamespace(ACCOUNT_TRADE_MODE_DEMO=0, initialize=lambda: True,
                              account_info=lambda: SimpleNamespace(trade_mode=2),
                              shutdown=lambda: calls.append("shutdown"), last_error=lambda: None)
        with patch.dict(sys.modules, {"MetaTrader5": mt5}):
            with self.assertRaises(SystemExit) as stop:
                forex_bot.connect()
        self.assertIn("DEMO", str(stop.exception))
        self.assertEqual(calls, ["shutdown"])

    def test_stop_waits_until_target_then_locks_cost_and_trails(self):
        common = dict(entry=1.1000, target=1.1004, atr=0.0006, cost_pips=0.7, pip=0.0001)
        # hedefe gelmedi: dokunma
        self.assertEqual(forex_bot.next_stop("LONG", current_sl=1.0991, best=1.1003, price=1.1002,
                                             trailing=False, **common), (None, False))
        # hedefe geldi: stop giriş + maliyet (1 ATR gerisi daha aşağıda kalıyor)
        sl, trailing = forex_bot.next_stop("LONG", current_sl=1.0991, best=1.1004, price=1.1004,
                                           trailing=False, **common)
        self.assertTrue(trailing)
        self.assertAlmostEqual(sl, 1.10007)
        # fiyat ilerledi: en iyi fiyatın 1 ATR gerisi
        sl, _ = forex_bot.next_stop("LONG", current_sl=1.10007, best=1.1020, price=1.1018, trailing=True, **common)
        self.assertAlmostEqual(sl, 1.1014)
        # geri çekilmede stop geri gelmez
        self.assertEqual(forex_bot.next_stop("LONG", current_sl=1.1014, best=1.1020, price=1.1015,
                                             trailing=True, **common), (None, True))

    def test_short_trailing_is_mirrored_and_never_crosses_price(self):
        common = dict(entry=1.1000, target=1.0996, atr=0.0006, cost_pips=0.7, pip=0.0001)
        sl, trailing = forex_bot.next_stop("SHORT", current_sl=1.1009, best=1.0996, price=1.0996,
                                           trailing=False, **common)
        self.assertTrue(trailing)
        self.assertAlmostEqual(sl, 1.09993)
        sl, _ = forex_bot.next_stop("SHORT", current_sl=1.09993, best=1.0980, price=1.0982, trailing=True, **common)
        self.assertAlmostEqual(sl, 1.0986)
        # fiyat stopun öbür tarafına geçtiyse sunucuya geçersiz stop gönderme
        self.assertEqual(forex_bot.next_stop("SHORT", current_sl=1.1009, best=1.0996, price=1.1001,
                                             trailing=False, **common), (None, True))

    def test_legacy_position_gets_target_from_tp_and_half_stop(self):
        fields = forex_bot.legacy_fields("SHORT", entry=0.98449, sl=0.98520, tp=0.98419, pip=0.0001)
        self.assertEqual(fields["target"], 0.98419)
        self.assertAlmostEqual(fields["early_exit"], 0.984845)
        self.assertAlmostEqual(fields["atr"], 0.00071 / 1.5)
        self.assertEqual(fields["best"], 0.98449)

    def test_second_bot_is_refused(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE heartbeat (id INTEGER PRIMARY KEY CHECK (id = 1), pid INTEGER, at TEXT)")
        with patch.object(forex_bot, "other_bot_processes", return_value=[]):
            forex_bot.claim_heartbeat(conn, 111)
            with self.assertRaises(SystemExit):
                forex_bot.claim_heartbeat(conn, 222)
            forex_bot.claim_heartbeat(conn, 111)  # aynı bot yeniden yazabilir
        with patch.object(forex_bot, "other_bot_processes", return_value=[2984]):
            with self.assertRaises(SystemExit) as stop:
                forex_bot.claim_heartbeat(conn, 111)
        self.assertIn("2984", str(stop.exception))

    def test_only_stale_breakeven_positions_are_swapped(self):
        import pandas as pd
        now = pd.Timestamp("2026-10-26 20:00", tz="UTC")
        old, fresh = now - pd.Timedelta(hours=5), now - pd.Timedelta(hours=1)
        self.assertTrue(forex_bot.swappable(False, old, now, 0.0, 4))      # başabaş, 5 saat: takas
        self.assertTrue(forex_bot.swappable(False, old, now, 0.8, 4))
        self.assertFalse(forex_bot.swappable(False, old, now, -0.1, 4))    # zararda: takas yok
        self.assertFalse(forex_bot.swappable(False, fresh, now, 0.8, 4))   # henüz 1 saat
        self.assertFalse(forex_bot.swappable(True, old, now, 3.0, 4))      # hedefe gelmiş, takipte: dokunma

    def test_decision_uses_karar_rule(self):
        stats = {"pairs": {"EURUSD": {"ratios": {
            "0.5": {"ml": {"target_rate": 0.67}, "opposite": {"target_rate": 0.63}},
            "1.0": {"ml": {"target_rate": 0.51}, "opposite": {"target_rate": 0.49}},
        }}}}
        item = {"name": "EURUSD", "price": 1.1, "pip": 0.0001, "atr": 0.0005, "probability_up": 0.45,
                "pip_value": 10.0, "cost_pips": 0.7}
        out = forex_bot.decide(dict(item), 30.0, 100.0, 100000.0, stats)
        self.assertEqual(out["plan"]["side"], "SHORT")
        self.assertTrue(out["check"]["ok"])
        self.assertFalse(forex_bot.decide(dict(item), 200.0, 100.0, 100000.0, stats)["check"]["ok"])
        # 500 $ hesap, %1 risk: aynı oran, küçük lot
        small = forex_bot.decide(dict(item), 1.5, 5.0, 500.0, stats)
        self.assertTrue(small["check"]["ok"])
        self.assertEqual(small["plan"]["lot"], 0.06)

    def test_budget_follows_virtual_balance(self):
        args = SimpleNamespace(loss_usd=None, target_usd=None, risk_pct=1.0, target_r=0.3)
        self.assertEqual(forex_bot.budget(500.0, args), (1.5, 5.0))
        self.assertEqual(forex_bot.budget(420.0, args), (1.26, 4.2))
        fixed = SimpleNamespace(loss_usd=10.0, target_usd=4.0, risk_pct=1.0, target_r=0.3)
        self.assertEqual(forex_bot.budget(500.0, fixed), (4.0, 10.0))

    def test_virtual_balance_counts_only_its_own_closed_trades(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE trades (sizing_balance REAL, closed_at TEXT, profit_usd REAL)")
        conn.executemany("INSERT INTO trades VALUES (?,?,?)", [
            (None, "x", 40.0),      # ilk sürüm, 98 bin $ ile boyutlanmış: sayılmaz
            (500.0, "x", -5.0), (500.0, "x", 1.5), (500.0, None, 99.0)])  # sonuncusu açık
        self.assertAlmostEqual(forex_bot.virtual_balance(conn, 500.0), 496.5)


if __name__ == "__main__":
    unittest.main()
