import sys
import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import forex_bot

FAKE_MT5 = SimpleNamespace(ACCOUNT_TRADE_MODE_DEMO=0, ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1,
                           ORDER_FILLING_RETURN=2)


class ForexBotTests(unittest.TestCase):
    def test_new_entry_defaults_and_confidence_argument_validation(self):
        args = forex_bot.parse_args([])
        self.assertEqual((args.risk_pct, args.min_ml_confidence, args.exit_mode), (1.0, .55, "fixed"))
        self.assertEqual(args.max_cost_share, .10)
        for value in ("0.49", "1.01", "nan"):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                forex_bot.parse_args(["--min-ml-confidence", value])
        for value in ("0", "1", "nan", "inf", "-0.1"):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                forex_bot.parse_args(["--max-cost-share", value])

    def test_three_observed_weak_signals_are_blocked_in_experiment_mode(self):
        stats = {"pooled": {"ratios": {
            "0.5": {"ml": {"target_rate": .68, "net_r": -.1}, "opposite": {"target_rate": .68}},
            "1.0": {"ml": {"target_rate": .51, "net_r": -.1}, "opposite": {"target_rate": .51}},
        }}}
        for probability in (.5003010804, .5235804124, .5040830117, .5499, .4501, .55, .45):
            item = {"name": "EURUSD", "price": 1.1, "pip": .0001, "atr": .0005,
                    "probability_up": probability, "pip_value": 10., "cost_pips": .7}
            result = forex_bot.decide(item, 30., 100., 1000., stats, experiment=True)
            self.assertEqual(result["check"]["ok"], probability in (.55, .45))
            if not result["check"]["ok"]:
                self.assertIn("Ham ML yön güveni", " ".join(result["check"]["reasons"]))

    def test_old_database_migration_keeps_existing_trades_on_trailing(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "journal.db"
            conn = sqlite3.connect(path)
            old_columns = forex_bot.TRADE_COLUMNS.rsplit(", exit_mode", 1)[0]
            conn.execute(f"CREATE TABLE trades ({old_columns})")
            conn.execute("INSERT INTO trades (ticket,symbol,lot) VALUES (123,'EURUSD',.14)")
            conn.commit()
            conn.close()
            with patch.object(forex_bot, "DB_PATH", path):
                conn = forex_bot.db()
                row = conn.execute("SELECT * FROM trades WHERE ticket=123").fetchone()
                self.assertEqual((row["exit_mode"], row["lot"]), ("trailing", .14))
                conn.close()

    def test_new_order_has_server_target_and_persists_its_exit_policy(self):
        for mode, expected_tp in (("fixed", 1.1004), ("trailing", 0.0)):
            conn = sqlite3.connect(":memory:")
            conn.execute(f"CREATE TABLE trades ({forex_bot.TRADE_COLUMNS})")
            sent = []
            mt5 = SimpleNamespace(TRADE_ACTION_DEAL=1, ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1,
                ORDER_TIME_GTC=0, TRADE_RETCODE_DONE=10009, ORDER_FILLING_FOK=0,
                symbol_info_tick=lambda _: SimpleNamespace(bid=1.09999, ask=1.10001),
                order_send=lambda request: sent.append(request) or SimpleNamespace(retcode=10009, order=123, price=1.10001))
            item = {"name": "EURUSD", "info": SimpleNamespace(digits=5, volume_step=.01,
                volume_min=.01, volume_max=10, filling_mode=1), "balance": 400., "margin": 30.,
                "pip": .0001, "atr": .0006, "cost_pips": .7, "spread_pips": .2,
                "probability_up": .56, "target_usd": 1.2, "loss_usd": 4.,
                "check": {"hit": .67}, "check_half": {"hit": .53},
                "plan": {"side": "LONG", "lot": .04, "entry": 1.1, "stop": 1.0991,
                         "target": 1.1004, "stop_pips": 9.}}
            with patch.object(forex_bot, "log"):
                ticket = forex_bot.open_position(mt5, conn, item, SimpleNamespace(exit_mode=mode),
                    forex_bot.pd.Timestamp("2026-10-27 13:00", tz="UTC"))
            self.assertEqual(ticket, 123)
            self.assertEqual(sent[0]["tp"], expected_tp)
            self.assertEqual(conn.execute("SELECT exit_mode FROM trades WHERE ticket=123").fetchone()[0], mode)
            conn.close()

    def test_fixed_exit_takes_target_and_keeps_half_stop_for_both_sides(self):
        for side, sign in (("LONG", 1), ("SHORT", -1)):
            for move, reason in ((.0006, "HEDEF"), (-.0006, "YARIM STOP"), (.0001, None)):
                conn = sqlite3.connect(":memory:")
                conn.row_factory = sqlite3.Row
                conn.execute(f"CREATE TABLE trades ({forex_bot.TRADE_COLUMNS})")
                conn.execute("""INSERT INTO trades (ticket,side,entry,target,early_exit,atr,pip,best,trailing,exit_mode)
                    VALUES (123,?,?,?,?,.0006,.0001,1.1,0,'fixed')""",
                    (side, 1.1, 1.1 + sign * .0005, 1.1 - sign * .0005))
                position = SimpleNamespace(ticket=123, symbol="EURUSD", sl=1.1-sign*.001,
                    time=int(forex_bot.pd.Timestamp.now(tz="UTC").timestamp()))
                price = 1.1 + sign * move
                mt5 = SimpleNamespace(symbol_info_tick=lambda _: SimpleNamespace(bid=price, ask=price),
                    symbol_info=lambda _: SimpleNamespace(digits=5))
                with patch.object(forex_bot, "own_positions", return_value=[position]), \
                     patch.object(forex_bot, "server_offset", return_value=forex_bot.pd.Timedelta(0)), \
                     patch.object(forex_bot, "sync_closed"), patch.object(forex_bot, "next_stop") as trailing, \
                     patch.object(forex_bot, "close_position", return_value=True) as close:
                    forex_bot.manage(mt5, conn, check_ml=False)
                    trailing.assert_not_called()
                    if reason:
                        self.assertEqual(close.call_args.args[-1], reason)
                    else:
                        close.assert_not_called()
                conn.close()

    def test_broker_take_profit_is_recorded_as_target_exit_with_fees(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(f"CREATE TABLE trades ({forex_bot.TRADE_COLUMNS})")
        conn.execute("INSERT INTO trades (ticket) VALUES (123)")
        deal = lambda entry, profit, commission: SimpleNamespace(entry=entry, profit=profit, commission=commission,
            swap=0, fee=0, reason=5, time=1793100000, symbol="EURUSD", price=1.1005)
        mt5 = SimpleNamespace(DEAL_ENTRY_OUT=1, DEAL_REASON_TP=5, DEAL_REASON_SL=4, DEAL_REASON_EXPERT=3,
            history_deals_get=lambda **_: [deal(0, 0, -.14), deal(1, 1.5, -.14)])
        with patch.object(forex_bot, "own_positions", return_value=[]), \
             patch.object(forex_bot, "server_offset", return_value=forex_bot.pd.Timedelta(0)), \
             patch.object(forex_bot, "log"):
            forex_bot.sync_closed(mt5, conn)
        row = conn.execute("SELECT outcome,profit_usd FROM trades WHERE ticket=123").fetchone()
        self.assertEqual(row["outcome"], "HEDEF")
        self.assertAlmostEqual(row["profit_usd"], 1.22)
        conn.close()

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

    def test_stop_waits_until_target_then_trails_one_atr_behind(self):
        common = dict(entry=1.1000, target=1.1004, atr=0.0006, pip=0.0001)
        # hedefe gelmedi: dokunma
        self.assertEqual(forex_bot.next_stop("LONG", current_sl=1.0991, best=1.1003, price=1.1002,
                                             trailing=False, **common), (None, False))
        # hedefe geldi: en iyi fiyatın 1 ATR gerisi (girişin 0,5 ATR altındaki tabanın üstünde)
        sl, trailing = forex_bot.next_stop("LONG", current_sl=1.0991, best=1.1004, price=1.1004,
                                           trailing=False, **common)
        self.assertTrue(trailing)
        self.assertAlmostEqual(sl, 1.0998)
        # taban: 1 ATR gerisi girişin 0,5 ATR altından aşağıdaysa taban kullanılır
        sl, _ = forex_bot.next_stop("LONG", current_sl=1.0991, best=1.1001, price=1.1001, trailing=True, **common)
        self.assertAlmostEqual(sl, 1.0997)
        # fiyat ilerledi: en iyi fiyatın 1 ATR gerisi
        sl, _ = forex_bot.next_stop("LONG", current_sl=1.0998, best=1.1020, price=1.1018, trailing=True, **common)
        self.assertAlmostEqual(sl, 1.1014)
        # geri çekilmede stop geri gelmez
        self.assertEqual(forex_bot.next_stop("LONG", current_sl=1.1014, best=1.1020, price=1.1015,
                                             trailing=True, **common), (None, True))

    def test_short_trailing_is_mirrored_and_never_crosses_price(self):
        common = dict(entry=1.1000, target=1.0996, atr=0.0006, pip=0.0001)
        sl, trailing = forex_bot.next_stop("SHORT", current_sl=1.1009, best=1.0996, price=1.0996,
                                           trailing=False, **common)
        self.assertTrue(trailing)
        self.assertAlmostEqual(sl, 1.1002)
        sl, _ = forex_bot.next_stop("SHORT", current_sl=1.1002, best=1.0980, price=1.0982, trailing=True, **common)
        self.assertAlmostEqual(sl, 1.0986)
        # fiyat stopun öbür tarafına geçtiyse sunucuya geçersiz stop gönderme
        self.assertEqual(forex_bot.next_stop("SHORT", current_sl=1.1009, best=1.0996, price=1.1003,
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

    def test_same_direction_currency_is_limited_opposite_is_allowed(self):
        held = [("AUDUSD", "SHORT")]
        self.assertIn("AUD", forex_bot.exposure_block("AUDCAD", "SHORT", held, 1))   # ikinci AUD satımı
        self.assertIsNone(forex_bot.exposure_block("AUDCAD", "LONG", held, 1))       # AUD alımı dengeler
        # AUDUSD SHORT = USD alımı: EURUSD SHORT ikinci USD alımı olur, EURUSD LONG (USD satımı) dengeler
        self.assertIn("USD", forex_bot.exposure_block("EURUSD", "SHORT", held, 1))
        self.assertIsNone(forex_bot.exposure_block("EURUSD", "LONG", held, 1))
        self.assertIsNone(forex_bot.exposure_block("AUDCAD", "SHORT", held, 2))      # sınır 2 ise olur
        self.assertEqual(forex_bot.currency_exposure([("EURUSD", "LONG"), ("USDJPY", "LONG")]),
                         {"EUR": 1, "USD": 0, "JPY": -1})

    def test_expensive_moments_block_entry(self):
        import pandas as pd
        monday = pd.Timestamp("2026-10-26 12:00", tz="UTC")  # Londra-New York, sakin saat
        item = {"name": "EURUSD", "spread_pips": 0.1, "median_spread_pips": 0.0, "cost_pips": 0.7,
                "plan": {"stop_pips": 10.0}}
        events = pd.DataFrame({"time": [monday + pd.Timedelta(minutes=30)], "currency": ["USD"],
                               "event": ["CPI m/m"], "impact": [3]})
        no_news = events.iloc[0:0]
        self.assertEqual(forex_bot.entry_blockers(item, monday, no_news), [])
        self.assertIn("spread", forex_bot.entry_blockers({**item, "spread_pips": 1.2}, monday, no_news)[0])
        self.assertIn("CPI", " ".join(forex_bot.entry_blockers(item, monday, events)))
        self.assertIn("takvimi", forex_bot.entry_blockers(item, monday, None)[0])
        rollover = pd.Timestamp("2026-10-26 21:10", tz="UTC")  # New York 17:10
        self.assertIn("Gün sonu", " ".join(forex_bot.entry_blockers(item, rollover, no_news)))
        pricey = {**item, "cost_pips": 2.0, "plan": {"stop_pips": 5.0}}  # maliyet riskin %29'u
        self.assertIn("Maliyet", " ".join(forex_bot.entry_blockers(pricey, monday, no_news)))

    def test_entry_cost_cap_accepts_ten_percent_and_blocks_just_above(self):
        now = forex_bot.pd.Timestamp("2026-10-27 12:00", tz="UTC")
        events = forex_bot.pd.DataFrame(columns=["time", "currency", "event"])
        events["time"] = forex_bot.pd.to_datetime(events["time"], utc=True)
        item = {"name": "EURUSD", "spread_pips": .1, "median_spread_pips": .1,
                "cost_pips": 1., "plan": {"stop_pips": 9.}}
        self.assertEqual(forex_bot.entry_blockers(item, now, events), [])
        item["plan"]["stop_pips"] = 8.99
        reasons = forex_bot.entry_blockers(item, now, events)
        self.assertIn("sınır %10", " ".join(reasons))
        self.assertEqual(forex_bot.entry_blockers(item, now, events, max_cost_share=.15), [])

    def test_decision_uses_karar_rule(self):
        stats = {"pairs": {"EURUSD": {"ratios": {
            "0.5": {"ml": {"target_rate": 0.67, "net_r": -0.1}, "opposite": {"target_rate": 0.63, "net_r": -0.12}},
            "1.0": {"ml": {"target_rate": 0.51, "net_r": -0.1}, "opposite": {"target_rate": 0.49, "net_r": -0.12}},
        }}}}
        item = {"name": "EURUSD", "price": 1.1, "pip": 0.0001, "atr": 0.0005, "probability_up": 0.45,
                "pip_value": 10.0, "cost_pips": 0.7}
        out = forex_bot.decide(dict(item), 30.0, 100.0, 100000.0, stats)
        self.assertEqual(out["plan"]["side"], "SHORT")
        # %67 ulaşma %65'i geçiyor ama 30/100'de başabaş ≈ %76 ve ölçülen sonuç eksi: Karar kutusu gibi BEKLE
        self.assertFalse(out["check"]["ok"])
        self.assertTrue(any("başabaş" in r for r in out["check"]["reasons"]))
        self.assertTrue(any("zarar etti" in r or "test edilmedi" in r for r in out["check"]["reasons"]))
        # --deney: ulaşma ≥ %65 ve maliyet şartı; başabaş/ölçüm engelleri atlanır
        self.assertTrue(forex_bot.decide(dict(item), 30.0, 100.0, 100000.0, stats, experiment=True)["check"]["ok"])
        self.assertFalse(forex_bot.decide(dict(item), 200.0, 100.0, 100000.0, stats, experiment=True)["check"]["ok"])
        # 500 $ hesap, %1 risk: aynı oran, küçük lot
        small = forex_bot.decide(dict(item), 1.5, 5.0, 500.0, stats, experiment=True)
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

    def feed_basket(self, watch, values, tickets=frozenset({1, 2, 3}), start_minute=0):
        """Her 30 sn'de bir sepet değeri verir; ilk çıkış sebebini (dakika, sebep) döner."""
        start = forex_bot.pd.Timestamp("2026-10-26 19:00", tz="UTC")
        for i, value in enumerate(values):
            minute = start_minute + i / 2
            why = watch.update(tickets, start + forex_bot.pd.Timedelta(minutes=minute), value, 15.0)
            if why:
                return minute, why
        return None

    def test_basket_waits_until_armed(self):
        self.assertIsNone(self.feed_basket(forex_bot.BasketWatch(), [14.0] * 60 + [9.0] * 20))

    def test_basket_exits_after_giving_back_part_of_the_peak(self):
        minute, why = self.feed_basket(forex_bot.BasketWatch(), [10, 14, 16, 15, 13, 11, 12])
        self.assertEqual(minute, 2.5)
        self.assertIn("geri verdi", why)

    def test_basket_exits_when_stalled_and_turning_down(self):
        rising = [15 + i * 0.1 for i in range(10)]  # tepe 15,9 $ (4,5. dk)
        drifting = [15.5 + (0.2 if i % 2 else 0) for i in range(30)]  # 15 dk yatay
        sliding = [15.5 - i * 0.05 for i in range(1, 20)]  # sonra yavaş düşüş
        minute, why = self.feed_basket(forex_bot.BasketWatch(), rising + drifting + sliding)
        self.assertGreater(minute, 4.5 + forex_bot.BASKET_STALL_MINUTES)
        self.assertIn("düşüyor", why)

    def test_basket_keeps_rising_basket_open(self):
        self.assertIsNone(self.feed_basket(forex_bot.BasketWatch(), [15 + i * 0.2 for i in range(120)]))

    def test_basket_restarts_when_positions_change(self):
        watch = forex_bot.BasketWatch()
        self.assertIsNone(self.feed_basket(watch, [16, 17]))
        self.assertIsNone(self.feed_basket(watch, [5, 6], tickets=frozenset({1, 2}), start_minute=1))


if __name__ == "__main__":
    unittest.main()
