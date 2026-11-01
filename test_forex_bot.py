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
            self.assertEqual(sent[0]["sl"], 1.09955)  # yarım stop; bot kapalıyken de broker uygular
            self.assertEqual(len(sent), 1)  # kayma yoksa ikinci istek gönderilmez
            row = conn.execute("SELECT exit_mode,stop,early_exit FROM trades WHERE ticket=123").fetchone()
            self.assertEqual(row[0], mode)
            self.assertAlmostEqual(row[1], row[2])
            conn.close()

    def open_with_fill(self, side, fill, modify_ok=True):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(f"CREATE TABLE trades ({forex_bot.TRADE_COLUMNS})")
        sent = []

        def order_send(request):
            sent.append(request)
            retcode = 10009 if request["action"] == 1 or modify_ok else 10016
            return SimpleNamespace(retcode=retcode, order=123, price=fill, comment="")

        mt5 = SimpleNamespace(TRADE_ACTION_DEAL=1, TRADE_ACTION_SLTP=6, ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1,
            ORDER_TIME_GTC=0, TRADE_RETCODE_DONE=10009, ORDER_FILLING_FOK=0, order_send=order_send,
            symbol_info_tick=lambda _: SimpleNamespace(bid=1.09999, ask=1.10001))
        sign = 1 if side == "LONG" else -1
        item = {"name": "EURUSD", "info": SimpleNamespace(digits=5, volume_step=.01, volume_min=.01, volume_max=10,
            filling_mode=1), "balance": 400., "margin": 30., "pip": .0001, "atr": .0006, "cost_pips": .7,
            "spread_pips": .2, "probability_up": .56, "target_usd": 1.2, "loss_usd": 4.,
            "check": {"hit": .67}, "check_half": {"hit": .53},
            "plan": {"side": side, "lot": .04, "entry": 1.1, "stop": 1.1 - sign * .0009,
                     "target": 1.1 + sign * .0004, "stop_pips": 9.}}
        with patch.object(forex_bot, "log"):
            forex_bot.open_position(mt5, conn, item, SimpleNamespace(exit_mode="fixed"),
                                    forex_bot.pd.Timestamp("2026-10-27 13:00", tz="UTC"))
        row = dict(conn.execute("SELECT * FROM trades WHERE ticket=123").fetchone())
        conn.close()
        return sent, row

    def test_levels_follow_the_real_fill_for_both_sides(self):
        for side, sign, fill in (("LONG", 1, 1.10011), ("SHORT", -1, 1.09989)):
            sent, row = self.open_with_fill(side, fill)  # 1 pip aleyhe kayma (spread hariç)
            self.assertEqual(len(sent), 2)
            self.assertEqual(sent[1]["action"], 6)
            self.assertEqual(sent[1]["position"], 123)
            self.assertAlmostEqual(sent[1]["sl"], 1.1 + sign * (.0001 - .00045))
            self.assertAlmostEqual(sent[1]["tp"], 1.1 + sign * (.0001 + .0004))
            self.assertAlmostEqual(row["entry"], 1.1 + sign * .0001)
            self.assertAlmostEqual(row["target"], sent[1]["tp"])
            self.assertAlmostEqual(row["early_exit"], sent[1]["sl"])
            self.assertAlmostEqual(row["stop"], sent[1]["sl"])
            self.assertEqual(row["fill"], fill)

    def test_failed_level_move_keeps_journal_equal_to_the_broker(self):
        sent, row = self.open_with_fill("LONG", 1.10011, modify_ok=False)
        self.assertEqual(len(sent), 2)
        self.assertEqual((row["entry"], row["target"]), (1.1, 1.1004))
        self.assertEqual(row["stop"], sent[0]["sl"])

    def test_server_stop_is_named_after_its_level(self):
        row = {"trailing": 0, "stop": 1.09955, "early_exit": 1.0995512, "pip": .0001}
        self.assertEqual(forex_bot.server_stop_outcome(row), "YARIM STOP")
        self.assertEqual(forex_bot.server_stop_outcome({**row, "stop": 1.0991}), "STOP")
        self.assertEqual(forex_bot.server_stop_outcome({**row, "trailing": 1}), "TAKİP STOPU")
        self.assertEqual(forex_bot.server_stop_outcome({**row, "early_exit": None, "pip": None}), "STOP")

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

    def test_rolling_hourly_bars_end_at_anchor(self):
        import pandas as pd
        index = pd.date_range("2026-10-27 10:00", "2026-10-27 12:40", freq="5min", tz="UTC")
        bars = pd.DataFrame({"Open": range(len(index)), "High": range(len(index)), "Low": range(len(index)),
                             "Close": range(len(index))}, index=index, dtype=float)
        anchor = pd.Timestamp("2026-10-27 12:45", tz="UTC")
        hourly = forex_bot.rolling_hourly(bars, anchor)
        self.assertEqual(hourly.index[-1], pd.Timestamp("2026-10-27 11:45", tz="UTC"))
        last = bars.loc["2026-10-27 11:45":"2026-10-27 12:40"]
        self.assertEqual(hourly.iloc[-1].tolist(), [last.Open.iloc[0], last.High.max(), last.Low.min(),
                                                     last.Close.iloc[-1]])
        # henüz kapanmamış 5M mum (12:45 açılışı) alınmaz
        later = forex_bot.rolling_hourly(pd.concat([bars, bars.iloc[[-1]].set_axis(
            [pd.Timestamp("2026-10-27 12:45", tz="UTC")])]), anchor)
        self.assertEqual(later.iloc[-1].tolist(), hourly.iloc[-1].tolist())

    def five_minute_bars(self, start, end):
        pd = forex_bot.pd
        index = pd.date_range(start, end, freq="5min", tz="UTC")
        return pd.DataFrame(1., index=index, columns=["Open", "High", "Low", "Close"])

    def test_rolling_hourly_keeps_broker_gaps_and_drops_only_the_cut_first_hour(self):
        pd = forex_bot.pd
        anchor = pd.Timestamp("2026-10-27 12:45", tz="UTC")
        bars = self.five_minute_bars("2026-10-27 10:00", "2026-10-27 12:40")
        expected = [pd.Timestamp("2026-10-27 10:45", tz="UTC"), pd.Timestamp("2026-10-27 11:45", tz="UTC")]
        # 09:45–10:45 yüklemenin başında kesilmiş; sonraki saatler tam.
        self.assertEqual(forex_bot.rolling_hourly(bars, anchor).index.tolist(), expected)
        # Tam saat sınırından başlayan geçmişin ilk saati kesik değildir.
        whole = bars[bars.index >= expected[0]]
        self.assertEqual(forex_bot.rolling_hourly(whole, anchor).index.tolist(), expected)
        # Geçmişteki olağan boşluk saati silmez; broker H1 de bu saati tutar.
        gap = bars.drop(pd.Timestamp("2026-10-27 12:00", tz="UTC"))
        self.assertEqual(forex_bot.rolling_hourly(gap, anchor).index.tolist(), expected)
        # Yinelenen ve geçersiz mumlar saati değiştirmez.
        duplicate = pd.concat([gap, gap.loc[[pd.Timestamp("2026-10-27 12:05", tz="UTC")]]])
        self.assertEqual(forex_bot.rolling_hourly(duplicate, anchor).index.tolist(), expected)
        null = bars.copy()
        null.loc[pd.Timestamp("2026-10-27 12:00", tz="UTC"), "High"] = float("nan")
        self.assertEqual(forex_bot.rolling_hourly(null, anchor).index.tolist(), expected)
        self.assertTrue(forex_bot.rolling_hourly(bars.iloc[:0], anchor).empty)

    def test_last_hour_needs_ten_distinct_valid_closed_bars(self):
        pd = forex_bot.pd
        anchor = pd.Timestamp("2026-10-27 12:45", tz="UTC")
        hour = self.five_minute_bars("2026-10-27 09:45", "2026-10-27 12:40")
        history, last = hour[hour.index < anchor - pd.Timedelta(hours=1)], hour[hour.index >= anchor - pd.Timedelta(hours=1)]
        with_last = lambda keep: pd.concat([history, last.iloc[:keep]])
        self.assertTrue(forex_bot.last_hour_complete(hour, anchor))
        self.assertTrue(forex_bot.last_hour_complete(with_last(11), anchor))
        self.assertTrue(forex_bot.last_hour_complete(with_last(10), anchor))
        self.assertFalse(forex_bot.last_hour_complete(with_last(9), anchor))
        # Yinelenen mum sayıyı şişirmez, geçersiz mum sayılmaz.
        self.assertFalse(forex_bot.last_hour_complete(pd.concat([with_last(9), last.iloc[[0]]]), anchor))
        null = with_last(10).copy()
        null.iloc[-1, 1] = float("nan")
        self.assertFalse(forex_bot.last_hour_complete(null, anchor))
        # Eski son saat ve kapanmamış mum kabul edilmez.
        self.assertFalse(forex_bot.last_hour_complete(history, anchor))
        unclosed = pd.concat([with_last(9), last.iloc[[-1]].set_axis([anchor])])
        self.assertFalse(forex_bot.last_hour_complete(unclosed, anchor))

    def test_rolling_ml_retries_missing_latest_hour_without_caching_failure(self):
        pd = forex_bot.pd
        anchor = pd.Timestamp("2026-10-27 12:45", tz="UTC")
        full = self.five_minute_bars("2026-10-27 10:45", "2026-10-27 12:40")
        incomplete = full.iloc[:-3]
        mt5 = SimpleNamespace(TIMEFRAME_M5=5)
        with patch.dict(forex_bot._ML_CACHE, {}, clear=True), \
             patch.object(forex_bot, "rates_utc", side_effect=[incomplete, full]) as rates, \
             patch.object(forex_bot, "build_research_prediction", return_value={
                 "status": "ready", "probability_up": .56}) as predict:
            self.assertIsNone(forex_bot.ml_probability(mt5, "EURUSD", anchor, 5))
            predict.assert_not_called()
            self.assertEqual(forex_bot.ml_probability(mt5, "EURUSD", anchor, 5), .56)
            self.assertEqual(forex_bot.ml_probability(mt5, "EURUSD", anchor, 5), .56)
            self.assertEqual(rates.call_count, 2)
            self.assertEqual(predict.call_count, 1)

    def test_rejected_close_does_not_assign_a_successful_exit_reason(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE trades (ticket INTEGER, close_reason TEXT)")
        conn.execute("INSERT INTO trades VALUES (123, NULL)")
        mt5 = SimpleNamespace(TRADE_ACTION_DEAL=1, POSITION_TYPE_BUY=0,
            ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1, ORDER_TIME_GTC=0, TRADE_RETCODE_DONE=10009,
            symbol_info_tick=lambda _: SimpleNamespace(bid=1.1, ask=1.1001),
            symbol_info=lambda _: SimpleNamespace(),
            order_send=lambda _: SimpleNamespace(retcode=10018, comment="Market closed"))
        position = SimpleNamespace(ticket=123, symbol="EURUSD", type=0, volume=.1)
        with patch.object(forex_bot, "filling", return_value=0), patch.object(forex_bot, "log"):
            self.assertFalse(forex_bot.close_position(mt5, conn, position, "ML YÖN DÖNDÜ"))
            self.assertIsNone(conn.execute("SELECT close_reason FROM trades").fetchone()[0])
            mt5.order_send = lambda _: SimpleNamespace(retcode=10009)
            self.assertTrue(forex_bot.close_position(mt5, conn, position, "ML YÖN DÖNDÜ"))
            self.assertEqual(conn.execute("SELECT close_reason FROM trades").fetchone()[0], "ML YÖN DÖNDÜ")

    def test_report_counts_fixed_tp_as_reaching_target(self):
        from contextlib import redirect_stdout
        from io import StringIO
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute(f"CREATE TABLE trades ({forex_bot.TRADE_COLUMNS})")
        conn.execute("CREATE TABLE decisions (action TEXT)")
        conn.executemany("""INSERT INTO trades
            (ticket,sizing_balance,closed_at,profit_usd,outcome,trailing,hit,hit_half_stop)
            VALUES (?,500,'2026-10-27',1,?,?,.67,.55)""",
            [(1, "HEDEF", 0), (2, "TAKİP STOPU", 1), (3, "STOP", 0)])
        output = StringIO()
        with patch.object(forex_bot, "db", return_value=conn), redirect_stdout(output):
            forex_bot.report()
        self.assertIn("Hedefe (sabit TP veya takip başlangıcına) gelen: %67", output.getvalue())

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
