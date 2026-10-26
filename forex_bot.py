"""Demo hesapta Karar kuralıyla al-sat yapan canlı MT5 botu: uygulamadaki kuralın ileri testi.

Lot (kullanıcı 500 $ ile başlayacak, 26 Ekim 2026): demo hesabın bakiyesi değil, --balance ile başlayan sanal
bakiye kullanılır; botun bu düzende kapanan işlemleri bakiyeye eklenir. Stop zararı sanal bakiyenin --risk-pct'i,
hedef bunun --target-r katı (Karar kutusundaki 30/100 oranı). Açık pozisyonların teminatı sanal bakiyenin
yarısını geçmez; --leverage gerçek hesabın kaldıracıyla teminatı yeniden ölçekler.

Giriş (Özet'teki Karar kutusuyla aynı hesap, forex_trade_plan + forex_decision):
- Yön: ML 4 saatlik yön modeli (son kapanmış saatlik mum; yön yalnız saat başında değişebilir).
- Stop: kapanmış 15M mumlarında ATR14 × 1,5; lot bu stop ve zarar bütçesinden. Hedef: dolar hedefinden.
- Şart: hedefin stoptan önce gelme oranı ≥ %60 (2023–2026 ölçümü) ve maliyet hedefin yarısından az.
- Her dakika tarar; parite başına tek pozisyon, en fazla --max-positions açık pozisyon (önce en yüksek oran);
  aynı paritede aynı saatlik sinyalle ikinci kez girmez.

Yönetim (kullanıcı seçimi, 26 Ekim 2026; geçmişte ölçülmedi, günlük hangi kuralın kapattığını yazar):
- Sabit TP yok. Fiyat hedefe gelince stop giriş + maliyete çekilir, sonra en iyi fiyatın 1 ATR gerisinden takip eder.
- Fiyat stop mesafesinin yarısına gelirse çıkar; saat başında ML yönü dönerse çıkar (takip başlamadıysa).
- Takip başlamamış pozisyon 24 saatte kapanır. Tam stop sunucuda güvenlik olarak durur.

Yalnız DEMO hesapta çalışır; gerçek hesaba bağlıysa emir göndermeden çıkar. Emir için MT5'te
"Algo Trading" açık olmalıdır. Kararlar ve işlemler data/bot_journal.db'ye yazılır.

  python forex_bot.py --dry-run --once   # emir göndermeden tek tarama
  python forex_bot.py                    # sürekli çalış (Ctrl+C ile dur)
  python forex_bot.py --report           # sonuçlar
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from forex_config import SYMBOL_LIST, get_pip_size
from forex_costs import ECN_COMMISSION_PIPS
from forex_decision import HOLD_HOURS, target_check
from forex_indicators import add_indicators
from forex_analysis import latest_valid_row
from forex_ml_live import build_research_prediction, has_research_model
from forex_mt5 import rates_utc, server_offset, server_time_to_utc
from forex_trade_plan import build_trade_plan, load_plan_stats

DB_PATH = Path(__file__).resolve().parent / "data" / "bot_journal.db"
MAGIC = 26102026
COMMENT = "karar-bot"
MANAGE_SECONDS = 5  # açık pozisyon kontrolü
SCAN_SECONDS = 60  # yeni fırsat taraması
ML_CHECK_DELAY_SECONDS = 60  # saatlik mum kapandıktan sonra ML yön kontrolü
MAX_TICK_AGE_SECONDS = 180  # daha eski tick: piyasa kapalı
DEVIATION_POINTS = 10
EARLY_EXIT_SHARE = 0.5  # stop mesafesinin bu kadarında çık
TRAIL_ATR = 1.0  # takip eden stop: en iyi fiyatın bu kadar ATR gerisi
MIN_SL_STEP_PIPS = 0.1
MAX_MARGIN_SHARE = 0.5  # açık pozisyon teminatı en fazla sanal bakiyenin bu kadarı
HEARTBEAT_SECONDS = 60


def log(text: str) -> None:
    print(f"{pd.Timestamp.now(tz='Europe/Istanbul'):%d.%m %H:%M:%S}  {text}", flush=True)


# ----------------------------------------------------------------------------- günlük

TRADE_COLUMNS = (
    "ticket INTEGER PRIMARY KEY, symbol TEXT, side TEXT, opened_at TEXT, signal_hour TEXT, lot REAL, entry REAL,"
    " fill REAL, stop REAL, target REAL, early_exit REAL, atr REAL, pip REAL, cost_pips REAL, spread_pips REAL,"
    " probability REAL, hit REAL, hit_half_stop REAL, target_usd REAL, loss_usd REAL, best REAL,"
    " trailing INTEGER DEFAULT 0, close_reason TEXT, closed_at TEXT, exit_price REAL, profit_usd REAL, outcome TEXT,"
    " sizing_balance REAL, margin REAL")


def db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, scanned_at TEXT, symbol TEXT, side TEXT, probability REAL,
        hit REAL, ok INTEGER, reasons TEXT, price REAL, atr REAL, stop_pips REAL, target_pips REAL,
        cost_pips REAL, lot REAL, action TEXT)""")
    conn.execute(f"CREATE TABLE IF NOT EXISTS trades ({TRADE_COLUMNS})")
    conn.execute("CREATE TABLE IF NOT EXISTS heartbeat (id INTEGER PRIMARY KEY CHECK (id = 1), pid INTEGER, at TEXT)")
    present = {r["name"] for r in conn.execute("PRAGMA table_info(trades)")}
    for column in TRADE_COLUMNS.split(","):  # eski günlüğe yeni sütunları ekle
        name = column.split()[0]
        if name not in present and "PRIMARY KEY" not in column:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {column.strip()}")
    conn.commit()
    return conn


def other_bot_processes(pid: int) -> list[int]:
    """Çalışan başka forex_bot.py süreçleri (heartbeat yazmayan ilk sürüm dahil); psutil yoksa boş."""
    try:
        import psutil
    except ImportError:
        return []
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = proc.info.get("cmdline") or []
        if proc.info["pid"] != pid and any(str(part).endswith("forex_bot.py") for part in cmd)                 and not {"--report", "--dry-run"} & set(cmd):
            found.append(proc.info["pid"])
    return found


def claim_heartbeat(conn, pid: int) -> None:
    """Aynı anda tek bot: başka bir bot süreci varsa ya da son HEARTBEAT_SECONDS içinde yazdıysa çık."""
    others = other_bot_processes(pid)
    if others:
        raise SystemExit(f"Başka bir bot çalışıyor (PID {', '.join(map(str, others))}). Önce onu durdur (Ctrl+C).")
    row = conn.execute("SELECT pid, at FROM heartbeat WHERE id=1").fetchone()
    if row and row["pid"] != pid and pd.Timestamp.now(tz="UTC") - pd.Timestamp(row["at"]) < pd.Timedelta(
            seconds=HEARTBEAT_SECONDS):
        raise SystemExit(f"Başka bir bot çalışıyor (PID {row['pid']}). Önce onu durdur (Ctrl+C).")
    beat(conn, pid)


def beat(conn, pid: int) -> None:
    conn.execute("INSERT OR REPLACE INTO heartbeat (id, pid, at) VALUES (1, ?, ?)",
                 (pid, str(pd.Timestamp.now(tz="UTC"))))
    conn.commit()


def record_decision(conn, scanned_at, symbol, item: dict, action: str) -> None:
    plan, check = item.get("plan") or {}, item.get("check") or {}
    conn.execute(
        "INSERT INTO decisions (scanned_at, symbol, side, probability, hit, ok, reasons, price, atr, stop_pips,"
        " target_pips, cost_pips, lot, action) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (str(scanned_at), symbol, plan.get("side"), item.get("probability_up"), check.get("hit"),
         int(bool(check.get("ok"))), json.dumps(check.get("reasons") or item.get("reasons") or [], ensure_ascii=False),
         item.get("price"), item.get("atr"), plan.get("stop_pips"), plan.get("target_pips"),
         item.get("cost_pips"), plan.get("lot"), action))
    conn.commit()


# ----------------------------------------------------------------------------- MT5

def connect():
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"MT5'e bağlanılamadı: {mt5.last_error()}. Terminal açık ve hesaba giriş yapılmış olmalı.")
    account = mt5.account_info()
    if account is None or account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        mt5.shutdown()
        raise SystemExit("Bu bot yalnız DEMO hesapta çalışır; bağlı hesap demo değil.")
    return mt5


def own_positions(mt5) -> list:
    return [p for p in (mt5.positions_get() or ()) if p.magic == MAGIC]


def ml_probability(mt5, name: str, now: pd.Timestamp) -> Optional[float]:
    end = now.to_pydatetime()
    hourly = rates_utc(mt5, name, mt5.TIMEFRAME_H1, end - timedelta(days=150), end)
    prediction = build_research_prediction(name, bars=hourly, now=now) if not hourly.empty else {}
    return float(prediction["probability_up"]) if prediction.get("status") == "ready" else None


def pair_inputs(mt5, symbol: str) -> dict:
    """Karar için canlı girdiler: fiyat, ML yön olasılığı, 15M ATR, maliyet ve broker pip değeri."""
    name = symbol.replace("=X", "")
    if not mt5.symbol_select(name, True):
        return {"reasons": ["Sembol MT5'te yok."]}
    info, tick = mt5.symbol_info(name), mt5.symbol_info_tick(name)
    now = pd.Timestamp.now(tz="UTC")
    if tick is None or (now - server_time_to_utc(tick.time, server_offset(mt5, name))).total_seconds() > MAX_TICK_AGE_SECONDS:
        return {"reasons": ["Piyasa kapalı veya fiyat akmıyor."]}
    end = now.to_pydatetime()
    m15 = rates_utc(mt5, name, mt5.TIMEFRAME_M15, end - timedelta(days=10), end)
    if len(m15) < 40:
        return {"reasons": ["15M geçmiş alınamadı."]}
    probability_up = ml_probability(mt5, name, now)
    if probability_up is None:
        return {"reasons": ["ML tahmini yok."]}
    row = latest_valid_row(add_indicators(m15.iloc[:-1]))
    pip = get_pip_size(symbol)
    spread_points = pd.to_numeric(m15["Spreadpoints"], errors="coerce").tail(100).median()
    return {
        "name": name, "info": info, "pip": pip,
        "price": (float(tick.bid) + float(tick.ask)) / 2,
        "probability_up": probability_up,
        "atr": float(row["ATR14"]) if row is not None else None,
        "cost_pips": float(spread_points) * info.point / pip + ECN_COMMISSION_PIPS,
        "spread_pips": (float(tick.ask) - float(tick.bid)) / pip,
        "pip_value": float(info.trade_tick_value) * pip / float(info.trade_tick_size),
    }


def virtual_balance(conn, start: float) -> float:
    """Sanal hesap: başlangıç + sanal bakiyeyle boyutlanmış ve kapanmış bot işlemlerinin net sonucu."""
    total = conn.execute("SELECT COALESCE(SUM(profit_usd), 0) FROM trades "
                         "WHERE sizing_balance IS NOT NULL AND closed_at IS NOT NULL").fetchone()[0]
    return float(start) + float(total)


def budget(balance: float, args) -> tuple[float, float]:
    """İşlem başına (hedef $, stop zararı $): elle verilmediyse bakiyenin yüzdesi ve onun katı."""
    loss = args.loss_usd if args.loss_usd else balance * args.risk_pct / 100
    target = args.target_usd if args.target_usd else loss * args.target_r
    return float(target), float(loss)


def decide(item: dict, target_usd: float, loss_usd: float, account_usd: float, stats: dict) -> dict:
    """Karar kutusunun hesabı; ayrıca erken çıkışla fiilen yarıya inen stopa göre oran (yalnız kayıt için)."""
    pair = item["name"].upper()
    plan = build_trade_plan(pair, item["price"], item["pip"], item["atr"], item["probability_up"],
                            target_usd, loss_usd, account_usd, item["pip_value"], item["cost_pips"], {}, [])
    item["plan"] = plan
    if plan["status"] != "ready":
        item["check"] = {"ok": False, "reasons": [plan.get("reason", "Plan hesaplanamadı.")]}
        return item
    item["check"] = target_check(stats, pair, True, plan["target_pips"], plan["stop_pips"], item["cost_pips"])
    item["check_half"] = target_check(stats, pair, True, plan["target_pips"], plan["stop_pips"] * EARLY_EXIT_SHARE,
                                      item["cost_pips"])
    return item


def volume_for(info, lot: float) -> float:
    step = float(info.volume_step) or 0.01
    lot = math.floor(lot / step + 1e-9) * step
    return round(min(max(lot, float(info.volume_min)), float(info.volume_max)), 2)


def filling(mt5, info) -> int:
    if info.filling_mode & 1:
        return mt5.ORDER_FILLING_FOK
    if info.filling_mode & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def margin_for(mt5, item: dict, volume: float, leverage: float) -> Optional[float]:
    """Pozisyonun teminatı; gerçek hesabın kaldıracı demo hesabınkinden farklıysa ona göre ölçeklenir."""
    tick = mt5.symbol_info_tick(item["name"])
    buy = item["plan"]["side"] == "LONG"
    margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL, item["name"], volume,
                                   float(tick.ask if buy else tick.bid))
    return None if margin is None else float(margin) * mt5.account_info().leverage / leverage


def open_position(mt5, conn, item: dict, args, signal_hour: pd.Timestamp) -> Optional[int]:
    info, plan = item["info"], item["plan"]
    tick = mt5.symbol_info_tick(item["name"])
    buy = plan["side"] == "LONG"
    sign = 1 if buy else -1
    volume = volume_for(info, plan["lot"])
    balance = virtual_balance(conn, args.balance)
    margin = margin_for(mt5, item, volume, args.leverage or mt5.account_info().leverage)
    used = conn.execute("SELECT COALESCE(SUM(margin), 0) FROM trades WHERE closed_at IS NULL "
                        "AND sizing_balance IS NOT NULL").fetchone()[0]
    if margin is None or used + margin > balance * MAX_MARGIN_SHARE:
        log(f"– {item['name']} atlandı: teminat {0 if margin is None else margin:.2f} $ + açık {used:.2f} $ > "
            f"sanal bakiyenin %{MAX_MARGIN_SHARE * 100:.0f}'i ({balance * MAX_MARGIN_SHARE:.2f} $)")
        return None
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": item["name"], "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL,
        "price": float(tick.ask if buy else tick.bid),
        "sl": round(plan["stop"], info.digits), "tp": 0.0,
        "deviation": DEVIATION_POINTS, "magic": MAGIC, "comment": COMMENT,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling(mt5, info),
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        detail = mt5.last_error() if result is None else f"{result.retcode} {result.comment}"
        log(f"✗ {item['name']} {plan['side']} emri reddedildi: {detail}")
        return None
    ticket = int(result.order)
    early_exit = plan["entry"] - sign * plan["stop_pips"] * EARLY_EXIT_SHARE * item["pip"]
    conn.execute(
        "INSERT OR REPLACE INTO trades (ticket, symbol, side, opened_at, signal_hour, lot, entry, fill, stop, target,"
        " early_exit, atr, pip, cost_pips, spread_pips, probability, hit, hit_half_stop, target_usd, loss_usd, best,"
        " sizing_balance, margin) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ticket, item["name"], plan["side"], str(pd.Timestamp.now(tz="UTC")), str(signal_hour), request["volume"],
         plan["entry"], float(result.price), request["sl"], plan["target"], early_exit, item["atr"], item["pip"],
         item["cost_pips"], item["spread_pips"], item["probability_up"], item["check"]["hit"],
         item["check_half"].get("hit"), item["target_usd"], item["loss_usd"], float(result.price), balance, margin))
    conn.commit()
    log(f"✓ {item['name']} {plan['side']} {request['volume']:.2f} lot @ {result.price} · stop {request['sl']} · "
        f"hedef/takip başlangıcı {round(plan['target'], info.digits)} · erken çıkış {round(early_exit, info.digits)} · "
        f"ulaşma ≈%{item['check']['hit'] * 100:.0f}")
    return ticket


def close_position(mt5, conn, position, reason: str) -> bool:
    tick = mt5.symbol_info_tick(position.symbol)
    info = mt5.symbol_info(position.symbol)
    buy = position.type == mt5.POSITION_TYPE_BUY
    conn.execute("UPDATE trades SET close_reason=? WHERE ticket=?", (reason, position.ticket))
    conn.commit()
    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL, "symbol": position.symbol, "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if buy else mt5.ORDER_TYPE_BUY, "position": position.ticket,
        "price": float(tick.bid if buy else tick.ask), "deviation": DEVIATION_POINTS, "magic": MAGIC,
        "comment": COMMENT, "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling(mt5, info),
    })
    ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    log(f"{'✓' if ok else '✗'} {position.symbol} kapatma ({reason}): "
        f"{'tamam' if ok else (mt5.last_error() if result is None else result.comment)}")
    return ok


def move_stop(mt5, position, new_sl: float, digits: int) -> bool:
    result = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": position.symbol,
                             "position": position.ticket, "sl": round(new_sl, digits), "tp": 0.0})
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


def next_stop(side: str, entry: float, current_sl: float, best: float, price: float, target: float, atr: float,
              cost_pips: float, pip: float, trailing: bool) -> tuple[Optional[float], bool]:
    """Kâr tarafı: hedefe gelince stop giriş + maliyete, sonra en iyi fiyatın TRAIL_ATR × ATR gerisine.

    Döner: (yeni stop ya da None, takip başladı mı). Stop yalnız lehe ve fiyatın doğru tarafında taşınır.
    """
    sign = 1 if side == "LONG" else -1
    trailing = trailing or sign * (best - target) >= 0
    if not trailing:
        return None, False
    floor = entry + sign * cost_pips * pip
    candidate = best - sign * TRAIL_ATR * atr
    wanted = max(floor, candidate) if sign == 1 else min(floor, candidate)
    if sign * (wanted - current_sl) < MIN_SL_STEP_PIPS * pip or sign * (price - wanted) <= 0:
        return None, True
    return wanted, True


def legacy_fields(side: str, entry: float, sl: float, tp: float, pip: float) -> dict:
    """İlk sürümün (sabit TP, saatlik tarama) açtığı pozisyonu yeni yönetime uyarlar.

    Hedef eski TP, erken çıkış stop mesafesinin yarısı; ATR stop = 1,5 ATR'den geri hesaplanır.
    Maliyet bilinmediği için yalnız komisyon alınır (Pepperstone ham spread ≈ 0).
    """
    sign = 1 if side == "LONG" else -1
    stop_distance = abs(entry - sl)
    return {"target": tp, "early_exit": entry - sign * stop_distance * EARLY_EXIT_SHARE,
            "atr": stop_distance / 1.5, "pip": pip, "cost_pips": ECN_COMMISSION_PIPS, "best": entry}


def adopt_legacy(mt5, conn, position, row) -> Optional[sqlite3.Row]:
    if row["target"] is not None:
        return row
    if not position.tp or not position.sl:
        return None
    fields = legacy_fields(row["side"], float(row["fill"] or position.price_open), float(position.sl),
                           float(position.tp), get_pip_size(position.symbol))
    fields["entry"] = float(row["entry"] or position.price_open)
    conn.execute(f"UPDATE trades SET {', '.join(f'{k}=?' for k in fields)} WHERE ticket=?",
                 (*fields.values(), position.ticket))
    conn.commit()
    info = mt5.symbol_info(position.symbol)
    if move_stop(mt5, position, float(position.sl), info.digits):  # TP kaldırılır, stop aynı kalır
        log(f"⇄ {position.symbol} ilk sürümden devralındı: sabit hedef kaldırıldı, takip/erken çıkış uygulanacak.")
    return conn.execute("SELECT * FROM trades WHERE ticket=?", (position.ticket,)).fetchone()


def manage(mt5, conn, check_ml: bool) -> None:
    """Açık pozisyonları yönetir ve kapananların sonucunu günlüğe yazar."""
    now = pd.Timestamp.now(tz="UTC")
    for position in own_positions(mt5):
        row = conn.execute("SELECT * FROM trades WHERE ticket=?", (position.ticket,)).fetchone()
        if row is None:
            continue
        row = adopt_legacy(mt5, conn, position, row)
        if row is None:
            continue
        tick, info = mt5.symbol_info_tick(position.symbol), mt5.symbol_info(position.symbol)
        sign = 1 if row["side"] == "LONG" else -1
        price = float(tick.bid if sign == 1 else tick.ask)  # pozisyonu kapatacak fiyat
        best = max(row["best"], price) if sign == 1 else min(row["best"], price)
        if best != row["best"]:
            conn.execute("UPDATE trades SET best=? WHERE ticket=?", (best, position.ticket))
        trailing = bool(row["trailing"])
        new_sl, now_trailing = next_stop(row["side"], row["entry"], float(position.sl), best, price, row["target"],
                                         row["atr"], row["cost_pips"], row["pip"], trailing)
        if now_trailing and not trailing:
            conn.execute("UPDATE trades SET trailing=1 WHERE ticket=?", (position.ticket,))
            log(f"↗ {position.symbol} hedefe geldi; stop maliyet dahil başabaşa çekiliyor, takip başladı.")
        if new_sl is not None and move_stop(mt5, position, new_sl, info.digits):
            log(f"↗ {position.symbol} stop {position.sl} → {round(new_sl, info.digits)}")
        conn.commit()
        if now_trailing:
            continue
        opened = server_time_to_utc(position.time, server_offset(mt5, position.symbol))
        if sign * (price - row["early_exit"]) <= 0:
            close_position(mt5, conn, position, "YARIM STOP")
        elif now - opened >= pd.Timedelta(hours=HOLD_HOURS):
            close_position(mt5, conn, position, "SÜRE")
        elif check_ml:
            probability_up = ml_probability(mt5, position.symbol, now)
            if probability_up is not None and probability_up != 0.5 and (probability_up > 0.5) != (sign == 1):
                close_position(mt5, conn, position, "ML YÖN DÖNDÜ")
    sync_closed(mt5, conn)


def sync_closed(mt5, conn) -> None:
    open_tickets = {p.ticket for p in own_positions(mt5)}
    for row in conn.execute("SELECT ticket, close_reason, trailing FROM trades WHERE closed_at IS NULL").fetchall():
        if row["ticket"] in open_tickets:
            continue
        deals = mt5.history_deals_get(position=row["ticket"]) or ()
        exits = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
        if not exits:
            continue
        last = exits[-1]
        if last.reason == mt5.DEAL_REASON_SL:
            outcome = "TAKİP STOPU" if row["trailing"] else "STOP"
        elif last.reason == mt5.DEAL_REASON_EXPERT and row["close_reason"]:
            outcome = row["close_reason"]
        else:
            outcome = "ELLE/DİĞER"
        profit = sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0.0) for d in deals)
        closed_at = server_time_to_utc(last.time, server_offset(mt5, last.symbol))
        conn.execute("UPDATE trades SET closed_at=?, exit_price=?, profit_usd=?, outcome=? WHERE ticket=?",
                     (str(closed_at), float(last.price), float(profit), outcome, row["ticket"]))
        conn.commit()
        log(f"• {last.symbol} kapandı: {outcome} · {profit:+.2f} $")


def scan(mt5, conn, args, quiet: bool = False) -> None:
    stats = load_plan_stats()
    balance = virtual_balance(conn, args.balance)
    target_usd, loss_usd = budget(balance, args)
    now = pd.Timestamp.now(tz="UTC")
    scanned_at, signal_hour = now.floor("s"), now.floor("h")
    held = {p.symbol for p in own_positions(mt5)}
    entered = {r["symbol"] for r in conn.execute("SELECT symbol FROM trades WHERE signal_hour=?", (str(signal_hour),))}
    slots = max(args.max_positions - len(held), 0)
    if slots == 0 and not args.dry_run:
        return
    candidates = []
    for symbol in SYMBOL_LIST:
        name = symbol.replace("=X", "")
        if not has_research_model(name, "direction") or name in held or name in entered:
            continue
        try:
            item = pair_inputs(mt5, symbol)
            if "name" in item:
                item = decide(item, target_usd, loss_usd, balance, stats)
                item.update(target_usd=target_usd, loss_usd=loss_usd)
        except Exception as exc:  # tek paritenin hatası taramayı durdurmasın
            item = {"reasons": [f"Hata: {exc}"]}
        if item.get("check", {}).get("ok"):
            candidates.append(item)
        elif not quiet:
            record_decision(conn, scanned_at, name, item, "BEKLE")
    candidates.sort(key=lambda x: -x["check"]["hit"])
    opened = 0
    for item in candidates:
        if args.dry_run:
            record_decision(conn, scanned_at, item["name"], item, "DRY-RUN")
            log(f"(deneme) {item['name']} {item['plan']['side']} {item['plan']['lot']:.2f} lot · ulaşma "
                f"≈%{item['check']['hit'] * 100:.0f} (yarım stopa göre ≈%{item['check_half']['hit'] * 100:.0f})")
            continue
        if opened >= slots:
            if not quiet:
                record_decision(conn, scanned_at, item["name"], item, "LİMİT DOLU")
            continue
        ticket = open_position(mt5, conn, item, args, signal_hour)
        record_decision(conn, scanned_at, item["name"], item, "AÇILDI" if ticket else "AÇILAMADI")
        opened += bool(ticket)
    if opened or not quiet:
        log(f"Tarama: {len(candidates)} GİR adayı, {opened} işlem açıldı, açık pozisyon "
            f"{len(own_positions(mt5))}/{args.max_positions}.")


def report() -> None:
    conn = db()
    trades = pd.read_sql("SELECT * FROM trades", conn)
    decisions = pd.read_sql("SELECT action, COUNT(*) n FROM decisions GROUP BY action", conn)
    print("Kararlar:\n" + (decisions.to_string(index=False) if not decisions.empty else "  henüz yok"))
    closed = trades.dropna(subset=["closed_at"])
    print(f"\nİşlem: {len(trades)} açıldı, {len(closed)} kapandı, {len(trades) - len(closed)} açık.")
    sized = closed[closed["sizing_balance"].notna()] if "sizing_balance" in closed else closed.iloc[0:0]
    if not trades.empty and trades["sizing_balance"].notna().any():
        start = float(trades.loc[trades["sizing_balance"].notna(), "sizing_balance"].iloc[0])
        result = float(sized["profit_usd"].sum())
        print(f"Sanal hesap: {start:.2f} $ → {start + result:.2f} $ ({result / start * 100:+.1f}%)")
    if closed.empty:
        return
    reached = closed["trailing"].astype(bool).mean()
    print(f"Hedefe (takip başlangıcına) gelen: %{reached * 100:.0f} · ölçümün beklediği: tam stopa göre "
          f"%{closed['hit'].mean() * 100:.0f}, yarım stopa göre %{closed['hit_half_stop'].mean() * 100:.0f}")
    print(f"Net sonuç: {closed['profit_usd'].sum():+.2f} $ · işlem başına {closed['profit_usd'].mean():+.2f} $")
    by_outcome = closed.groupby("outcome")["profit_usd"].agg(["count", "mean", "sum"]).round(2)
    print("\nKapanış sebebi (adet · ortalama $ · toplam $):\n" + by_outcome.to_string())
    if len(closed) < 100:
        print(f"\nNot: {len(closed)} işlem az; oran ±{100 / math.sqrt(len(closed)):.0f} puan kadar oynayabilir.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--balance", type=float, default=500.0, help="Sanal başlangıç bakiyesi ($)")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="İşlem başına stop zararı, sanal bakiyenin %%'si")
    parser.add_argument("--target-r", type=float, default=0.3, help="Takip başlangıcı: stop zararının katı")
    parser.add_argument("--target-usd", type=float, help="Sabit hedef $ (verilirse --target-r yerine)")
    parser.add_argument("--loss-usd", type=float, help="Sabit stop zararı $ (verilirse --risk-pct yerine)")
    parser.add_argument("--leverage", type=float, help="Gerçek hesabın kaldıracı (teminat için; varsayılan demo)")
    parser.add_argument("--max-positions", type=int, default=5, help="Aynı anda en fazla açık pozisyon")
    parser.add_argument("--dry-run", action="store_true", help="Emir gönderme, yalnız kararları yaz")
    parser.add_argument("--once", action="store_true", help="Tek tarama yap ve çık")
    parser.add_argument("--report", action="store_true", help="Günlükteki sonuçları göster")
    args = parser.parse_args()
    if args.report:
        report()
        return
    mt5 = connect()
    if not args.dry_run and not mt5.terminal_info().trade_allowed:
        mt5.shutdown()
        raise SystemExit("MT5'te 'Algo Trading' kapalı. Terminalde üst çubuktaki Algo Trading düğmesine bas, sonra tekrar çalıştır.")
    conn = db()
    if not args.dry_run:
        claim_heartbeat(conn, os.getpid())
    account = mt5.account_info()
    balance = virtual_balance(conn, args.balance)
    target_usd, loss_usd = budget(balance, args)
    log(f"Bot başladı · {account.server} #{account.login} (DEMO) · sanal bakiye {balance:,.2f} $ · "
        f"stop zararı {loss_usd:.2f} $ / takip başlangıcı {target_usd:.2f} $ · kaldıraç 1:"
        f"{args.leverage or account.leverage:.0f} · en fazla {args.max_positions} pozisyon"
        + (" · DENEME (emir yok)" if args.dry_run else ""))
    last_scan, last_ml_hour = None, pd.Timestamp.now(tz="UTC").floor("h")
    try:
        while True:
            now = pd.Timestamp.now(tz="UTC")
            hour = now.floor("h")
            check_ml = hour > last_ml_hour and (now - hour).total_seconds() >= ML_CHECK_DELAY_SECONDS
            if not args.dry_run:
                beat(conn, os.getpid())
                manage(mt5, conn, check_ml)
            if check_ml:
                last_ml_hour = hour
            if last_scan is None or (now - last_scan).total_seconds() >= SCAN_SECONDS:
                scan(mt5, conn, args, quiet=last_scan is not None and last_scan.floor("h") == hour)
                last_scan = now
                if args.once:
                    break
            time.sleep(MANAGE_SECONDS)
    except KeyboardInterrupt:
        log("Bot durduruldu. Açık pozisyonların stopu sunucuda durur; takip ve erken çıkış bot açıkken çalışır.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
