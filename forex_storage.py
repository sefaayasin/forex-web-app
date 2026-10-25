"""SQLite persistence and webhook delivery for Forex Analyzer Pro."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
import pytz

from forex_config import normalize_symbol


APP_DIR = Path(__file__).resolve().parent
APP_DB_PATH = APP_DIR / "forex_analyzer.db"
TR_TZ = pytz.timezone("Europe/Istanbul")
LOGGER = logging.getLogger("forex_analyzer")


def init_trade_journal() -> None:
    with closing(sqlite3.connect(APP_DB_PATH)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS alert_history (
                alert_key TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                payload TEXT NOT NULL
            )"""
        )
        conn.commit()


def add_trade_journal_entry(entry: dict) -> None:
    init_trade_journal()
    payload = json.dumps(entry, ensure_ascii=False, default=str)
    with closing(sqlite3.connect(APP_DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO trade_journal(created_at, payload) VALUES (?, ?)",
            (datetime.now(tz=TR_TZ).isoformat(), payload),
        )
        conn.commit()


def journal_dataframe() -> pd.DataFrame:
    init_trade_journal()
    with closing(sqlite3.connect(APP_DB_PATH)) as conn:
        rows = conn.execute("SELECT id, created_at, payload FROM trade_journal ORDER BY id DESC").fetchall()
    records = []
    for row_id, created_at, payload in rows:
        try:
            item = json.loads(payload)
        except json.JSONDecodeError:
            continue
        item["Kayıt ID"] = row_id
        item.setdefault("Kayıt Zamanı", created_at)
        records.append(item)
    return pd.DataFrame(records)


def clear_trade_journal() -> None:
    init_trade_journal()
    with closing(sqlite3.connect(APP_DB_PATH)) as conn:
        conn.execute("DELETE FROM trade_journal")
        conn.commit()


def _init_plan_tracking(conn) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS tracked_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL, provider TEXT NOT NULL,
        created_at TEXT NOT NULL, archived_at TEXT, payload TEXT NOT NULL
    )""")
    conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS one_active_plan_per_source
        ON tracked_plans(symbol, provider) WHERE archived_at IS NULL""")


def save_tracked_plan(plan: dict) -> int:
    """Freeze one active plan per symbol/source; refreshes never move its levels."""
    from forex_position_chart import validate_plan
    validate_plan(plan)
    with closing(sqlite3.connect(APP_DB_PATH)) as conn:
        _init_plan_tracking(conn)
        conn.execute("INSERT OR IGNORE INTO tracked_plans(symbol, provider, created_at, payload) VALUES (?, ?, ?, ?)",
                     (plan["symbol"], plan["provider"], plan["tracked_at"],
                      json.dumps(plan, ensure_ascii=False, allow_nan=False)))
        row = conn.execute("SELECT id FROM tracked_plans WHERE symbol=? AND provider=? AND archived_at IS NULL",
                           (plan["symbol"], plan["provider"])).fetchone()
        conn.commit()
    return int(row[0])


def tracked_plans(provider: str | None = None) -> list[dict]:
    with closing(sqlite3.connect(APP_DB_PATH)) as conn:
        _init_plan_tracking(conn)
        sql, params = "SELECT id, payload FROM tracked_plans WHERE archived_at IS NULL", ()
        if provider is not None:
            sql, params = sql + " AND provider=?", (provider,)
        rows = conn.execute(sql + " ORDER BY id DESC", params).fetchall()
        conn.commit()
    plans = []
    for row_id, payload in rows:
        try:
            plans.append({**json.loads(payload), "id": row_id})
        except (ValueError, TypeError):
            LOGGER.warning("Tracked plan %s contains invalid JSON", row_id)
    return plans


def archive_tracked_plan(plan_id: int) -> None:
    """Archive a visual watch plan, without declaring a broker trade closed."""
    with closing(sqlite3.connect(APP_DB_PATH)) as conn:
        _init_plan_tracking(conn)
        conn.execute("UPDATE tracked_plans SET archived_at=? WHERE id=? AND archived_at IS NULL",
                     (datetime.now(tz=TR_TZ).isoformat(), int(plan_id)))
        conn.commit()


def record_alert_once(symbol: str, side: str, candle_time: str, payload: dict) -> bool:
    """Store a symbol/side/candle alert once and report whether it was new."""
    init_trade_journal()
    alert_key = f"{normalize_symbol(symbol)}|{side}|{candle_time}"
    try:
        with closing(sqlite3.connect(APP_DB_PATH)) as conn:
            conn.execute(
                "INSERT INTO alert_history(alert_key, created_at, symbol, side, payload) VALUES (?, ?, ?, ?, ?)",
                (
                    alert_key,
                    datetime.now(tz=TR_TZ).isoformat(),
                    normalize_symbol(symbol),
                    side,
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def alert_history_dataframe(limit: int = 200) -> pd.DataFrame:
    init_trade_journal()
    with closing(sqlite3.connect(APP_DB_PATH)) as conn:
        return pd.read_sql_query(
            "SELECT created_at AS Zaman, symbol AS Sembol, side AS Yön, payload AS Detay "
            "FROM alert_history ORDER BY created_at DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )


def send_webhook_notification(webhook_url: str, payload: dict) -> tuple[bool, str]:
    if not str(webhook_url).strip():
        return False, "Webhook adresi tanımlı değil."
    try:
        request = Request(
            str(webhook_url).strip(),
            data=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "ForexAnalyzer/1.0"},
            method="POST",
        )
        with urlopen(request, timeout=8) as response:
            return 200 <= response.status < 300, f"HTTP {response.status}"
    except (URLError, ValueError, TimeoutError) as exc:
        LOGGER.warning("Webhook notification failed: %s", exc)
        return False, str(exc)
