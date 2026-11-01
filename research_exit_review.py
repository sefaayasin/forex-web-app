"""Read-only MT5 exit audit. Never sends orders or changes the live journal.

Capture raw deal/tick times together to avoid today's DST offset changing old
trade durations. Counterfactual cash results use observed deal cash sensitivity;
cross-currency conversion and fills are approximate, not a portfolio backtest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research/exit_review_20261101"


def capture():
    import MetaTrader5 as mt5

    OUT.mkdir(parents=True, exist_ok=True)
    if not mt5.initialize():
        raise RuntimeError(mt5.last_error())
    try:
        account = mt5.account_info()
        if account is None or account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RuntimeError("Expected the existing demo account")
        with sqlite3.connect((ROOT / "data/bot_journal.db").as_uri() + "?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            journal = [dict(r) for r in conn.execute(
                "SELECT * FROM trades WHERE sizing_balance IS NOT NULL ORDER BY opened_at")]
        trades, windows = [], {}
        for row in journal:
            deals = mt5.history_deals_get(position=row["ticket"])
            if deals is None:
                raise RuntimeError(mt5.last_error())
            entries = [d for d in deals if d.entry == mt5.DEAL_ENTRY_IN]
            exits = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
            if not entries or not exits:
                continue
            start, end = min(d.time_msc for d in entries), max(d.time_msc for d in exits)
            finish = max(start + 4 * 3600000, end + 3600000)
            old = windows.get(row["symbol"], (start, finish))
            windows[row["symbol"]] = min(start, old[0]), max(finish, old[1])
            trades.append({"journal": row, "deals": [d._asdict() for d in deals],
                           "open_raw_msc": start, "close_raw_msc": end})
        quality = {}
        for symbol, (start, end) in windows.items():
            raw = mt5.copy_ticks_range(symbol, pd.to_datetime(start, unit="ms", utc=True).to_pydatetime(),
                                      pd.to_datetime(end, unit="ms", utc=True).to_pydatetime(), mt5.COPY_TICKS_ALL)
            if raw is None or not len(raw):
                raise RuntimeError(f"No ticks: {symbol}: {mt5.last_error()}")
            ticks = pd.DataFrame(raw).sort_values("time_msc", kind="stable")
            ticks = ticks[(ticks.bid > 0) & (ticks.ask >= ticks.bid)]
            if ticks.empty:
                raise RuntimeError(f"No valid quotes: {symbol}")
            np.savez_compressed(OUT / f"{symbol}_ticks.npz",
                                **{k: ticks[k].to_numpy() for k in ("time_msc", "bid", "ask")})
            quality[symbol] = {"count": len(ticks), "start_raw_msc": int(ticks.time_msc.iloc[0]),
                               "end_raw_msc": int(ticks.time_msc.iloc[-1]),
                               "max_gap_seconds": float(ticks.time_msc.diff().max() / 1000)}
            print(symbol, len(ticks), flush=True)
        positions = mt5.positions_get()
        if positions is None:
            raise RuntimeError(f"Position lookup failed: {mt5.last_error()}")
        snapshot = {"captured_utc": str(pd.Timestamp.now(tz="UTC")), "trades": trades, "ticks": quality,
                    "open_positions": len(positions),
                    "time_note": "All paths match raw deal/tick timestamps; no current server-offset subtraction."}
        (OUT / "capture.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        mt5.shutdown()


def analyze():
    snapshot = json.loads((OUT / "capture.json").read_text(encoding="utf-8"))
    rows = []
    for trade in snapshot["trades"]:
        row, deals = trade["journal"], trade["deals"]
        entries = [d for d in deals if d["entry"] == 0]
        exits = [d for d in deals if d["entry"] == 1]
        # This audit compares full single-entry/single-exit positions only.
        if len(entries) != 1 or len(exits) != 1 or entries[0]["volume"] != exits[0]["volume"]:
            raise ValueError(f"Unsupported partial/reversal trade: {row['ticket']}")
        entry, close = entries[0], exits[0]
        sign = 1 if row["side"] == "LONG" else -1
        fees = sum(d["commission"] + d["swap"] + d["fee"] for d in deals)
        actual = sum(d["profit"] for d in deals) + fees
        delta = sign * (close["price"] - entry["price"])
        sensitivity = close["profit"] / delta if abs(delta) > 1e-12 else None
        reason = ("HEDEF" if close["reason"] == 5 else
                  ("TAKİP STOPU" if row["trailing"] else "STOP") if close["reason"] == 4 else
                  row["close_reason"] or "ELLE/DİĞER")
        r = {k: row[k] for k in ("ticket", "symbol", "side", "opened_at", "exit_mode", "target_usd", "loss_usd")}
        r.update(actual_net=actual, outcome=reason, journal_stale=row["closed_at"] is None,
                 duration_min=(close["time_msc"] - entry["time_msc"]) / 60000,
                 ml_confidence=max(row["probability"], 1 - row["probability"]))
        with np.load(OUT / f"{row['symbol']}_ticks.npz") as ticks:
            times = ticks["time_msc"]
            prices = ticks["bid" if sign == 1 else "ask"]
            # First executable quote at/after exit + 1h, with 60s tolerance.
            horizon = close["time_msc"] + 3600000
            i = int(np.searchsorted(times, horizon))
            if i < len(times) and times[i] - horizon <= 60000 and sensitivity is not None:
                r["unprotected_net_exit_plus_60m"] = sign * (prices[i] - entry["price"]) * sensitivity + fees
            # Same entry, recorded half-stop and fixed TP, cap at entry + 4h.
            mask = (times >= entry["time_msc"]) & (times <= entry["time_msc"] + 4 * 3600000)
            path, stamps = prices[mask], times[mask]
            if len(path) and stamps[0] - entry["time_msc"] <= 60000 and sensitivity is not None:
                stop = row["early_exit"]
                hit = (sign * (path - row["target"]) >= 0) | (sign * (path - stop) <= 0)
                hits = np.flatnonzero(hit)
                j = int(hits[0]) if len(hits) else len(path) - 1
                why = ("TARGET" if sign * (path[j] - row["target"]) >= 0 else "HALF_STOP") if len(hits) else "TIME_4H"
                # Incomplete paths cannot be marked as a full four-hour hold.
                if len(hits) or entry["time_msc"] + 4 * 3600000 - stamps[-1] <= 60000:
                    r.update(fixed_half_4h_net=sign * (path[j] - entry["price"]) * sensitivity + fees,
                             fixed_half_4h_reason=why)
            r["tick_max_gap_seconds"] = float(np.diff(stamps).max() / 1000) if len(stamps) > 1 else None
        rows.append(r)
    frame = pd.DataFrame(rows)
    assert frame.ticket.is_unique
    frame.to_csv(OUT / "ledger.csv", index=False, encoding="utf-8-sig")
    summary = {"trades": len(frame), "net": float(frame.actual_net.sum()),
               "stale_journal": int(frame.journal_stale.sum()),
               "last6_net": float(frame.tail(6).actual_net.sum()),
               "groups": frame.groupby(["exit_mode", "outcome"]).agg(
                   n=("ticket", "size"), net=("actual_net", "sum"),
                   mean=("actual_net", "mean")).reset_index().to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(frame.tail(6).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "analyze"))
    args = parser.parse_args()
    capture() if args.action == "capture" else analyze()
