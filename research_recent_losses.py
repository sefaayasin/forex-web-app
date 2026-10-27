"""Read-only MT5 loss review and bounded exit-policy experiments.

capture reads MT5 deals/quotes only; replay/history do not connect to MT5.
No order-sending functions are used. Outputs are separate from the live journal.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research/recent_losses_20261027"
FOCUS = {411021217, 411045182, 411059397}


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def capture():
    import MetaTrader5 as mt5
    from forex_mt5 import server_offset

    if not mt5.initialize():
        raise RuntimeError(mt5.last_error())
    try:
        account = mt5.account_info()
        if account is None or account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RuntimeError("Expected the existing demo account")
        captured = pd.Timestamp.now(tz="UTC")
        conn = sqlite3.connect((ROOT / "data/bot_journal.db").resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        journal = [dict(row) for row in conn.execute(
            "SELECT * FROM trades WHERE opened_at >= ? AND sizing_balance IS NOT NULL ORDER BY opened_at",
            ("2026-10-27 00:00:00",))]
        conn.close()
        trades, windows = [], {}
        fields = ("ticket", "time_msc", "entry", "position_id", "volume", "price", "profit", "commission", "swap", "fee", "reason")
        offsets = {}
        for row in journal:
            symbol = row["symbol"]
            deals = mt5.history_deals_get(position=row["ticket"])
            if deals is None:
                raise RuntimeError(f"Deal lookup failed: {symbol}: {mt5.last_error()}")
            entries = [deal for deal in deals if deal.entry == mt5.DEAL_ENTRY_IN]
            exits = [deal for deal in deals if deal.entry == mt5.DEAL_ENTRY_OUT]
            if not entries or not exits:
                continue  # Open positions are excluded, never assigned a hypothetical win/loss.
            start, end = min(d.time_msc for d in entries), max(d.time_msc for d in exits)
            offsets.setdefault(symbol, server_offset(mt5, symbol).total_seconds())
            windows[symbol] = min(windows.get(symbol, start), start)
            trades.append({"journal": row, "focus": row["ticket"] in FOCUS,
                "deals": [{key: getattr(deal, key, 0) for key in fields} for deal in deals],
                "open_msc_raw": start, "close_msc_raw": end,
                "actual_net": sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0) for d in deals)})
        tick_info = {}
        for symbol, start in windows.items():
            end = captured + pd.Timedelta(seconds=offsets[symbol])
            raw = mt5.copy_ticks_range(symbol,
                pd.to_datetime(start, unit="ms", utc=True).to_pydatetime(), end.to_pydatetime(), mt5.COPY_TICKS_ALL)
            if raw is None or not len(raw):
                raise RuntimeError(f"Tick download failed: {symbol}: {mt5.last_error()}")
            ticks = pd.DataFrame(raw)[["time_msc", "bid", "ask"]].sort_values("time_msc", kind="stable")
            ticks = ticks[(ticks.bid > 0) & (ticks.ask >= ticks.bid)]
            np.savez_compressed(OUT / f"{symbol}_ticks.npz", **{key: ticks[key].to_numpy() for key in ticks})
            tick_info[symbol] = {"n": len(ticks), "start_raw_msc": int(ticks.time_msc.iloc[0]),
                "end_raw_msc": int(ticks.time_msc.iloc[-1]),
                "max_gap_seconds": float(ticks.time_msc.diff().max() / 1000),
                "observed_server_offset_seconds": offsets[symbol]}
            print(symbol, "ticks", len(ticks), flush=True)
        write("capture.json", {"captured_utc": str(captured), "trades": trades, "ticks": tick_info,
            "time_note": "Match raw deal and tick milliseconds directly. Current broker timestamps are observed shifted; offset is recorded, not assumed from documentation."})
        print("Closed trades", len(trades), "focus", sum(t["focus"] for t in trades))
    finally:
        mt5.shutdown()


def exit_replay(trade, ticks, half_share=.5, fixed_target=False):
    row = trade["journal"]
    sign = 1 if row["side"] == "LONG" else -1
    entry = next(d["price"] for d in trade["deals"] if d["entry"] == 0)
    # USD-quoted focus trades have exactly $10/pip/lot; derive observed cash sensitivity for other crosses.
    close = next(d for d in reversed(trade["deals"]) if d["entry"] == 1)
    actual_delta = sign * (close["price"] - entry)
    if abs(actual_delta) < 1e-12:
        return None
    value_per_price = close["profit"] / actual_delta
    fees = sum(d["commission"] + d["swap"] + d["fee"] for d in trade["deals"])
    start = trade["open_msc_raw"]
    end = min(start + 4 * 3600 * 1000, int(ticks["time_msc"][-1]))
    i0 = np.searchsorted(ticks["time_msc"], start, side="left")
    i1 = np.searchsorted(ticks["time_msc"], end, side="right")
    quotes = ticks["bid" if sign == 1 else "ask"][i0:i1]
    times = ticks["time_msc"][i0:i1]
    if not len(quotes):
        return None
    half_stop = row["entry"] - sign * abs(row["entry"] - row["stop"]) * half_share
    target, atr = row["target"], row["atr"]
    best, trailing, stop = entry, False, row["stop"]
    reason, price, when = "CAPTURE_END_MARK", float(quotes[-1]), int(times[-1])
    for timestamp, quote in zip(times, quotes):
        if sign * (quote - stop) <= 0:
            reason, price, when = "TRAIL" if trailing else "STOP", float(quote), int(timestamp)
            break
        best = max(best, quote) if sign == 1 else min(best, quote)
        if sign * (quote - target) >= 0:
            if fixed_target:
                reason, price, when = "TARGET", float(quote), int(timestamp)
                break
            trailing = True
        if trailing:
            candidate = best - sign * atr
            floor = row["entry"] - sign * .5 * atr
            wanted = max(candidate, floor) if sign == 1 else min(candidate, floor)
            stop = max(stop, wanted) if sign == 1 else min(stop, wanted)
        elif half_share < 1 and sign * (quote - half_stop) <= 0:
            reason, price, when = "EARLY_STOP", float(quote), int(timestamp)
            break
    pnl = sign * (price - entry) * value_per_price + fees
    movement = sign * (quotes - entry)
    before_actual_close = movement[times <= trade["close_msc_raw"]]
    return {"net_usd": float(pnl), "reason": reason, "closed_raw_msc": when,
        "max_favourable_pips": float(max(0, movement.max()) / row["pip"]),
        "max_adverse_pips": float(max(0, -movement.min()) / row["pip"]),
        "before_actual_close_mfe_pips": float(max(0, before_actual_close.max()) / row["pip"]) if len(before_actual_close) else None,
        "before_actual_close_mae_pips": float(max(0, -before_actual_close.min()) / row["pip"]) if len(before_actual_close) else None}


def replay():
    snapshot = json.loads((OUT / "capture.json").read_text(encoding="utf-8"))
    policies = {"half_50_trail": (.5, False), "half_75_trail": (.75, False),
                "full_stop_trail": (1., False), "fixed_target_full_stop": (1., True)}
    rows = []
    for trade in snapshot["trades"]:
        row = trade["journal"]
        with np.load(OUT / f"{row['symbol']}_ticks.npz") as data:
            ticks = {key: data[key] for key in data.files}
        for name, (share, fixed) in policies.items():
            result = exit_replay(trade, ticks, share, fixed)
            if result:
                rows.append({"ticket": row["ticket"], "symbol": row["symbol"], "focus": trade["focus"],
                    "policy": name, "actual_net": trade["actual_net"], **result})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "tick_replay.csv", index=False)
    print(frame[frame.focus][["ticket", "symbol", "policy", "actual_net", "net_usd", "reason"]].to_string(index=False))
    print("ALL CLOSED TODAY (exploratory, same entries, not portfolio):")
    print(frame.groupby("policy").agg(n=("net_usd", "size"), net=("net_usd", "sum"),
        marked_open=("reason", lambda s: (s == "CAPTURE_END_MARK").sum())).to_string())


def history_worker(config):
    import research_bot_backtest as bt
    label, share, fixed, *options = config
    keep_ml_swap = bool(options and options[0])
    bt.EARLY_EXIT_SHARE = share
    variant = bt.Variant(label, half_stop=share < 1, trailing=not fixed, ml_exit=not fixed or keep_ml_swap,
        swap=not fixed or keep_ml_swap, floor_atr=.5, basket_pct=.03, basket_trail=True, basket_giveback=.3)
    market = bt.load_market(parallel=False)
    result = bt.run(variant, market, bt.high_impact_events(), bt.load_plan_stats(), fixed_risk=True)
    # Same summary utility as existing research; no outputs in its source/cache are rewritten.
    summary = bt.summarize(result)
    summary["variant"] = label
    summary["half_share"] = share
    return summary


def history():
    configs = [("H50 current", .5, False), ("H75 later early stop", .75, False),
               ("H100 full stop", 1., False), ("TP fixed target full stop", 1., True)]
    with Pool(4) as pool:
        summaries = pool.map(history_worker, configs)
    write("historical_results.json", summaries)
    for summary in summaries:
        print(json.dumps({k: summary.get(k) for k in ("variant", "trades", "mean_r", "mean_r_choose", "mean_r_confirm", "win_rate")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "replay", "history", "half-target"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "half-target":
        result = history_worker(("TP fixed target current half-stop", .5, True, True))
        write("historical_half_target.json", result)
        print(json.dumps({k: result.get(k) for k in ("variant", "trades", "mean_r", "mean_r_choose", "mean_r_confirm", "win_rate")}, ensure_ascii=False))
    else:
        {"capture": capture, "replay": replay, "history": history}[args.stage]()


if __name__ == "__main__":
    main()
