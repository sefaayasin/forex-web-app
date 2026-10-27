"""Read MT5 today's completed demo positions; compare winners and losers offline.

capture performs only history/quote reads. analyze never connects to MT5.
Output goes to research/daily_review_20261027, preserving prior loss snapshots.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from research_recent_losses import exit_replay

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research/daily_review_20261027"
DAY = "2026-10-27"
MAGIC = 26102026


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
            raise RuntimeError("Expected existing demo account")
        now = pd.Timestamp.now(tz="UTC")
        midnight = pd.Timestamp(DAY, tz="Europe/Istanbul").tz_convert("UTC")
        offset = server_offset(mt5, "EURUSD")
        deals = mt5.history_deals_get((midnight + offset).to_pydatetime(), (now + offset).to_pydatetime())
        if deals is None:
            raise RuntimeError(mt5.last_error())
        positions = mt5.positions_get()
        if positions is None:
            raise RuntimeError(mt5.last_error())
        opened_ids = {p.ticket for p in positions}
        conn = sqlite3.connect((ROOT / "data/bot_journal.db").resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        journal = {row["ticket"]: dict(row) for row in conn.execute("SELECT * FROM trades")}
        conn.close()
        fields = ("ticket", "time_msc", "entry", "type", "magic", "position_id", "volume", "price", "profit", "commission", "swap", "fee", "reason", "symbol")
        exited = sorted({d.position_id for d in deals if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY)})
        records, all_closed = [], []
        for position_id in exited:
            position_deals = mt5.history_deals_get(position=position_id)
            if position_deals is None:
                raise RuntimeError(mt5.last_error())
            entries = [d for d in position_deals if d.entry == mt5.DEAL_ENTRY_IN]
            exits = [d for d in position_deals if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY)]
            if not entries or not exits or position_id in opened_ids:
                continue
            last = max(exits, key=lambda d: d.time_msc)
            closed_utc = pd.to_datetime(last.time_msc, unit="ms", utc=True) - offset
            if not midnight <= closed_utc <= now:
                continue
            net = sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0) for d in position_deals)
            row = journal.get(position_id)
            group = "sized_bot" if row and row.get("sizing_balance") is not None else (
                "legacy_bot" if any(d.magic == MAGIC for d in position_deals) else "other")
            first = min(entries, key=lambda d: d.time_msc)
            cash = {"ticket": position_id, "symbol": last.symbol, "group": group,
                "opened_local": str((pd.to_datetime(first.time_msc, unit="ms", utc=True) - offset).tz_convert("Europe/Istanbul")),
                "closed_local": str(closed_utc.tz_convert("Europe/Istanbul")), "actual_net": net,
                "gross_profit": sum(d.profit for d in position_deals),
                "fees_swap": sum(d.commission + d.swap + getattr(d, "fee", 0) for d in position_deals)}
            all_closed.append(cash)
            if group == "sized_bot":
                records.append({"journal": row, "focus": False,
                    "deals": [{key: getattr(d, key, 0) for key in fields} for d in position_deals],
                    "open_msc_raw": first.time_msc, "close_msc_raw": last.time_msc,
                    "actual_net": net, "cash": cash})
        windows = {}
        for trade in records:
            symbol = trade["journal"]["symbol"]
            windows[symbol] = min(windows.get(symbol, trade["open_msc_raw"]), trade["open_msc_raw"])
        tick_quality = {}
        for symbol, start in windows.items():
            raw = mt5.copy_ticks_range(symbol, pd.to_datetime(start, unit="ms", utc=True).to_pydatetime(),
                (now + offset).to_pydatetime(), mt5.COPY_TICKS_ALL)
            if raw is None or len(raw) == 0:
                raise RuntimeError(f"No ticks: {symbol}: {mt5.last_error()}")
            ticks = pd.DataFrame(raw)[["time_msc", "bid", "ask"]].sort_values("time_msc", kind="stable")
            ticks = ticks[(ticks.bid > 0) & (ticks.ask >= ticks.bid)]
            np.savez_compressed(OUT / f"{symbol}_ticks.npz", **{key: ticks[key].to_numpy() for key in ticks})
            tick_quality[symbol] = {"n": len(ticks), "max_gap_seconds": float(ticks.time_msc.diff().max() / 1000)}
            print(symbol, len(ticks), flush=True)
        write("snapshot.json", {"captured_utc": str(now), "captured_local": str(now.tz_convert("Europe/Istanbul")),
            "day_local": DAY, "observed_server_offset_seconds": offset.total_seconds(),
            "all_closed": all_closed, "trades": records, "tick_quality": tick_quality,
            "open_sized_bot": [{"ticket": p.ticket, "symbol": p.symbol, "volume": p.volume,
                "floating_gross": p.profit, "exit_mode": journal.get(p.ticket, {}).get("exit_mode")}
                for p in positions if p.ticket in journal and journal[p.ticket].get("sizing_balance") is not None],
            "day_deal_cash_flow": sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0) for d in deals if d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL))})
        print("Sized closed", len(records), "all closed", len(all_closed), "snapshot", str(now))
    finally:
        mt5.shutdown()


def analyze():
    snapshot = json.loads((OUT / "snapshot.json").read_text(encoding="utf-8"))
    ledger, comparisons = [], []
    policies = {"half_stop_trailing": (.5, False), "later_75_stop": (.75, False),
                "full_stop_trailing": (1., False), "fixed_target_half_stop": (.5, True),
                "fixed_target_full_stop": (1., True)}
    for trade in snapshot["trades"]:
        row = trade["journal"]
        with np.load(OUT / f"{row['symbol']}_ticks.npz") as data:
            ticks = {key: data[key] for key in data.files}
        before = (ticks["time_msc"] >= trade["open_msc_raw"]) & (ticks["time_msc"] <= trade["close_msc_raw"])
        sign = 1 if row["side"] == "LONG" else -1
        entry = next(d["price"] for d in trade["deals"] if d["entry"] == 0)
        exit_deal = next(d for d in reversed(trade["deals"]) if d["entry"] == 1)
        delta = sign * (exit_deal["price"] - entry)
        cash_per_price = exit_deal["profit"] / delta if abs(delta) > 1e-12 else None
        fees = sum(d["commission"] + d["swap"] + d["fee"] for d in trade["deals"])
        quotes = ticks["bid" if sign == 1 else "ask"][before]
        net_path = sign * (quotes - entry) * cash_per_price + fees if cash_per_price else None
        best_net = float(net_path.max()) if net_path is not None and len(net_path) else None
        worst_net = float(net_path.min()) if net_path is not None and len(net_path) else None
        target_seen = bool(np.any(sign * (quotes - row["target"]) >= 0))
        touches = np.flatnonzero(sign * (quotes - row["target"]) >= 0)
        # Change only target-taking on the real path before its actual close. Keep every other actual exit.
        isolated_target_net = float(net_path[touches[0]]) if len(touches) and net_path is not None else trade["actual_net"]
        stop_pips = abs(row["entry"] - row["stop"]) / row["pip"]
        record = {**trade["cash"], "side": row["side"], "lot": row["lot"],
            "outcome": row["outcome"], "exit_mode": row.get("exit_mode", "trailing"),
            "ml_confidence": max(row["probability"], 1-row["probability"]), "historical_target_hit": row["hit"],
            "cost_share": row["cost_pips"] / (stop_pips + row["cost_pips"]),
            "best_net_before_close": best_net, "worst_net_before_close": worst_net,
            "target_seen_before_close": target_seen,
            "target_or_actual_exit_net": isolated_target_net,
            "duration_minutes": (trade["close_msc_raw"] - trade["open_msc_raw"]) / 60000}
        ledger.append(record)
        for name, (share, fixed) in policies.items():
            result = exit_replay(trade, ticks, share, fixed)
            if result:
                comparisons.append({"ticket": row["ticket"], "symbol": row["symbol"], "policy": name,
                    "actual_net": trade["actual_net"], **result})
    frame = pd.DataFrame(ledger).sort_values("closed_local")
    if frame.ticket.duplicated().any() or not np.allclose(frame.actual_net, frame.gross_profit + frame.fees_swap):
        raise AssertionError("MT5 position/commission reconciliation failed")
    frame.to_csv(OUT / "ledger.csv", index=False)
    paths = pd.DataFrame(comparisons)
    paths.to_csv(OUT / "exit_counterfactuals.csv", index=False)
    filters = []
    for name, mask in [("all", pd.Series(True, index=frame.index))] + [
        (f"ml_at_least_{value}", frame.ml_confidence >= value) for value in (.52, .55)] + [
        (f"cost_share_at_most_{value}", frame.cost_share <= value) for value in (.10, .15)]:
        kept = frame[mask]
        filters.append({"filter": name, "kept": len(kept), "kept_winners": int((kept.actual_net > 0).sum()),
            "kept_losers": int((kept.actual_net < 0).sum()), "kept_actual_net": float(kept.actual_net.sum()),
            "removed_winner_profit": float(frame.loc[~mask & (frame.actual_net > 0), "actual_net"].sum()),
            "removed_loss": float(frame.loc[~mask & (frame.actual_net < 0), "actual_net"].sum())})
    summary = {"snapshot_local": snapshot["captured_local"], "closed_sized_bot": len(frame),
        "winners": int((frame.actual_net > 0).sum()), "losers": int((frame.actual_net < 0).sum()),
        "gross_winning_net": float(frame.loc[frame.actual_net > 0, "actual_net"].sum()),
        "gross_losing_net": float(frame.loc[frame.actual_net < 0, "actual_net"].sum()),
        "net": float(frame.actual_net.sum()), "fees_swap": float(frame.fees_swap.sum()),
        "target_only_same_entries_net": float(frame.target_or_actual_exit_net.sum()),
        "losers_never_net_positive": int(((frame.actual_net < 0) & (frame.best_net_before_close <= 0)).sum()),
        "losers_reached_target": int(((frame.actual_net < 0) & frame.target_seen_before_close).sum()),
        "open_sized_bot": snapshot["open_sized_bot"], "filters": filters,
        "closed_by_account_group": pd.DataFrame(snapshot["all_closed"]).groupby("group").actual_net.agg(["size", "sum"]).reset_index().to_dict("records")}
    write("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(frame[["symbol", "closed_local", "actual_net", "outcome", "ml_confidence", "best_net_before_close", "target_seen_before_close"]].to_string(index=False))
    print(paths.groupby("policy").agg(trades=("ticket", "size"), net=("net_usd", "sum"),
        unfinished=("reason", lambda s: (s == "CAPTURE_END_MARK").sum())).to_string())


def cost_worker(config):
    import research_bot_backtest as bt
    policy, cost_cap = config
    modern = policy == "new_fixed_55_risk1"
    bt.RISK_PCT = 1.0 if modern else 2.5
    variant = bt.Variant(f"{policy}_cost{cost_cap}", trailing=not modern, half_stop=True,
        floor_atr=.5, min_conf=.55 if modern else 0., max_cost_share=cost_cap,
        basket_pct=bt.RISK_PCT / 100 * 1.2, basket_trail=True, basket_giveback=.3)
    result = bt.run(variant, bt.load_market(parallel=False), bt.high_impact_events(),
        bt.load_plan_stats(), fixed_risk=True)
    summary = bt.summarize(result)
    summary.update(policy=policy, max_cost_share=cost_cap, sizing_risk_pct=bt.RISK_PCT)
    return summary


def history():
    configs = [(policy, cap) for policy in ("old_trailing_risk2.5", "new_fixed_55_risk1") for cap in (.15, .10)]
    with Pool(4) as pool:
        results = pool.map(cost_worker, configs)
    write("cost_cap_history.json", results)
    for result in results:
        print(json.dumps({key: result.get(key) for key in ("policy", "max_cost_share", "trades",
            "trades_per_week", "mean_r_choose", "mean_r_confirm", "net_usd")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "analyze", "history"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"capture": capture, "analyze": analyze, "history": history}[args.stage]()


if __name__ == "__main__":
    main()
