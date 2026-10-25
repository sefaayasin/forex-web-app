"""Do the opportunity feed's 15M + 5M LONG/SHORT cards predict the next move?

Rules are in research/feed_calls/protocol.json and were written before any outcome was computed.
Offline research; never changes the live app. Stages (run in order):

    python -X utf8 research_feed_calls.py backtest   # Dukascopy archive, 2025-09-11 .. 2026-09-11
    python -X utf8 research_feed_calls.py yahoo      # the app's live source, last ~8 weeks
    python -X utf8 research_feed_calls.py evaluate

Scan rows are written per pair to research/feed_calls/rows (git-ignored); finished pairs are skipped on rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from feed_replay import ENTRY_TFS, ReplayFeed, load_scan_functions, scan_at
from forex_config import get_pip_size
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "feed_calls"
ROWS = OUTPUT / "rows"
YAHOO_CACHE = OUTPUT / "yahoo_cache"
BACKTEST_START, BACKTEST_END = pd.Timestamp("2025-09-11", tz="UTC"), pd.Timestamp("2026-09-11", tz="UTC")
YAHOO_END = pd.Timestamp("2026-09-25 21:00", tz="UTC")
HORIZONS = {"15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "24h": pd.Timedelta(hours=24)}
WARMUP = pd.Timedelta(days=70)
PAIRS = sorted(p.stem for p in (ROOT / "data" / "historical_15m").glob("*.csv") if p.stem != "EURZAR")
WORKERS = 14
BOOTSTRAPS = 2000


def read_archive(folder: str, pair: str, start: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / folder / f"{pair}.csv")
    df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
    df.columns = [c.capitalize() for c in df.columns]
    df = df.sort_index()
    return df[df.index >= start][["Open", "High", "Low", "Close", "Volume"]]


def decision_times(bars_5m: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """15-minute boundaries inside [start, end) at which a 5-minute bar opens (market open)."""
    times = bars_5m.index[(bars_5m.index >= start) & (bars_5m.index < end)]
    return times[(times.minute % 15) == 0]


def signed_moves(bars_5m: pd.DataFrame, when: pd.Timestamp, side: int, pip: float) -> dict:
    opens = bars_5m["Open"]
    entry = float(opens.iloc[bars_5m.index.searchsorted(when)])
    out = {"entry": entry}
    for name, horizon in HORIZONS.items():
        pos = bars_5m.index.searchsorted(when + horizon, side="right") - 1
        out[f"move_{name}"] = side * (float(opens.iloc[pos]) - entry) / pip
    return out


def replay_pair(pair: str, bars_5m: pd.DataFrame, bars_15m: pd.DataFrame, bars_1h: pd.DataFrame,
                start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    feed = ReplayFeed(bars_5m, bars_15m, bars_1h)
    functions = load_scan_functions(feed.fetch_ohlc)
    pip = get_pip_size(pair)
    rows = []
    for when in decision_times(bars_5m, start, end):
        verdicts = scan_at(functions, feed, f"{pair}=X", when)
        for tf, v in verdicts.items():
            if not v["direction_allowed"]:
                continue
            direction = 1 if "Alım" in v["bias"] else -1
            rows.append({"pair": pair, "tf": tf, "time": when, "side": v["side"], "direction": direction,
                         "gate_passed": v["gate_passed"], "score": v["score"],
                         **signed_moves(bars_5m, when, direction, pip)})
    return pd.DataFrame(rows)


def backtest_worker(pair: str) -> str:
    target = ROWS / f"backtest_{pair}.csv.gz"
    if target.exists():
        return f"{pair}: exists"
    start = BACKTEST_START - WARMUP
    bars_5m, bars_15m = read_archive("historical_5m", pair, start), read_archive("historical_15m", pair, start)
    bars_1h = read_archive("historical_1h_from_15m", pair, start)
    frame = replay_pair(pair, bars_5m, bars_15m, bars_1h, BACKTEST_START, BACKTEST_END)
    frame.to_csv(target, index=False, compression="gzip")
    return f"{pair}: {len(frame)} rows, {int((frame.side != 0).sum()) if len(frame) else 0} card-scans"


def fetch_yahoo(pair: str, interval: str, period: str) -> pd.DataFrame:
    """Yahoo bars cached on first download so the replay is reproducible."""
    cached = YAHOO_CACHE / f"{pair}_{interval}.csv"
    if not cached.exists():
        import time
        import yfinance as yf
        for attempt in range(3):
            raw = yf.download(f"{pair}=X", interval=interval, period=period, progress=False, auto_adjust=False, threads=False)
            if raw is not None and not raw.empty:
                break
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"{pair} {interval}: Yahoo returned no data")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        raw.index = raw.index.tz_convert("UTC") if raw.index.tz is not None else raw.index.tz_localize("UTC")
        raw.to_csv(cached, index_label="time")
    frame = pd.read_csv(cached, index_col="time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def yahoo_worker(pair: str) -> str:
    target = ROWS / f"yahoo_{pair}.csv.gz"
    if target.exists():
        return f"{pair}: exists"
    bars_5m, bars_15m, bars_1h = fetch_yahoo(pair, "5m", "60d"), fetch_yahoo(pair, "15m", "60d"), fetch_yahoo(pair, "60m", "180d")
    start = bars_5m.index[0] + pd.Timedelta(days=5)
    frame = replay_pair(pair, bars_5m, bars_15m, bars_1h, start, YAHOO_END)
    frame.to_csv(target, index=False, compression="gzip")
    return f"{pair}: {len(frame)} rows from {start.date()}, {int((frame.side != 0).sum()) if len(frame) else 0} card-scans"


def safe(worker, pair: str) -> str:
    try:
        return worker(pair)
    except Exception as exc:  # noqa: BLE001 - one bad pair must not stop the others
        return f"{pair}: FAILED {type(exc).__name__}: {exc}"


def safe_backtest(pair: str) -> str:
    return safe(backtest_worker, pair)


def safe_yahoo(pair: str) -> str:
    return safe(yahoo_worker, pair)


def run_stage(worker, pairs: list[str], workers: int) -> None:
    ROWS.mkdir(parents=True, exist_ok=True)
    YAHOO_CACHE.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool:
        for message in pool.imap_unordered(worker, pairs):
            print(message, flush=True)


def mark_new_calls(rows: pd.DataFrame) -> pd.DataFrame:
    """A card is new if the same pair/timeframe did not show the same side at the previous 15-minute scan."""
    rows = rows.sort_values(["pair", "tf", "time"]).copy()
    previous_time = rows.groupby(["pair", "tf"]).time.shift(1)
    previous_side = rows.groupby(["pair", "tf"]).side.shift(1)
    continued = (previous_time == rows.time - pd.Timedelta(minutes=15)) & (previous_side == rows.side)
    rows["new_call"] = (rows.side != 0) & ~continued
    return rows


def add_costs(rows: pd.DataFrame) -> pd.DataFrame:
    profile = load_spread_profile()
    ny_hour = rows.time.dt.tz_convert("America/New_York").dt.hour
    spread = [measured_spread_pips(profile, p, h) for p, h in zip(rows.pair, ny_hour)]
    rows["cost_pips"] = pd.Series(spread, index=rows.index, dtype=float).fillna(1.5) + ECN_COMMISSION_PIPS
    for name in HORIZONS:
        rows[f"net_{name}"] = rows[f"move_{name}"] - rows.cost_pips
    return rows


def day_bootstrap(values: pd.Series, days: pd.Series, seed: int = 42) -> tuple[float, float]:
    """95% interval of the mean, resampling whole calendar days."""
    sums = values.groupby(days).sum()
    counts = values.groupby(days).count()
    s, c = sums.to_numpy(), counts.to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTSTRAPS):
        w = np.bincount(rng.integers(0, len(s), len(s)), minlength=len(s))
        draws.append((w @ s) / (w @ c))
    low, high = np.quantile(draws, [.025, .975])
    return float(low), float(high)


def summarize(calls: pd.DataFrame) -> dict:
    out = {"calls": int(len(calls)), "long": int((calls.side == 1).sum()), "short": int((calls.side == -1).sum()),
           "days": int(calls.time.dt.floor("D").nunique()), "median_cost_pips": float(calls.cost_pips.median()) if len(calls) else None}
    days = calls.time.dt.floor("D")
    for name in HORIZONS:
        if not len(calls):
            continue
        hit = (calls[f"move_{name}"] > 0).astype(float)
        out[name] = {
            "hit_rate": float(hit.mean()), "hit_rate_ci": day_bootstrap(hit, days),
            "mean_move_pips": float(calls[f"move_{name}"].mean()),
            "mean_net_pips": float(calls[f"net_{name}"].mean()), "mean_net_ci": day_bootstrap(calls[f"net_{name}"], days),
        }
    return out


def evaluate() -> None:
    results = {}
    for source in ("backtest", "yahoo"):
        files = sorted(ROWS.glob(f"{source}_*.csv.gz"))
        if not files:
            continue
        rows = pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in files], ignore_index=True)
        rows["time"] = pd.to_datetime(rows.time, utc=True)
        rows = add_costs(mark_new_calls(rows))
        block = {"pairs": int(rows.pair.nunique()), "period": [str(rows.time.min()), str(rows.time.max())],
                 "scans_with_direction": int(len(rows)), "card_scans": int((rows.side != 0).sum())}
        for tf in ENTRY_TFS:
            part = rows[rows.tf == tf]
            new = part[part.new_call]
            first_blocked = part[(part.side == 0) & ~part.gate_passed]
            first_blocked = first_blocked[first_blocked.groupby("pair").time.diff().ne(pd.Timedelta(minutes=15))]
            block[tf] = {"new_cards": summarize(new),
                         "info_direction_but_gate_blocked": summarize(first_blocked.assign(side=first_blocked.direction)),
                         "info_new_cards_by_score": {
                             label: summarize(new[mask]) for label, mask in
                             (("score_below_45", new.score < 45), ("score_45_60", (new.score >= 45) & (new.score < 60)),
                              ("score_60_plus", new.score >= 60))}}
        results[source] = block

    def passes(block: dict, tf: str) -> bool:
        cards = block.get(tf, {}).get("new_cards", {})
        return all(cards.get(h, {}).get("mean_net_ci", [0, 0])[0] > 0 for h in ("1h", "4h"))

    def not_contradicted(block: dict, tf: str) -> bool:
        cards = block.get(tf, {}).get("new_cards", {})
        return all(cards.get(h, {}).get("mean_net_pips", -1) >= 0 for h in ("1h", "4h"))

    verdict = {tf: {"backtest_passes": passes(results.get("backtest", {}), tf),
                    "yahoo_not_contradicting": not_contradicted(results.get("yahoo", {}), tf)} for tf in ENTRY_TFS}
    results["verdict"] = {"per_entry_tf": verdict,
                          "cards_useful": any(v["backtest_passes"] and v["yahoo_not_contradicting"] for v in verdict.values())}
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for source in ("backtest", "yahoo"):
        if source not in results:
            continue
        print(f"=== {source}: {results[source]['period']} | pairs {results[source]['pairs']} | card scans {results[source]['card_scans']}")
        for tf in ENTRY_TFS:
            cards = results[source][tf]["new_cards"]
            line = {h: (round(cards[h]["hit_rate"], 3), round(cards[h]["mean_move_pips"], 2), round(cards[h]["mean_net_pips"], 2),
                        [round(x, 2) for x in cards[h]["mean_net_ci"]]) for h in HORIZONS if h in cards}
            print(f"  {tf}: new cards {cards['calls']} (L {cards['long']} / S {cards['short']}), median cost {cards['median_cost_pips']}")
            for h, v in line.items():
                print(f"     {h}: hit {v[0]}, move {v[1]} pips, net {v[2]} pips, net CI {v[3]}")
    print(json.dumps(results["verdict"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["backtest", "yahoo", "evaluate"])
    parser.add_argument("--pairs", nargs="+", default=PAIRS)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.stage == "backtest":
        run_stage(safe_backtest, args.pairs, args.workers)
    elif args.stage == "yahoo":
        run_stage(safe_yahoo, args.pairs, args.workers)
    else:
        evaluate()
