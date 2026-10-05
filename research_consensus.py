"""When every direction view on the Özet tab agrees, does entering that way beat the cost?

Rules are in research/consensus/protocol.json and were written before any outcome was computed.
Offline research; never changes the live app. Stages (run in order):

    python -X utf8 research_consensus.py build      # replay the app's views hourly, 2023-01-01 .. archive end
    python -X utf8 research_consensus.py evaluate

Per pair-year moment rows go to research/consensus/rows (git-ignored); finished chunks are skipped on rerun.
"""
from __future__ import annotations

import argparse
import ast
import json
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from feed_replay import APP, ReplayFeed
from forex_config import get_pip_size
from forex_costs import ECN_COMMISSION_PIPS, load_spread_profile, measured_spread_pips
from forex_ml_live import build_research_prediction, load_research_model
from forex_ml_tournament import build_features

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "consensus"
ROWS = OUTPUT / "rows"
PAIRS = sorted(p.name.split("_")[0] for p in (ROOT / "data" / "ml" / "tournament").glob("*_direction_research.joblib"))
START, ARCHIVE_END = pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2026-09-11 21:00", tz="UTC")
YEARS = [2023, 2024, 2025, 2026]
WARMUP = pd.Timedelta(days=70)  # the app's longest window is 60 days of hourly bars for 4H
HORIZONS = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "24h": pd.Timedelta(hours=24)}
TESTED_HORIZONS = ("1h", "4h")
SUB_PERIODS = {"2023": (2023, 2023), "2024": (2024, 2024), "2025-2026": (2025, 2026)}
RADAR_CANDIDATE = 42.0  # forex_decision_core.classify_opportunity_readiness candidate_threshold
TF_COLUMNS = {"4 Saat": "tf4h", "1 Saat": "tf1h", "15 Dakika": "tf15", "5 Dakika": "tf5"}
CHECK_SAMPLES = 20
BOOTSTRAPS = 2000
CI_LEVEL = 0.975  # two-sided, Bonferroni 0.05 / 2 horizons
WORKERS = 14
VIEWS = ("radar", "firsatlar", "pariteler", "ml")


def load_app_functions(fetch_ohlc: Callable) -> dict:
    """analyse_symbol, global_bias and build_intraday_opportunity compiled from the app, fetch_ohlc swapped."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    constants = {t.id: n for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                 if isinstance(t, ast.Name) and t.id.isupper()}
    wanted = {"analyse_symbol", "global_bias", "build_intraday_opportunity"}
    while True:
        names = {n.id for name in wanted for n in ast.walk(definitions[name]) if isinstance(n, ast.Name)}
        extra = (names & definitions.keys()) - {"fetch_ohlc"} - wanted
        if not extra:
            break
        wanted |= extra
    used = {n.id for name in wanted for n in ast.walk(definitions[name]) if isinstance(n, ast.Name)} & constants.keys()
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Try))]
    body = imports + [constants[c] for c in sorted(used)] + [n for n in tree.body if getattr(n, "name", None) in wanted]
    namespace: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(APP), "exec"), namespace)
    namespace["fetch_ohlc"] = fetch_ohlc
    return namespace


def read_archive(folder: str, pair: str, start: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / folder / f"{pair}.csv")
    df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
    df.columns = [c.capitalize() for c in df.columns]
    df = df.sort_index()
    return df[df.index >= start][["Open", "High", "Low", "Close", "Volume"]]


def ml_probabilities(pair: str, hourly: pd.DataFrame) -> pd.Series:
    """probability_up of the frozen direction model per hourly bar (indexed by bar open time); NaN if not scorable."""
    bundle = load_research_model(pair, "direction")
    features, _ = build_features(hourly[["Open", "High", "Low", "Close"]], pair)
    rows = features[bundle["columns"]]
    rows = rows[rows.index >= START - pd.Timedelta(days=7)]
    usable = rows.notna().all(axis=1)
    out = pd.Series(np.nan, index=rows.index)
    out[usable] = bundle["model"].predict_proba(rows[usable])[:, 1]
    return out


def ml_at(probs: pd.Series, when: pd.Timestamp) -> float:
    """The live app scores the last hourly bar closed by `when`."""
    pos = probs.index.searchsorted(when - pd.Timedelta(hours=1), side="right") - 1
    return float(probs.iloc[pos]) if pos >= 0 else np.nan


def views_at(functions: dict, feed: ReplayFeed, pair: str, when: pd.Timestamp, prob_up: float) -> dict | None:
    feed.now = when
    summary, _ = functions["analyse_symbol"](pair)
    if (summary["Bias"] == "Veri yok").any():
        return None
    row = {TF_COLUMNS[tf]: label for tf, label in zip(summary["Zaman Dilimi"], summary["Bias"])}
    row.update({f"{TF_COLUMNS[tf]}_score": score for tf, score in zip(summary["Zaman Dilimi"], summary["Skor"])})
    row["g15"] = functions["global_bias"](summary, "15 Dakika")[0]
    row["g5"] = functions["global_bias"](summary, "5 Dakika")[0]
    prediction = ({"status": "ready", "task": "direction", "probability_up": prob_up}
                  if np.isfinite(prob_up) else {"status": "insufficient_data"})
    radar = functions["build_intraday_opportunity"](
        symbol=pair, summary=summary, current_price=None, account_size=10_000.0, risk_pct=1.0,
        pip_value_per_lot=10.0, total_cost_pips=0.0, target_usd=0.0, research_prediction=prediction)
    row.update({"radar_side": radar["side"], "radar_score": round(float(radar["confidence"]), 6), "prob_up": prob_up})
    return row


def build_chunk(task: tuple[str, int]) -> str:
    pair, year = task
    target = ROWS / f"{pair}_{year}.csv.gz"
    if target.exists():
        return f"{pair} {year}: exists"
    lo = max(START, pd.Timestamp(f"{year}-01-01", tz="UTC"))
    hi = min(ARCHIVE_END, pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
    bars5 = read_archive("historical_5m", pair, lo - WARMUP)
    bars15 = read_archive("historical_15m", pair, lo - WARMUP)
    hourly = read_archive("historical_1h_from_15m", pair, pd.Timestamp("2000-01-01", tz="UTC"))
    probs = ml_probabilities(pair, hourly)
    hourly_window = hourly[hourly.index >= lo - WARMUP]

    feed = ReplayFeed(bars5, bars15, hourly_window)
    functions = load_app_functions(feed.fetch_ohlc)
    times = bars5.index[(bars5.index >= lo) & (bars5.index < hi) & (bars5.index.minute == 0)]
    records = []
    for when in times:
        row = views_at(functions, feed, pair, when, ml_at(probs, when))
        if row is not None:
            records.append({"time": when, **row})
    frame = pd.DataFrame(records).set_index("time")
    check_chunk(pair, frame, bars5, bars15, hourly, hourly_window)

    pip = get_pip_size(pair)
    opens = bars5["Open"]
    entry = opens.reindex(frame.index).to_numpy()
    profile = load_spread_profile()
    ny_hour = frame.index.tz_convert("America/New_York").hour
    frame["cost"] = pd.Series([measured_spread_pips(profile, pair, h) for h in ny_hour], dtype=float).fillna(1.5).to_numpy() \
        + ECN_COMMISSION_PIPS
    for name, horizon in HORIZONS.items():
        exit_pos = opens.index.searchsorted(frame.index + horizon, side="right") - 1
        up = (opens.to_numpy()[exit_pos] - entry) / pip
        complete = (frame.index + horizon) <= opens.index[-1]
        frame[f"up_{name}"] = np.where(complete, up, np.nan)
    frame.insert(0, "pair", pair)
    frame.to_csv(target, compression="gzip")
    return f"{pair} {year}: {len(frame):,} moments, check passed on {min(CHECK_SAMPLES, len(frame))}"


def check_chunk(pair, frame, bars5, bars15, hourly, hourly_window) -> None:
    """Recompute sampled moments from scratch (fresh feed, fresh compile, live ML path); stop on any mismatch."""
    rng = np.random.default_rng(zlib.crc32(f"{pair} {frame.index[0]}".encode()))
    sample = frame.index[np.sort(rng.choice(len(frame), min(CHECK_SAMPLES, len(frame)), replace=False))]
    feed = ReplayFeed(bars5, bars15, hourly_window)
    functions = load_app_functions(feed.fetch_ohlc)
    for when in sample:
        live = build_research_prediction(pair, bars=hourly[hourly.index < when], now=when)
        live_prob = float(live["probability_up"]) if live["status"] == "ready" else np.nan
        stored = frame.loc[when]
        if not (np.isnan(live_prob) and np.isnan(stored.prob_up)) and not abs(live_prob - stored.prob_up) <= 1e-9:
            raise AssertionError(f"{pair} {when}: ML {stored.prob_up} vs live path {live_prob}")
        again = views_at(functions, feed, pair, when, float(stored.prob_up))
        for key, value in again.items():
            same = (np.isnan(value) and np.isnan(stored[key])) if isinstance(value, float) and np.isnan(value) else value == stored[key]
            if not same:
                raise AssertionError(f"{pair} {when}: {key} {stored[key]!r} vs recomputed {value!r}")


def safe_build(task: tuple[str, int]) -> str:
    try:
        return build_chunk(task)
    except Exception as exc:  # noqa: BLE001 - one bad chunk must not stop the others; a failed check is reported
        return f"{task}: FAILED {type(exc).__name__}: {exc}"


def label_side(label: pd.Series) -> pd.Series:
    text = label.astype(str)
    return pd.Series(np.where(text.str.contains("Alım"), 1, np.where(text.str.contains("Satış"), -1, 0)), index=label.index)


def view_sides(rows: pd.DataFrame) -> pd.DataFrame:
    """Each Özet view's side (+1 LONG, -1 SHORT, 0 none) per moment, as the app's tab derives it."""
    tf = pd.concat([label_side(rows[c]) for c in TF_COLUMNS.values()], axis=1)
    pariteler = np.where((tf == 1).all(axis=1), 1, np.where((tf == -1).all(axis=1), -1, 0))
    strong_long = (rows.g15 == "Güçlü Alım Yönlü") | (rows.g5 == "Güçlü Alım Yönlü")
    strong_short = (rows.g15 == "Güçlü Satış Yönlü") | (rows.g5 == "Güçlü Satış Yönlü")
    firsatlar = np.where(strong_long & ~strong_short, 1, np.where(strong_short & ~strong_long, -1, 0))
    radar_dir = rows.radar_side.map({"LONG": 1, "SHORT": -1}).fillna(0)
    radar = np.where(rows.radar_score >= RADAR_CANDIDATE, radar_dir, 0)
    ml = np.sign(rows.prob_up.fillna(0.5) - 0.5)
    out = pd.DataFrame({"radar": radar, "firsatlar": firsatlar, "pariteler": pariteler, "ml": ml}, index=rows.index).astype(int)
    first = out["radar"]
    out["technical"] = np.where((out[["radar", "firsatlar", "pariteler"]].eq(first, axis=0)).all(axis=1) & (first != 0), first, 0)
    out["consensus"] = np.where((out["technical"] != 0) & (out["ml"] == out["technical"]), out["technical"], 0)
    return out


def outcomes(rows: pd.DataFrame, side: pd.Series) -> pd.DataFrame:
    """Signed move, net move and hit, in the direction `side`, for every horizon."""
    out = pd.DataFrame(index=rows.index)
    for name in HORIZONS:
        move = side * rows[f"up_{name}"]
        out[f"move_{name}"] = move
        out[f"net_{name}"] = move - rows["cost"]
        out[f"hit_{name}"] = (move > 0).astype(float).where(move.notna())
    return out


def bootstrap_mean(values: pd.Series, days: pd.Series, seed: int = 42) -> dict:
    """Mean and day-block bootstrap draws of the mean (all pairs' moments of a day resampled together)."""
    ok = values.notna()
    per_day = pd.DataFrame({"day": days[ok], "v": values[ok]}).groupby("day")["v"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(0, len(per_day), len(per_day)), minlength=len(per_day))
                        for _ in range(BOOTSTRAPS)])
    draws = (weights @ per_day["sum"].to_numpy()) / (weights @ per_day["count"].to_numpy())
    tail = (1 - CI_LEVEL) / 2
    return {"n": int(per_day["count"].sum()), "mean": float(per_day["sum"].sum() / per_day["count"].sum()),
            "ci": [float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))]}


def summary_row(result: pd.DataFrame) -> dict:
    row = {"n": int(result["move_1h"].notna().sum())}
    for name in HORIZONS:
        for kind in ("move", "net", "hit"):
            row[f"{kind}_{name}"] = float(result[f"{kind}_{name}"].mean())
    return row


def evaluate() -> None:
    rows = pd.concat([pd.read_csv(f, parse_dates=["time"], index_col="time") for f in sorted(ROWS.glob("*.csv.gz"))])
    rows = rows.sort_index(kind="stable")
    sides = view_sides(rows)
    days = pd.Series(rows.index.floor("D"), index=rows.index)
    year = pd.Series(rows.index.year, index=rows.index)
    consensus = sides["consensus"] != 0
    result = outcomes(rows, sides["consensus"])

    h1 = {"all": {h: bootstrap_mean(result.loc[consensus, f"net_{h}"], days[consensus]) for h in TESTED_HORIZONS}}
    for label, (y0, y1) in SUB_PERIODS.items():
        part = consensus & year.between(y0, y1)
        h1[label] = {"net_4h": float(result.loc[part, "net_4h"].mean()), "n": int(part.sum())}
    h1["passes"] = bool(all(h1["all"][h]["ci"][0] > 0 for h in TESTED_HORIZONS)
                        and all(h1[label]["net_4h"] > 0 for label in SUB_PERIODS))

    info = {}
    for view in VIEWS + ("technical",):
        on = sides[view] != 0
        info[f"view_{view}"] = summary_row(outcomes(rows[on], sides.loc[on, view]))
    tech = sides["technical"] != 0
    for label, mask in (("technical_ml_agrees", tech & (sides.ml == sides.technical)),
                        ("technical_ml_disagrees", tech & (sides.ml == -sides.technical))):
        info[label] = summary_row(outcomes(rows[mask], sides.loc[mask, "technical"]))
    previous = sides.groupby(rows["pair"])["consensus"].shift(1).fillna(0)
    new_call = consensus & (previous != sides["consensus"])
    info["consensus_new_calls"] = summary_row(result[new_call])
    info["consensus_new_calls_ci"] = {h: bootstrap_mean(result.loc[new_call, f"net_{h}"], days[new_call]) for h in TESTED_HORIZONS}
    info["consensus_by_sub_period"] = {label: summary_row(result[consensus & year.between(y0, y1)])
                                       for label, (y0, y1) in SUB_PERIODS.items()}
    info["consensus_by_pair"] = {pair: int(n) for pair, n in rows.loc[consensus, "pair"].value_counts().sort_index().items()}
    info["consensus_long_short"] = {"LONG": int((sides.consensus == 1).sum()), "SHORT": int((sides.consensus == -1).sum())}

    results = {"pairs": int(rows.pair.nunique()), "period": [str(rows.index.min()), str(rows.index.max())],
               "moments": len(rows), "consensus_moments": int(consensus.sum()),
               "consensus": summary_row(result[consensus]), "H1_consensus_beats_cost": h1, "information_only": info,
               "note": "net = move - (measured spread at NY hour + 0.7 pip); moves in pips in the consensus direction"}
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"pairs {results['pairs']} | {results['period']} | moments {results['moments']:,} | consensus {results['consensus_moments']:,}")
    print(f"=== H1 passes={h1['passes']}")
    for h in TESTED_HORIZONS:
        s = h1["all"][h]
        print(f"  net {h}: {s['mean']:+.2f} [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] n={s['n']:,}")
    for label in SUB_PERIODS:
        print(f"  {label}: net 4h {h1[label]['net_4h']:+.2f} n={h1[label]['n']:,}")
    print("--- information only")
    for name, row in info.items():
        if isinstance(row, dict) and "move_1h" in row:
            print(f"  {name:24s} n={row['n']:>8,}  move 1h {row['move_1h']:+.2f} 4h {row['move_4h']:+.2f} 24h {row['move_24h']:+.2f}  "
                  f"net 1h {row['net_1h']:+.2f} 4h {row['net_4h']:+.2f}  hit 1h {row['hit_1h']:.3f} 4h {row['hit_4h']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "evaluate"])
    parser.add_argument("--pairs", nargs="+", default=PAIRS)
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.stage == "build":
        ROWS.mkdir(parents=True, exist_ok=True)
        tasks = [(pair, year) for year in args.years for pair in args.pairs]
        with Pool(args.workers) as pool:
            for message in pool.imap_unordered(safe_build, tasks):
                print(message, flush=True)
    else:
        evaluate()
