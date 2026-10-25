"""Frozen-rule historical replay of Pivot Trader v2. No fitting or threshold search.

python -X utf8 research_pivot_advisor.py --pairs EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD
The same advise(), indicators and model prediction path as the app are used.
Decision every completed 5M candle; price is that known close, never a forming
15M candle's eventual close. Execution starts at the next 5M open.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from forex_analysis import bias_score_frame, market_structure_frame
from forex_config import MAJOR_PAIRS, TIMEFRAMES, get_pip_size
from forex_costs import current_ny_hour, estimated_round_trip_cost_pips, load_spread_profile
from forex_freshness import reversal_warnings
from forex_ml_live import build_research_prediction, has_research_model
from forex_pivot_advisor import ENTRY_ZONE_ATR, STRATEGY_ID, advise, closed_bars, recent_bar
from forex_pivot_plan import atr14, pivot_plan
from forex_pivots import levels_for_day

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research/pivot_advisor"
APP = ROOT / "forex_web_app_streamlit_v14_alert_decision.py"
CALENDAR = ROOT / "research/calendar_sources/raw/forexfactory"
START = pd.Timestamp("2024-10-01", tz="UTC")
END = pd.Timestamp("2026-09-12", tz="UTC")
TF = {"4H": ("4h", 240, 60), "1H": ("60m", 60, 30),
      "15M": ("15m", 15, 10), "5M": ("5m", 5, 5)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_bars(pair, folder, start=None):
    data = pd.read_csv(ROOT / "data" / folder / f"{pair}.csv")
    data.index = pd.to_datetime(data.pop("timestamp"), unit="ms", utc=True)
    data = data.rename(columns=str.capitalize).sort_index()
    if data.index.has_duplicates:
        raise ValueError(f"duplicate times: {pair} {folder}")
    if start is not None:
        data = data[data.index >= start]
    if not np.isfinite(data[["Open", "High", "Low", "Close"]]).all().all():
        raise ValueError("invalid OHLC")
    # Archive contains a few one-tick bounds errors. Preserve O/C; expand bounds.
    data["High"] = data[["Open", "High", "Close"]].max(axis=1)
    data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)
    return data


def load_calendar():
    rows, weeks = [], set()
    for path in sorted(CALENDAR.glob("*.json")):
        week = pd.Timestamp(path.stem, tz="UTC")
        weeks.add(week)
        for day in json.loads(path.read_text(encoding="utf-8")):
            for event in day["events"]:
                if str(event.get("impactName", "")).lower() == "high":
                    rows.append({"time": pd.Timestamp(event["dateline"], unit="s", tz="UTC"),
                                 "currency": event["currency"], "title": event["name"]})
    return pd.DataFrame(rows).drop_duplicates().sort_values("time"), weeks


def news_at(events, pair, now):
    relevant = events[events.currency.isin([pair[:3], pair[3:]])]
    relevant = relevant[(relevant.time >= now - pd.Timedelta(minutes=15))
                        & (relevant.time <= now + pd.Timedelta(hours=1))]
    if relevant.empty:
        return None, ""
    row = relevant.iloc[0]
    return int((row.time - now).total_seconds() // 60), f"{row.currency} {row.title}"


class History:
    def __init__(self, pair, start):
        self.pair = pair
        self.five = read_bars(pair, "historical_5m", start - pd.Timedelta(days=70))
        self.fifteen = read_bars(pair, "historical_15m", start - pd.Timedelta(days=70))
        self.hourly = read_bars(pair, "historical_1h_from_15m", start - pd.Timedelta(days=160))
        self.daily = read_bars(pair, "historical_1d")
        self.four = self.hourly.resample("4h", label="right", closed="right").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        self.sources = {"5m": self.five, "15m": self.fifteen, "60m": self.hourly, "4h": self.four}
        self.now = None
        self.frame_cache = {}
        self.ml_cache = {}
        self.daily_cache = {}

    def price(self, now):
        pos = self.five.index.searchsorted(now - pd.Timedelta(minutes=5), side="right") - 1
        return float(self.five.Close.iloc[pos]), self.five.index[pos] + pd.Timedelta(minutes=5)

    def raw(self, symbol, interval, period):
        if interval == "1d":
            return self.daily[self.daily.index < self.now]
        source = self.sources[interval]
        days = int(period.removesuffix("d"))
        start = self.now - pd.Timedelta(days=days)
        part = source[(source.index >= start) & (source.index <= self.now)].copy()
        if interval == "15m":
            # Replace future contents of a still-forming candle by completed 5M data.
            part = part[part.index + pd.Timedelta(minutes=15) <= self.now]
            candle_start = self.now.floor("15min")
            partial = self.five[(self.five.index >= candle_start)
                                & (self.five.index + pd.Timedelta(minutes=5) <= self.now)]
            price, _ = self.price(self.now)
            part.loc[candle_start] = ([partial.Open.iloc[0], partial.High.max(), partial.Low.min(),
                                       partial.Close.iloc[-1], partial.Volume.sum()] if len(partial)
                                      else [price, price, price, price, 0])
        return part

    def levels(self, now):
        day = now.normalize()
        if day not in self.daily_cache:
            self.daily_cache[day] = levels_for_day(self.daily, day)
        return self.daily_cache[day]

    def frame(self, label, now):
        interval, minutes, days = TF[label]
        source = self.sources[interval]
        lo = int(source.index.searchsorted(now - pd.Timedelta(days=days), side="left"))
        hi = int(source.index.searchsorted(now - pd.Timedelta(minutes=minutes), side="right"))
        key = (label, lo, hi)
        if key not in self.frame_cache:
            part = source.iloc[lo:hi]
            if len(part) < 220:
                return pd.DataFrame()
            frame = market_structure_frame(part)
            frame["Score"] = bias_score_frame(part)["score"]
            # Most windows are never revisited; bound memory to a few recent windows.
            if len(self.frame_cache) > 24:
                self.frame_cache.clear()
            self.frame_cache[key] = frame
        return self.frame_cache[key]

    def prediction(self, now):
        lo = self.hourly.index.searchsorted(now - pd.Timedelta(days=150), side="left")
        hi = self.hourly.index.searchsorted(now - pd.Timedelta(hours=1), side="right")
        key = (lo, hi)
        if key not in self.ml_cache:
            self.ml_cache[key] = build_research_prediction(self.pair, "direction", self.hourly.iloc[lo:hi], now=now)
            if len(self.ml_cache) > 8:
                self.ml_cache = {key: self.ml_cache[key]}
        return self.ml_cache[key]

    def snapshot(self, now, plan, atr, price):
        frames = {label: self.frame(label, now) for label in TF}
        good = all(not frame.empty for frame in frames.values())
        fresh = good and all(recent_bar(frame.index[-1], now, TF[label][1]) for label, frame in frames.items())
        _, price_at = self.price(now)
        fresh = fresh and price_at == now
        levels = self.levels(now)
        previous = now.tz_localize(None).normalize() - pd.offsets.BDay(1)
        warnings = ([f"{label}: {warning}" for label in ("15M", "5M")
                     for warning in reversal_warnings(frames[label], plan["side"])] if good else None)
        return {**plan, "symbol": self.pair + "=X", "provider": "Archive replay", "as_of": now,
                "pip": get_pip_size(self.pair), "price": price, "atr": atr,
                "fresh": fresh, "pivot_fresh": pd.Timestamp(levels["source_day"]) == previous,
                "distance_atr": abs(price - plan["entry"]) / atr,
                "scores": {label: float(frame.Score.iloc[-1]) for label, frame in frames.items() if not frame.empty},
                "response_side": frames["15M"].ResponseSide.iloc[-1] if not frames["15M"].empty else "NONE",
                "reversal_warnings": warnings, "ml": self.prediction(now)}


def simulate(bars, now, side, entry, stop, target, pip, hours=4):
    """Stop-first ambiguity policy, next-open execution, adverse stop-gap fills."""
    sign = 1 if side == "LONG" else -1
    first = bars.index.searchsorted(now, side="left")
    if first >= len(bars) or bars.index[first] != now:
        return None
    deadline = now + pd.Timedelta(hours=hours)
    end = bars.index.searchsorted(deadline, side="left")
    if end == 0 or end > len(bars) or bars.index[min(end - 1, len(bars) - 1)] + pd.Timedelta(minutes=5) < deadline:
        return None
    path = bars.iloc[first:end]
    if path.empty or (path.index.to_series().diff().dropna() > pd.Timedelta(minutes=5)).any():
        return None
    actual_entry = float(path.Open.iloc[0])
    if sign * (actual_entry - stop) <= 0 or sign * (target - actual_entry) <= 0:
        return {"kind": "invalid_open", "entry_gap_pips": sign * (actual_entry - entry) / pip}
    ambiguity = False
    for stamp, row in path.iterrows():
        stop_hit = row.Low <= stop if sign == 1 else row.High >= stop
        target_hit = row.High >= target if sign == 1 else row.Low <= target
        if sign * (row.Open - stop) <= 0:
            price, kind = float(row.Open), "stop_gap"
        elif sign * (row.Open - target) >= 0:
            price, kind = target, "target"
        elif stop_hit:
            price, kind = stop, "stop"
            ambiguity = bool(target_hit)
        elif target_hit:
            price, kind = target, "target"
        else:
            continue
        return dict(kind=kind, exit_time=stamp + pd.Timedelta(minutes=5), exit_price=price,
                    actual_entry=actual_entry, gross_pips=sign * (price - actual_entry) / pip,
                    ambiguous=ambiguity, entry_gap_pips=sign * (actual_entry - entry) / pip)
    price = float(path.Close.iloc[-1])
    return dict(kind="time", exit_time=deadline, exit_price=price, actual_entry=actual_entry,
                gross_pips=sign * (price - actual_entry) / pip, ambiguous=False,
                entry_gap_pips=sign * (actual_entry - entry) / pip)


def direction_after(bars, now, hours, side, pip):
    target = now + pd.Timedelta(hours=hours)
    first = bars.index.searchsorted(now)
    end = bars.index.searchsorted(target)
    if first >= len(bars) or end >= len(bars) or bars.index[first] != now or bars.index[end] != target:
        return None
    if (bars.index[first + 1:end + 1] - bars.index[first:end] > pd.Timedelta(minutes=5)).any():
        return None
    return (1 if side == "LONG" else -1) * (float(bars.Open.iloc[end]) - float(bars.Open.iloc[first])) / pip


def verify_app_snapshot(history, now, expected):
    """Compile original app snapshot builder and compare from-scratch with replay."""
    fn = next(n for n in ast.parse(APP.read_text(encoding="utf-8")).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_pivot_advisor_snapshot")
    class TimestampClock:
        def __call__(self, *args, **kwargs):
            return pd.Timestamp(*args, **kwargs)
        def now(self, **kwargs):
            return now
    class PandasClock:
        Timestamp = TimestampClock()
        def __getattr__(self, name):
            return getattr(pd, name)
    history.now = now
    def prediction(symbol, task, bars=None, now=None, **kwargs):
        return build_research_prediction(symbol, task, bars=bars, now=now) if task == "direction" else {}
    namespace = dict(pd=PandasClock(), TIMEFRAMES=TIMEFRAMES, get_pip_size=get_pip_size,
                     summary_pivots=lambda sym: history.levels(now), fetch_ohlc=history.raw,
                     closed_bars=closed_bars, recent_bar=recent_bar, atr14=atr14, pivot_plan=pivot_plan,
                     _utc_index_df=lambda frame: frame, market_structure_frame=market_structure_frame,
                     score_series_for_backtest=lambda bars: bias_score_frame(bars)["score"],
                     reversal_warnings=reversal_warnings, has_research_model=has_research_model,
                     fetch_research_bars=lambda sym: history.raw(sym, "60m", "150d"),
                     research_range_ratio=lambda sym: None, build_research_prediction=prediction,
                     reliable_volatility_view=lambda pred: None)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "app_snapshot_check", "exec"), namespace)
    actual = namespace[fn.name](history.pair + "=X", "Archive replay")
    for key in ("side", "fresh", "pivot_fresh", "response_side", "reversal_warnings"):
        if actual.get(key) != expected.get(key):
            raise AssertionError(f"app parity {history.pair} {now}: {key}: {actual.get(key)} != {expected.get(key)}")
    for key in ("price", "entry", "stop", "target", "atr", "distance_atr"):
        if not np.isclose(actual[key], expected[key], rtol=0, atol=1e-12):
            raise AssertionError(f"app parity {key}")
    for key, value in expected["scores"].items():
        if value != actual["scores"][key]:
            raise AssertionError(f"app parity score {key}")
    if not np.isclose(actual["ml"].get("probability_up", np.nan), expected["ml"].get("probability_up", np.nan),
                      rtol=0, atol=1e-9, equal_nan=True):
        raise AssertionError("app parity ML")


def run_pair(pair, start, end, events, weeks, profile):
    tick = time.monotonic()
    history = History(pair, start)
    pip = get_pip_size(pair)
    five = history.five
    times = five.index + pd.Timedelta(minutes=5)
    positions = np.flatnonzero((times >= start) & (times < end) & (times.dayofweek < 5))
    atr_series = pd.Series(atr14(history.fifteen), index=history.fifteen.index)
    counts, rejected = Counter(), Counter()
    decisions, trades, checks = [], [], 0
    busy_until = start - pd.Timedelta(days=1)
    for count, pos in enumerate(positions, 1):
        now, price = times[pos], float(five.Close.iloc[pos])
        counts["moments"] += 1
        levels = history.levels(now)
        if not levels:
            continue
        m15pos = history.fifteen.index.searchsorted(now - pd.Timedelta(minutes=15), side="right") - 1
        if m15pos < 0:
            continue
        atr = float(atr_series.iloc[m15pos])
        plan = pivot_plan(price, levels, atr)
        if plan["side"] == "WAIT" or not np.isfinite(atr) or abs(price - plan["entry"]) / atr > ENTRY_ZONE_ATR:
            continue
        counts["near_entry"] += 1
        # Exact necessary conditions of market_structure_frame.ResponseSide.
        # This discards only candles that cannot possibly confirm a response;
        # no EMA approximation or outcome-based threshold is used.
        candle, prior = history.fifteen.iloc[m15pos], history.fifteen.iloc[m15pos - 1]
        breakout = (candle.Close > prior.High and candle.Close > candle.Open) if plan["side"] == "LONG" else (
            candle.Close < prior.Low and candle.Close < candle.Open)
        if not breakout:
            continue
        counts["near_and_breakout"] += 1
        m15 = history.frame("15M", now)
        if m15.empty or m15.ResponseSide.iloc[-1] != plan["side"]:
            continue
        counts["near_and_response"] += 1
        snap = history.snapshot(now, plan, atr, price)
        news_minutes, news_title = news_at(events, pair, now)
        monday = now.normalize() - pd.Timedelta(days=now.weekday())
        known = monday in weeks
        cost = estimated_round_trip_cost_pips(profile, pair, current_ny_hour(now))
        result = advise(snap, now=now, news_known=known, news_minutes=news_minutes, news_title=news_title, cost_pips=cost)
        if checks < 10 or (result["technical_ready"] and checks < 25):
            verify_app_snapshot(history, now, snap)
            checks += 1
        if not result["technical_ready"]:
            rejected.update(result["blockers"])
            continue
        counts["entry_decisions"] += 1
        record = {"pair": pair, "time": now, "side": plan["side"], "signal_price": price,
                  "entry": result["entry"], "stop": result["stop"], "target": result["target"],
                  "alignment": result["alignment"], "cost_pips": cost,
                  "ml_up": snap["ml"].get("probability_up"), "news_known": known}
        for hours in (1, 4):
            record[f"direction_{hours}h_pips"] = direction_after(five, now, hours, plan["side"], pip)
        decisions.append(record.copy())
        if now < busy_until:
            counts["overlapping_decisions"] += 1
            continue
        outcome = simulate(five, now, plan["side"], result["entry"], result["stop"], result["target"], pip)
        if outcome is None:
            counts["incomplete_paths"] += 1
            continue
        if outcome["kind"] == "invalid_open":
            counts["invalid_next_opens"] += 1
            continue
        risk = abs(outcome["actual_entry"] - result["stop"]) / pip
        record.update(outcome, net_pips=outcome["gross_pips"] - cost,
                      stress_net_pips=outcome["gross_pips"] - 2 * cost - .5,
                      risk_pips=risk, net_r=(outcome["gross_pips"] - cost) / (risk + cost))
        trades.append(record)
        busy_until = outcome["exit_time"]
        if len(trades) % 25 == 0:
            print(f"{pair}: {now.date()}, signals={len(decisions)}, trades={len(trades)}, {time.monotonic()-tick:.0f}s", flush=True)
    folder = OUT / pair
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decisions).to_csv(folder / "decisions.csv", index=False)
    pd.DataFrame(trades).to_csv(folder / "trades.csv", index=False)
    report = {"pair": pair, "counts": dict(counts), "app_parity_checks": checks,
              "rejections": dict(rejected), "elapsed_seconds": round(time.monotonic() - tick, 1)}
    (folder / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE {pair}: {report}", flush=True)
    return pd.DataFrame(decisions), pd.DataFrame(trades), report


def bootstrap(frame, column="net_pips"):
    if frame.empty:
        return None
    day = pd.to_datetime(frame.time, utc=True).dt.floor("D")
    grouped = frame.groupby(day)[column].agg(["sum", "count"])
    if len(grouped) < 2:
        return None
    rng = np.random.default_rng(20261020)
    draws = []
    for _ in range(2000):
        positions = rng.integers(0, len(grouped), len(grouped))
        sample = grouped.iloc[positions]
        draws.append(sample["sum"].sum() / sample["count"].sum())
    return np.quantile(draws, [.025, .975]).tolist()


def summarize(decisions, trades):
    result = {"decisions": len(decisions), "trades": len(trades)}
    if not decisions.empty:
        for h in (1, 4):
            moves = decisions[f"direction_{h}h_pips"].dropna()
            result[f"direction_{h}h"] = {"n": len(moves), "correct": int((moves > 0).sum()),
                                         "hit_rate": float((moves > 0).mean()) if len(moves) else None,
                                         "mean_pips": float(moves.mean()) if len(moves) else None}
    if not trades.empty:
        result.update(targets=int(trades.kind.eq("target").sum()),
                      stops=int(trades.kind.isin(["stop", "stop_gap"]).sum()),
                      timeouts=int(trades.kind.eq("time").sum()),
                      net_wins=int(trades.net_pips.gt(0).sum()),
                      net_win_rate=float(trades.net_pips.gt(0).mean()),
                      mean_net_pips=float(trades.net_pips.mean()), sum_net_pips=float(trades.net_pips.sum()),
                      mean_net_r=float(trades.net_r.mean()),
                      mean_stress_net_pips=float(trades.stress_net_pips.mean()),
                      ambiguous_candles=int(trades.ambiguous.sum()),
                      mean_cost_pips=float(trades.cost_pips.mean()),
                      mean_net_pips_ci95=bootstrap(trades))
    return result


def write_report(pairs, decisions, trades, audits, start=START, end=END):
    result = {"strategy_id": STRATEGY_ID, "period": [str(start), str(end)], "pairs": pairs,
              "all": summarize(decisions, trades),
              "by_side": {side: summarize(decisions[decisions.side == side] if not decisions.empty else decisions,
                                           trades[trades.side == side] if not trades.empty else trades)
                          for side in ("LONG", "SHORT")},
              "by_pair": {pair: summarize(decisions[decisions.pair == pair] if not decisions.empty else decisions,
                                           trades[trades.pair == pair] if not trades.empty else trades) for pair in pairs},
              "audits": audits}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    lines = ["# Pivot Trader v2 — geçmiş veri testi", "",
             f"Kurallar değiştirilmeden {len(pairs)} paritede tarandı. Model eğitimi 2023 öncesi; test {start.date()}–{(end-pd.Timedelta(days=1)).date()}.", "",
             "Karar her 5 dakikalık mum kapanışında. Oluşan 15M mumun sonradan gerçekleşen kapanışı kullanılmadı.",
             "Giriş sonraki 5M mum açılışında; aynı paritede açık işlem varken yeni işlem sayılmadı.",
             "Plan 4 saat sonunda kapanır. Aynı mumda stop ve hedef varsa stop kabul edildi; stop boşlukları kötü fiyattan yazıldı.", "",
             "| Yön | Giriş kararı | İşlem | 1s yön doğru | 4s yön doğru | Hedef | Stop | Süre sonu | Net kazanma | Net pip/işlem |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for side, value in [("Tümü", result["all"]), *result["by_side"].items()]:
        def hit(h):
            v = value.get(f"direction_{h}h", {})
            return f"%{100*v['hit_rate']:.1f} ({v['correct']}/{v['n']})" if v.get("hit_rate") is not None else "—"
        win = f"%{100*value['net_win_rate']:.1f}" if "net_win_rate" in value else "—"
        net = f"{value['mean_net_pips']:+.2f}" if "mean_net_pips" in value else "—"
        lines.append(f"| {side} | {value['decisions']} | {value['trades']} | {hit(1)} | {hit(4)} | "
                     f"{value.get('targets',0)} | {value.get('stops',0)} | {value.get('timeouts',0)} | {win} | {net} |")
    lines += ["", "## Sınırlar", "",
              "- Bu veri Yahoo/broker canlı akışının kaydı değildir; yerel Dukascopy fiyatlarıyla aynı kodun tekrar oynatılmasıdır.",
              "- Takvimde yalnız planlı yüksek etkili haber saatleri kullanıldı; actual/forecast/surprise okunmadı. Tarihsel takvim sürümleri doğrulanmadı.",
              "- Spread ve komisyon saatlik tarihsel profil tahminidir; her işlem anının gerçek alış/satış kotasyonu yoktur.",
              "- Stres sonucu iki kat maliyet ve ek 0,5 pip kayma içerir. Swap dahil değildir; fiyatlar USD kazancına çevrilmez.",
              "- 1s/4s yön ölçümü tüm GİR kararlarını kapsar; tekrarlanan kararlar bağımsız işlemler değildir.",
              "- İşlem sonucu ayrı ve çakışmasızdır. Eksik 5M yolları ve geçersiz sonraki açılışlar sayılmadı; audit.json sayıları içerir.",
              "- Tarihsel test görülmüş veridir; gerçek ileri test / %70 başarı doğrulaması değildir.",
              "", "Parite ayrıntıları, gün/saat, karar fiyatı, stop, hedef ve sonuçlar aynı klasördeki CSV dosyalarında."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["all"], ensure_ascii=False, indent=2), flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=[p.replace("=X", "") for p in MAJOR_PAIRS])
    parser.add_argument("--start", default=str(START.date()))
    parser.add_argument("--end", default=str(END.date()))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / p for p in ("forex_pivot_advisor.py", "forex_pivot_plan.py", "forex_analysis.py", "forex_indicators.py",
                                  "forex_trade_plan.py", "forex_ml_live.py", "forex_costs.py", "research_pivot_advisor.py")]
    sources.append(APP)
    inputs = [ROOT / "data/ml/spread_profile.json", ROOT / "data/news/fomc_statements.csv"]
    inputs += sorted(CALENDAR.glob("*.json"))
    for pair in args.pairs:
        inputs += [ROOT / "data" / folder / f"{pair}.csv" for folder in
                   ("historical_5m", "historical_15m", "historical_1h_from_15m", "historical_1d")]
        inputs.append(ROOT / "data/ml/tournament" / f"{pair}_direction_research.joblib")
    protocol = {"strategy": STRATEGY_ID, "start": str(start), "end": str(end), "pairs": args.pairs,
                "decision_grid": "every completed 5M candle", "exit_horizon_hours": 4,
                "selection": "none; frozen current rule, no search", "sha256": {p.name: digest(p) for p in sources},
                "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
                "known_limitations": ["Historical schedule vintages unverified", "hour-of-day spread profile scenario",
                                      "Dukascopy replay, not historical Yahoo quotes", "prior research periods already inspected"]}
    # Record before loading outcome paths; do not overwrite a different frozen run.
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Frozen protocol differs; use a separate output directory / preserve prior run.")
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    events, weeks = load_calendar()
    profile = load_spread_profile()
    all_decisions, all_trades, audits = [], [], []
    for pair in args.pairs:
        d, t, audit = run_pair(pair, start, end, events, weeks, profile)
        all_decisions.append(d)
        all_trades.append(t)
        audits.append(audit)
    decisions = pd.concat(all_decisions, ignore_index=True)
    trades = pd.concat(all_trades, ignore_index=True)
    write_report(args.pairs, decisions, trades, audits, start, end)


if __name__ == "__main__":
    main()
