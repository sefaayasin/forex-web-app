"""Which free economic-calendar source's forecast lands closest to the released value.

Rules live in research/calendar_sources/protocol.json and were written before any
result was computed. Investing.com is downloaded by fetch_investing.js (a browser
session is needed); the other three sources are plain HTTP.

Download: python -X utf8 research_calendar_sources.py download
Analyse:  python -X utf8 research_calendar_sources.py
"""
from __future__ import annotations

import html as html_lib
import json
import re
import sys
import time
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import requests
from scipy.stats import binomtest

from forex_calendar import extract_days as forexfactory_days, week_param

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "research" / "calendar_sources"
RAW = OUTPUT / "raw"
PROTOCOL = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
START, END = (date.fromisoformat(d) for d in PROTOCOL["period_utc"])
CURRENCIES = set(PROTOCOL["currencies"])
SOURCES = ("investing", "forexfactory", "fxstreet", "nasdaq")
# investing.com kept blocking the download; the main comparison uses the other three (protocol amendment 2).
MAIN_SOURCES = ("forexfactory", "fxstreet", "nasdaq")
INVESTING_WINDOW = (date(2024, 9, 30), date(2024, 12, 22))
NEW_YORK = pytz.timezone("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"}

MULTIPLIERS = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
MATCH_WINDOW = pd.Timedelta(minutes=5)
MIN_RELEASES_FOR_ERROR_SCALE = 6
BOOTSTRAP_DRAWS = 5000
NASDAQ_COUNTRY_CURRENCY = {
    "United States": "USD", "Euro Zone": "EUR", "Germany": "EUR", "France": "EUR", "Italy": "EUR", "Spain": "EUR",
    "United Kingdom": "GBP", "Japan": "JPY", "Canada": "CAD", "Australia": "AUD", "New Zealand": "NZD", "Switzerland": "CHF",
}


# ----------------------------------------------------------------------------- parsing

def parse_value(text: object) -> Optional[float]:
    """'162K' -> 162000, '-0.3%' -> -0.3, '1,826K' -> 1826000; anything non-numeric -> None."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    s = html_lib.unescape(str(text)).replace("−", "-").replace(",", "").replace("\xa0", " ").strip()
    s = re.sub(r"[%$€£¥]", "", s).strip()
    m = re.fullmatch(r"[<>]?\s*(-?\d+(?:\.\d+)?)\s*([KMBT]?)", s, flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1)) * MULTIPLIERS.get(m.group(2).upper(), 1.0)


def fxstreet_value(value: object, potency: object) -> Optional[float]:
    if value is None:
        return None
    return float(value) * MULTIPLIERS.get(str(potency or "").upper(), 1.0)


def normalize_name(name: str) -> set[str]:
    s = str(name).lower()
    for old, new in (("m/m", " mom "), ("y/y", " yoy "), ("q/q", " qoq "), ("3m/3m", " 3m3m ")):
        s = s.replace(old, new)
    s = re.sub(r"\((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4])\)", " ", s)
    return set(re.findall(r"[a-z0-9]+", s))


def name_similarity(a: str, b: str) -> float:
    ta, tb = normalize_name(a), normalize_name(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def nasdaq_release_time(day: date, hhmm: str) -> Optional[pd.Timestamp]:
    """Nasdaq's 'gmt' column is New York wall-clock time."""
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", str(hhmm).strip())
    if not m:
        return None
    local = NEW_YORK.localize(pd.Timestamp(day).to_pydatetime().replace(hour=int(m.group(1)), minute=int(m.group(2))))
    return pd.Timestamp(local).tz_convert("UTC")


def nasdaq_query_date(day: date) -> date:
    """api.nasdaq.com economicevents?date=D returns the releases of D-1."""
    return day + timedelta(days=1)


# ----------------------------------------------------------------------------- download

def _get(url: str, **headers) -> requests.Response:
    for attempt in range(3):
        try:
            r = requests.get(url, headers={**UA, **headers}, timeout=40)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"download failed: {url}")


def _mondays() -> list[date]:
    first = START - timedelta(days=START.weekday())
    return [first + timedelta(weeks=i) for i in range((END - first).days // 7 + 1)]


def download_forexfactory() -> None:
    out = RAW / "forexfactory"
    out.mkdir(parents=True, exist_ok=True)
    for monday in _mondays():
        path = out / f"{monday}.json"
        if path.exists():
            continue
        page = _get(f"https://www.forexfactory.com/calendar?week={week_param(monday)}").text
        path.write_text(json.dumps(forexfactory_days(page)), encoding="utf-8")
        time.sleep(1.0)


def download_fxstreet() -> None:
    out = RAW / "fxstreet"
    out.mkdir(parents=True, exist_ok=True)
    for month in pd.period_range(START, END, freq="M"):
        path = out / f"{month}.json"
        if path.exists():
            continue
        first, last = month.start_time.date(), month.end_time.date()
        url = f"https://calendar-api.fxstreet.com/en/api/v1/eventDates/{first}T00:00:00Z/{last}T23:59:59Z"
        path.write_text(_get(url, Origin="https://www.fxstreet.com", Referer="https://www.fxstreet.com/").text, encoding="utf-8")
        time.sleep(1.0)


def download_nasdaq() -> None:
    out = RAW / "nasdaq"
    out.mkdir(parents=True, exist_ok=True)
    day = START
    while day <= END:
        path = out / f"{day}.json"
        if not path.exists() and day.weekday() < 5:
            url = f"https://api.nasdaq.com/api/calendar/economicevents?date={nasdaq_query_date(day)}"
            path.write_text(_get(url, Accept="application/json", Origin="https://www.nasdaq.com", Referer="https://www.nasdaq.com/").text, encoding="utf-8")
            time.sleep(0.4)
        day += timedelta(days=1)


# ----------------------------------------------------------------------------- load

def load_forexfactory() -> pd.DataFrame:
    rows = []
    for path in sorted((RAW / "forexfactory").glob("*.json")):
        for day in json.loads(path.read_text(encoding="utf-8")):
            for e in day["events"]:
                rows.append({
                    "id": e["id"], "series": e["ebaseId"], "name": e["name"], "currency": e["currency"],
                    "time": pd.Timestamp(e["dateline"], unit="s", tz="UTC"), "impact": e.get("impactName", ""),
                    "actual": parse_value(e.get("actual")), "forecast": parse_value(e.get("forecast")),
                    "previous": parse_value(e.get("previous")),
                })
    return pd.DataFrame(rows).drop_duplicates("id")


def load_fxstreet() -> pd.DataFrame:
    rows = []
    for path in sorted((RAW / "fxstreet").glob("*.json")):
        for e in json.loads(path.read_text(encoding="utf-8")):
            rows.append({
                "name": e["name"], "currency": e.get("currencyCode"), "time": pd.Timestamp(e["dateUtc"]),
                "actual": fxstreet_value(e.get("actual"), e.get("potency")),
                "forecast": fxstreet_value(e.get("consensus"), e.get("potency")),
            })
    return pd.DataFrame(rows).drop_duplicates()


def load_nasdaq() -> pd.DataFrame:
    rows = []
    for path in sorted((RAW / "nasdaq").glob("*.json")):
        day = date.fromisoformat(path.stem)
        data = json.loads(path.read_text(encoding="utf-8")).get("data") or {}
        for e in data.get("rows") or []:
            when = nasdaq_release_time(day, e.get("gmt", ""))
            currency = NASDAQ_COUNTRY_CURRENCY.get(e.get("country", ""))
            if when is None or currency is None:
                continue
            rows.append({
                "name": e["eventName"], "currency": currency, "time": when,
                "actual": parse_value(e.get("actual")), "forecast": parse_value(e.get("consensus")),
            })
    return pd.DataFrame(rows).drop_duplicates()


INVESTING_ROW = re.compile(r'<tr id="eventRowId_(\d+)"[^>]*data-event-datetime="([^"]+)"[^>]*>(.*?)</tr>', re.S)


def _cell(row_html: str, cell_id: str) -> str:
    m = re.search(rf'id="{cell_id}_\d+"[^>]*>(.*?)</td>', row_html, re.S)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""


def parse_investing_page(page: str) -> list[dict]:
    rows = []
    for row_id, when, body in INVESTING_ROW.findall(page):
        currency = re.search(r'flagCur[^>]*>.*?</span>\s*([A-Z]{3})', body, re.S)
        name = re.search(r'class="left event"[^>]*>\s*<a[^>]*>(.*?)</a>', body, re.S)
        rows.append({
            "row_id": int(row_id),
            "time": pd.Timestamp(when.replace("/", "-"), tz="UTC"),
            "currency": currency.group(1) if currency else None,
            "importance": body.count("grayFullBullishIcon"),
            "name": html_lib.unescape(re.sub(r"\s+", " ", name.group(1))).strip() if name else "",
            "actual": parse_value(_cell(body, "eventActual")),
            "forecast": parse_value(_cell(body, "eventForecast")),
        })
    return rows


def load_investing() -> pd.DataFrame:
    rows = []
    for path in sorted((RAW / "investing").glob("*.json")):
        for page in json.loads(path.read_text(encoding="utf-8"))["pages"]:
            rows.extend(parse_investing_page(page))
    return pd.DataFrame(rows).drop_duplicates("row_id")


# ----------------------------------------------------------------------------- matching

def actuals_agree(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 or abs(a - b) <= 0.005 * max(abs(a), abs(b))


def match_source(reference: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """For each reference event, the same-currency row within 5 minutes whose actual agrees; best name wins."""
    rows = rows.dropna(subset=["actual", "currency"]).reset_index(drop=True)
    by_currency = {cur: grp for cur, grp in rows.groupby("currency")}
    used: set[int] = set()
    matched = []
    for _, ref in reference.sort_values("time").iterrows():
        pool = by_currency.get(ref["currency"])
        if pool is None:
            continue
        near = pool[(pool["time"] - ref["time"]).abs() <= MATCH_WINDOW]
        best, best_score = None, -1.0
        for idx, row in near.iterrows():
            if idx in used or not actuals_agree(ref["actual"], row["actual"]):
                continue
            score = name_similarity(ref["name"], row["name"])
            if score > best_score:
                best, best_score = idx, score
        if best is not None:
            used.add(best)
            matched.append({"id": ref["id"], "actual": rows.at[best, "actual"], "forecast": rows.at[best, "forecast"], "name": rows.at[best, "name"]})
    return pd.DataFrame(matched, columns=["id", "actual", "forecast", "name"])


def build_panel(reference: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per reference event; per source its absolute forecast error (NaN when missing)."""
    panel = reference[["id", "series", "name", "currency", "time", "impact", "actual", "previous"]].copy()
    for source, rows in sources.items():
        m = match_source(reference, rows) if source != "forexfactory" else reference[["id", "actual", "forecast"]]
        m = m.assign(**{f"err_{source}": (m["actual"] - m["forecast"]).abs()})
        panel = panel.merge(m[["id", f"err_{source}"]], on="id", how="left")
        panel[f"matched_{source}"] = panel["id"].isin(m["id"])
    return panel


def add_scaled_errors(panel: pd.DataFrame, sources: tuple[str, ...] = SOURCES) -> pd.DataFrame:
    err_cols = [f"err_{s}" for s in sources]
    per_event = panel[err_cols].median(axis=1)
    scales = {}
    for series, grp in panel.groupby("series"):
        scale = per_event.loc[grp.index].median() if len(grp) >= MIN_RELEASES_FOR_ERROR_SCALE else 0.0
        if not scale or not np.isfinite(scale):
            scale = (grp["actual"] - grp["previous"]).abs().median()
        scales[series] = scale if scale and np.isfinite(scale) else np.nan
    panel = panel.assign(scale=panel["series"].map(scales)).dropna(subset=["scale"])
    for s in sources:
        panel[f"scaled_{s}"] = panel[f"err_{s}"] / panel["scale"]
    return panel


# ----------------------------------------------------------------------------- statistics

def head_to_head(a: pd.Series, b: pd.Series) -> dict:
    both = a.notna() & b.notna()
    diff = (a[both] - b[both]).to_numpy()
    tie = np.isclose(diff, 0.0, atol=1e-12)
    wins, losses = int(((diff < 0) & ~tie).sum()), int(((diff > 0) & ~tie).sum())
    p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
    return {"n": int(both.sum()), "a_closer": wins, "tie": int(tie.sum()), "b_closer": losses, "sign_p": float(p)}


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def bootstrap_mean_difference(diff: pd.Series, days: pd.Series, draws: int = BOOTSTRAP_DRAWS, seed: int = 7) -> tuple[float, float]:
    """95% interval of the mean of `diff`, resampling whole release days."""
    per_day = pd.DataFrame({"d": diff.to_numpy(), "day": days.to_numpy()}).groupby("day")["d"].agg(["sum", "count"])
    sums, counts = per_day["sum"].to_numpy(), per_day["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(draws, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(panel: pd.DataFrame, sources: tuple[str, ...] = SOURCES) -> dict:
    coverage = {s: float(panel[f"err_{s}"].notna().mean()) for s in sources}
    own_mean = {s: float(panel[f"scaled_{s}"].mean()) for s in sources}
    pairs = []
    for a, b in combinations(sources, 2):
        h = head_to_head(panel[f"err_{a}"], panel[f"err_{b}"])
        both = panel[f"scaled_{a}"].notna() & panel[f"scaled_{b}"].notna()
        diff = panel.loc[both, f"scaled_{a}"] - panel.loc[both, f"scaled_{b}"]
        low, high = bootstrap_mean_difference(diff, panel.loc[both, "time"].dt.date) if both.sum() > 1 else (np.nan, np.nan)
        pairs.append({
            "a": a, "b": b, **h,
            "mean_scaled_a": float(panel.loc[both, f"scaled_{a}"].mean()),
            "mean_scaled_b": float(panel.loc[both, f"scaled_{b}"].mean()),
            "diff_ci_low": low, "diff_ci_high": high,
        })
    for pair, p_adj in zip(pairs, holm([p["sign_p"] for p in pairs])):
        pair["sign_p_holm"] = p_adj
    return {"events": int(len(panel)), "coverage": coverage, "mean_scaled_error_on_own_coverage": own_mean, "pairs": pairs}


def best_source(result: dict, sources: tuple[str, ...] = SOURCES) -> Optional[str]:
    """Protocol decision rule: lowest mean and significantly better than every other source."""
    for s in sources:
        beats_all = True
        for p in result["pairs"]:
            if s not in (p["a"], p["b"]):
                continue
            if p["a"] == s:
                ok = p["a_closer"] > p["b_closer"] and p["sign_p_holm"] < 0.05 and p["diff_ci_high"] < 0
            else:
                ok = p["b_closer"] > p["a_closer"] and p["sign_p_holm"] < 0.05 and p["diff_ci_low"] > 0
            beats_all &= bool(ok)
        if beats_all:
            return s
    return None


def main() -> None:
    if sys.argv[1:] == ["download"]:
        download_forexfactory()
        download_fxstreet()
        download_nasdaq()
        return

    ff = load_forexfactory()
    in_scope = (
        ff["currency"].isin(CURRENCIES) & ff["actual"].notna()
        & (ff["time"].dt.date >= START) & (ff["time"].dt.date <= END)
    )
    all_rows = {"investing": load_investing(), "forexfactory": ff, "fxstreet": load_fxstreet(), "nasdaq": load_nasdaq()}

    def run(reference: pd.DataFrame, sources: tuple[str, ...]) -> tuple[dict, pd.DataFrame]:
        panel = add_scaled_errors(build_panel(reference, {s: all_rows[s] for s in sources}), sources)
        result = compare(panel, sources)
        result["match_rate"] = {s: float(panel[f"matched_{s}"].mean()) for s in sources}
        result["by_currency_mean_scaled"] = {
            cur: {s: float(grp[f"scaled_{s}"].mean()) for s in sources} for cur, grp in panel.groupby("currency")
        }
        result["best_source"] = best_source(result, sources)
        return result, panel

    results, panels = {}, []
    window = (ff["time"].dt.date >= INVESTING_WINDOW[0]) & (ff["time"].dt.date <= INVESTING_WINDOW[1])
    for label, impact in (("primary_high", "high"), ("secondary_medium", "medium")):
        is_impact = in_scope & (ff["impact"] == impact)
        results[label], panel = run(ff[is_impact], MAIN_SOURCES)
        panels.append(panel.assign(set=label))
        results[f"side_investing_window_{impact}"], _ = run(ff[is_impact & window], SOURCES)

    pd.concat(panels).to_csv(OUTPUT / "events.csv", index=False)
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
