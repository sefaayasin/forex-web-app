"""Live ForexFactory economic calendar for the app's 📅 Takvim page.

research/calendar_sources/REPORT.md compared the forecasts of ForexFactory,
FXStreet, Nasdaq and (for 12 weeks) Investing against released values: none
was reliably closer than the others, and ForexFactory had by far the widest
forecast coverage (98% of high-impact releases), so the page uses it.

ForexFactory's weekly calendar page embeds its events as JSON, including the
released value and whether it beat the forecast. If that page cannot be
reached (e.g. blocked from the host), the public current-week feed is used
instead; it has forecast and previous values but no released value.

This module has no Streamlit dependency so it can be tested on its own.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PAGE_URL = "https://www.forexfactory.com/calendar?week={week}"
FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"}
IMPACT_LEVELS = {"high": 3, "medium": 2, "low": 1, "holiday": 0}
# ForexFactory's actualBetterWorse: 1 = better than forecast, 2 = worse, 0 = neither/no forecast.
OUTCOMES = {1: "better", 2: "worse"}
COLUMNS = ["time", "currency", "impact", "event", "actual", "forecast", "previous", "outcome", "time_label"]
# +1: a higher value is good for the currency, -1: a lower value is (unemployment, claims...).
# Learned from two years of ForexFactory releases by build_calendar_polarity.py.
POLARITY_PATH = Path(__file__).resolve().parent / "data" / "calendar_polarity.json"
VALUE_UNITS = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def pair_directions(currency: str, outcome: Optional[str], symbols: list[str]) -> tuple[list[str], list[str]]:
    """Pairs pushed up (LONG) and down (SHORT) when `currency` beats ('better') or misses ('worse') the forecast.

    A better-than-forecast release strengthens its currency: pairs quoting it as the
    base currency rise and pairs quoting it as the quote currency fall. This is the
    textbook first reaction to the surprise, not a trade signal.
    """
    if outcome not in ("better", "worse"):
        return [], []
    strong = outcome == "better"
    up, down = [], []
    for symbol in symbols:
        pair = symbol.replace("=X", "").upper()
        base, quote = pair[:3], pair[3:6]
        if currency == base:
            (up if strong else down).append(pair)
        elif currency == quote:
            (down if strong else up).append(pair)
    return up, down


def parse_value(text) -> Optional[float]:
    """'162K' -> 162000, '-0.3%' -> -0.3, '<0.1%' -> 0.1; None for text such as '2.92|2.1' or ''."""
    m = re.fullmatch(r"[<>]?(-?\d+(?:\.\d+)?)([KMBT%]?)", str(text or "").strip().replace(",", ""))
    if not m:
        return None
    return float(m.group(1)) * VALUE_UNITS.get(m.group(2), 1.0)


def learn_polarity(events: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Whether a higher value is good (+1) or bad (-1) for each (currency, event), from released surprises."""
    votes: dict[tuple[str, str], int] = {}
    for row in events.itertuples():
        actual, forecast = parse_value(row.actual), parse_value(row.forecast)
        if row.outcome not in ("better", "worse") or actual is None or forecast is None or actual == forecast:
            continue
        key = (row.currency, row.event)
        votes[key] = votes.get(key, 0) + (1 if (actual > forecast) == (row.outcome == "better") else -1)
    return {key: 1 if vote > 0 else -1 for key, vote in votes.items() if vote != 0}


def load_polarity(path: Path = POLARITY_PATH) -> dict[tuple[str, str], int]:
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {(item["currency"], item["event"]): int(item["polarity"]) for item in items}


def _direction(new: Optional[float], old: Optional[float], polarity: Optional[int]) -> Optional[str]:
    if new is None or old is None or not polarity:
        return None
    if new == old:
        return "equal"
    return "better" if (new > old) == (polarity > 0) else "worse"


def release_outcome(row, polarity: dict[tuple[str, str], int]) -> Optional[str]:
    """Released value against the forecast: ForexFactory's mark, else worked out from the event's polarity.

    ForexFactory leaves some surprises unmarked (e.g. US Unemployment Claims); 'equal' only when the values match.
    """
    if row.outcome in ("better", "worse"):
        return row.outcome
    actual, forecast = parse_value(row.actual), parse_value(row.forecast)
    if actual is not None and actual == forecast:
        return "equal"
    return _direction(actual, forecast, polarity.get((row.currency, row.event)))


def expected_outcome(row, polarity: dict[tuple[str, str], int]) -> Optional[str]:
    """Before the release: is the forecast better or worse for the currency than the previous value?

    Over two years of ForexFactory releases the data came out on the forecast's side of the
    previous value ~75% of the time, but the surprise against the forecast (what moves price)
    went the same way only ~53% of the time; see build_calendar_polarity.py.
    """
    return _direction(parse_value(row.forecast), parse_value(row.previous), polarity.get((row.currency, row.event)))


def extract_days(page: str) -> list[dict]:
    """The weekly page embeds `window.calendarComponentStates[1] = { days: [...] ...`."""
    start = page.index("days: [", page.index("window.calendarComponentStates")) + len("days: ")
    depth = 0
    for i in range(start, len(page)):
        depth += {"[": 1, "]": -1}.get(page[i], 0)
        if depth == 0:
            return json.loads(page[start:i + 1])
    raise ValueError("unterminated days array")


def rows_from_days(days: list[dict]) -> pd.DataFrame:
    rows = []
    for day in days:
        for e in day.get("events", []):
            rows.append({
                "id": e["id"],
                "time": pd.Timestamp(e["dateline"], unit="s", tz="UTC"),
                "currency": e.get("currency", ""),
                "impact": IMPACT_LEVELS.get(str(e.get("impactName", "")).lower(), 0),
                "event": e.get("name", ""),
                "actual": e.get("actual", "") or "",
                "forecast": e.get("forecast", "") or "",
                "previous": e.get("previous", "") or "",
                "outcome": OUTCOMES.get(e.get("actualBetterWorse")),
                "time_label": e.get("timeLabel", "") if e.get("timeMasked") else "",
            })
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows).drop_duplicates("id")[COLUMNS]


def week_param(day: date) -> str:
    """ForexFactory's week parameter, e.g. date(2026, 9, 28) -> 'sep28.2026'."""
    return f"{day.strftime('%b').lower()}{day.day}.{day.year}"


def fetch_week_page(day: date, timeout: float = 20) -> pd.DataFrame:
    resp = requests.get(PAGE_URL.format(week=week_param(day)), headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return rows_from_days(extract_days(resp.text))


def feed_rows(items: list[dict]) -> pd.DataFrame:
    rows = [{
        "time": pd.to_datetime(e.get("date"), utc=True, errors="coerce"),
        "currency": e.get("country", ""),
        "impact": IMPACT_LEVELS.get(str(e.get("impact", "")).lower(), 0),
        "event": e.get("title", ""),
        "actual": "",
        "forecast": e.get("forecast", "") or "",
        "previous": e.get("previous", "") or "",
        "outcome": None,
        "time_label": "",
    } for e in items]
    return pd.DataFrame(rows, columns=COLUMNS).dropna(subset=["time"])


def fetch_calendar(today: Optional[date] = None) -> tuple[pd.DataFrame, bool]:
    """Last, this and next week's events, sorted by time; the flag says whether released values are included."""
    today = today or pd.Timestamp.now(tz="UTC").date()
    frames = []
    try:
        for offset in (-7, 0, 7):
            frames.append(fetch_week_page(today + timedelta(days=offset)))
        with_actuals = True
    except (requests.RequestException, ValueError):
        resp = requests.get(FEED_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        frames = [feed_rows(resp.json())]
        with_actuals = False
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time", "currency", "event"]).sort_values(["time", "impact"], ascending=[True, False])
    return df.reset_index(drop=True), with_actuals
