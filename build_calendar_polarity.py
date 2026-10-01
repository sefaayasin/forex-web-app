"""Build data/calendar_polarity.json from the saved ForexFactory weeks and report how good the forecast is as a guide.

Polarity says whether a higher value is good (+1) or bad (-1) for the currency; it is
learned from ForexFactory's better/worse marks on two years of releases
(research/calendar_sources/raw/forexfactory). The report checks, on the same releases,
how often the forecast's direction against the previous value came true:
  - vs previous: did the released value land on the forecast's side of the previous value?
  - surprise:    did the released value beat/miss the forecast in that same direction?
Price reacts to the surprise, so the second number is the one that matters for a trade.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from forex_calendar import POLARITY_PATH, expected_outcome, learn_polarity, parse_value, rows_from_days

RAW_DIR = Path(__file__).resolve().parent / "research" / "calendar_sources" / "raw" / "forexfactory"


def load_history() -> pd.DataFrame:
    frames = [rows_from_days(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(RAW_DIR.glob("*.json"))]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time", "currency", "event"])


def hit_rates(events: pd.DataFrame, polarity: dict) -> pd.DataFrame:
    rows = []
    for row in events.itertuples():
        expected = expected_outcome(row, polarity)
        actual, forecast, previous = parse_value(row.actual), parse_value(row.forecast), parse_value(row.previous)
        if expected not in ("better", "worse") or actual is None:
            continue
        sign = polarity[(row.currency, row.event)] * (1 if expected == "better" else -1)
        rows.append({
            "impact": row.impact,
            "vs_previous": None if actual == previous else (actual - previous) * sign > 0,
            "surprise": None if actual == forecast else (actual - forecast) * sign > 0,
        })
    df = pd.DataFrame(rows)
    return df.groupby("impact").agg(
        n_vs_previous=("vs_previous", "count"), vs_previous=("vs_previous", "mean"),
        n_surprise=("surprise", "count"), surprise=("surprise", "mean"),
    ).sort_index(ascending=False)


def main() -> None:
    events = load_history()
    polarity = learn_polarity(events)
    items = [{"currency": c, "event": e, "polarity": p} for (c, e), p in sorted(polarity.items())]
    lines = ",\n".join(json.dumps(item, ensure_ascii=False) for item in items)
    POLARITY_PATH.write_text(f"[\n{lines}\n]\n", encoding="utf-8")
    print(f"{len(events)} releases, {len(items)} series -> {POLARITY_PATH}")
    print(hit_rates(events, polarity).round(3).to_string())


if __name__ == "__main__":
    main()
