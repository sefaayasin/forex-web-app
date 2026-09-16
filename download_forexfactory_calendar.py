"""Download the rolling ForexFactory economic calendar (this week + next week).

Data source: https://nfs.faireconomy.media/ff_calendar_thisweek.json
This is ForexFactory's own public JSON feed (the "Fair Economy" service that
powers their embeddable calendar widget) -- not a scrape of the HTML page.
It has no auth wall and no robots.txt/Cloudflare block, unlike investing.com's
calendar, and it is widely reused by community trading tools for this reason.

Limitation: it is a *rolling* window (the current calendar week only -- the
"nextweek"/"lastweek" feed variants some tools reference 404 on this host),
not a deep history -- ForexFactory does not publish years of past calendar
data through this feed. So it is useful for live "avoid trading around an
upcoming high-impact release" filtering (see news_blackout_status() in the
Streamlit app), not for historical backtesting/training -- FRED
(download_news_events.py) remains the source for that. Re-run this script
periodically (e.g. daily, or at least every Monday) to keep it current.

Output:
  data/news/forexfactory_calendar.csv
  Columns: time,currency,title,impact,forecast,previous,actual
  This schema matches what news_blackout_status() expects, so the app can
  auto-load it instead of requiring a manual CSV upload every session.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

OUT_PATH = Path(__file__).resolve().parent / "data" / "news" / "forexfactory_calendar.csv"
FEED_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
]


def fetch_feed(url: str) -> list[dict]:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.json()


def fetch_calendar() -> pd.DataFrame:
    """Fetch the current-week calendar and return it in news_blackout_status()'s schema.

    Raises on total failure (all feed URLs unreachable) rather than returning
    an empty frame, so a caller can decide whether to fall back to a stale
    on-disk copy instead of silently showing "no news" (which would be a
    dangerous default -- a fetch failure should never look like a clear
    calendar to the trading safety filter).
    """
    rows: list[dict] = []
    errors: list[str] = []
    for url in FEED_URLS:
        try:
            rows.extend(fetch_feed(url))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
    if not rows:
        raise RuntimeError("ForexFactory takvimi indirilemedi: " + "; ".join(errors))

    df = pd.DataFrame(rows)
    df = df.rename(columns={"country": "currency"})
    for col in ("forecast", "previous", "actual"):
        if col not in df.columns:
            df[col] = ""
    df["time"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[["time", "currency", "title", "impact", "forecast", "previous", "actual"]]
    return df.drop_duplicates(subset=["time", "currency", "title"]).sort_values("time").reset_index(drop=True)


def main() -> None:
    try:
        df = fetch_calendar()
    except RuntimeError as exc:
        print(f"HATA: {exc}")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    high_count = int((df["impact"].astype(str).str.lower() == "high").sum())
    print(f"\nToplam {len(df)} olay ({high_count} yüksek etkili) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
