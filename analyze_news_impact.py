"""Join macro/news events (data/news/events.csv) with daily FX history
(data/historical/<PAIR>.csv) to measure real price impact, and surface
simple cross-event connections (same-week clustering, rate-decision
correlation between countries).

Run download_news_events.py first to produce data/news/events.csv.

Output:
  data/news/events_with_impact.csv   each event + forward return (1d/3d/5d)
                                      for every pair touching that currency
  data/news/connections.csv          pairs of events that fell close in
                                      time, for spotting causal chains
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forex_config import SYMBOL_LIST

NEWS_DIR = Path(__file__).resolve().parent / "data" / "news"
HIST_DIR = Path(__file__).resolve().parent / "data" / "historical"
FORWARD_DAYS = (1, 3, 5)
CONNECTION_WINDOW_DAYS = 2  # events within this many days are "linked"


def pairs_for_currency(currency: str) -> list[str]:
    code = currency.upper()
    out = []
    for symbol in SYMBOL_LIST:
        clean = symbol.replace("=X", "")
        if code in (clean[:3], clean[3:6]):
            out.append(clean)
    return out


def load_price(pair: str) -> pd.DataFrame | None:
    path = HIST_DIR / f"{pair}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date").sort_index()


def compute_impact(events: pd.DataFrame) -> pd.DataFrame:
    price_cache: dict[str, pd.DataFrame] = {}
    rows = []
    for _, ev in events.iterrows():
        ev_date = pd.Timestamp(ev["date"])
        for pair in pairs_for_currency(ev["currency"]):
            if pair not in price_cache:
                price_cache[pair] = load_price(pair)
            price = price_cache[pair]
            if price is None:
                continue
            after = price[price.index >= ev_date]
            if after.empty:
                continue
            base_close = after["Close"].iloc[0]
            row = dict(ev)
            row["pair"] = pair
            for n in FORWARD_DAYS:
                if len(after) > n:
                    fwd_close = after["Close"].iloc[n]
                    row[f"ret_{n}d_pct"] = (fwd_close - base_close) / base_close * 100
                else:
                    row[f"ret_{n}d_pct"] = None
            rows.append(row)
    return pd.DataFrame(rows)


def find_connections(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values("date").reset_index(drop=True)
    events["date"] = pd.to_datetime(events["date"])
    links = []
    for i, a in events.iterrows():
        window = events[
            (events["date"] > a["date"])
            & (events["date"] <= a["date"] + pd.Timedelta(days=CONNECTION_WINDOW_DAYS))
            & (events.index != i)
        ]
        for _, b in window.iterrows():
            links.append(
                {
                    "date_a": a["date"].date(),
                    "event_a": a["event"],
                    "currency_a": a["currency"],
                    "date_b": b["date"].date(),
                    "event_b": b["event"],
                    "currency_b": b["currency"],
                    "days_apart": (b["date"] - a["date"]).days,
                }
            )
    return pd.DataFrame(links)


def main() -> None:
    events_path = NEWS_DIR / "events.csv"
    if not events_path.exists():
        print(f"HATA: {events_path} yok. Önce download_news_events.py çalıştırın.")
        return

    events = pd.read_csv(events_path)

    impact = compute_impact(events)
    impact_path = NEWS_DIR / "events_with_impact.csv"
    impact.to_csv(impact_path, index=False)
    print(f"[OK] {len(impact)} etki satırı -> {impact_path}")

    connections = find_connections(events)
    conn_path = NEWS_DIR / "connections.csv"
    connections.to_csv(conn_path, index=False)
    print(f"[OK] {len(connections)} bağlantı -> {conn_path}")


if __name__ == "__main__":
    main()
