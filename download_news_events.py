"""Download macro/news event history (2008-present) that moves FX pairs.

Data source: FRED (Federal Reserve Economic Data) official API.
FRED is free, official, and covers policy rates + macro releases back to
2008 (and long before). It does not provide "forecast vs actual" style
economic-calendar rows -- instead we treat every discrete level change in
a policy-rate series as a decision event, and every new print of a macro
series (CPI, payrolls, GDP, ...) as a release event.

Requires a free API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set it via the FRED_API_KEY environment variable before running.

Output:
  data/news/fred/<series_id>.csv       raw series history
  data/news/events.csv                 one row per decision/release event
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(__file__).resolve().parent / "data" / "news"
RAW_DIR = OUT_DIR / "fred"
START_DATE = "2008-01-01"
API_URL = "https://api.stlouisfed.org/fred/series/observations"

# series_id -> (country/currency, human label, event_type)
# event_type "rate": every level change is one policy-decision event.
# event_type "release": every new observation is one data-release event.
SERIES = {
    "DFEDTARU": ("USD", "Fed Funds Target Rate (Upper)", "rate"),
    "CPIAUCSL": ("USD", "US CPI (headline)", "release"),
    "CPILFESL": ("USD", "US Core CPI", "release"),
    "PAYEMS": ("USD", "US Nonfarm Payrolls", "release"),
    "UNRATE": ("USD", "US Unemployment Rate", "release"),
    "GDP": ("USD", "US GDP", "release"),
    "PCEPI": ("USD", "US PCE Price Index", "release"),
    "RSAFS": ("USD", "US Retail Sales", "release"),
    "INDPRO": ("USD", "US Industrial Production", "release"),
    "UMCSENT": ("USD", "US Michigan Consumer Sentiment", "release"),
    "ECBDFR": ("EUR", "ECB Deposit Facility Rate", "rate"),
    "IRSTCI01GBM156N": ("GBP", "UK Interbank Rate", "rate"),
    "IRSTCI01EZM156N": ("EUR", "Euro Area Interbank Rate", "rate"),
    "IRSTCI01JPM156N": ("JPY", "Japan Interbank Rate", "rate"),
    "IRSTCI01CAM156N": ("CAD", "Canada Interbank Rate", "rate"),
    "IRSTCI01AUM156N": ("AUD", "Australia Interbank Rate", "rate"),
    "IRSTCI01NZM156N": ("NZD", "New Zealand Interbank Rate", "rate"),
    "IRSTCI01CHM156N": ("CHF", "Switzerland Interbank Rate", "rate"),
    "VIXCLS": ("USD", "VIX Volatility Index", "spike"),
    "DCOILWTICO": ("USD", "WTI Crude Oil Price", "spike"),
}

# event_type "spike": flag days where the daily % change is an outlier
# relative to the series' own history (risk-on/risk-off shocks, oil shocks).
SPIKE_STD_MULTIPLIER = 3.0


def fetch_series(api_key: str, series_id: str) -> pd.DataFrame | None:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": START_DATE,
    }
    try:
        resp = requests.get(API_URL, params=params, timeout=20)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {series_id}: {exc}")
        return None
    if resp.status_code != 200:
        print(f"[FAIL] {series_id}: HTTP {resp.status_code} {resp.text[:150]}")
        return None
    obs = resp.json().get("observations", [])
    rows = [(o["date"], o["value"]) for o in obs if o["value"] != "."]
    if not rows:
        print(f"[FAIL] {series_id}: veri yok")
        return None
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["value"] = df["value"].astype(float)
    return df


def build_events(series_id: str, df: pd.DataFrame, currency: str, label: str, kind: str) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["previous"] = df["value"].shift(1)
    if kind == "rate":
        df = df[df["value"] != df["previous"]]
    elif kind == "spike":
        pct_change = (df["value"] / df["previous"] - 1) * 100
        threshold = pct_change.std() * SPIKE_STD_MULTIPLIER
        df = df[pct_change.abs() > threshold]
    df = df.dropna(subset=["previous"])
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "date": df["date"],
            "currency": currency,
            "event": label,
            "series_id": series_id,
            "event_type": kind,
            "actual": df["value"],
            "previous": df["previous"],
            "change": df["value"] - df["previous"],
        }
    )


def main() -> None:
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        print("HATA: FRED_API_KEY ortam değişkeni tanımlı değil.")
        print("Ücretsiz key al: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_events = []
    results = []
    for series_id, (currency, label, kind) in SERIES.items():
        df = fetch_series(api_key, series_id)
        time.sleep(0.3)
        if df is None:
            results.append((series_id, False))
            continue
        df.to_csv(RAW_DIR / f"{series_id}.csv", index=False)
        events = build_events(series_id, df, currency, label, kind)
        if not events.empty:
            all_events.append(events)
        results.append((series_id, True))
        print(f"[OK] {series_id} ({label}): {len(df)} gözlem, {len(events)} olay")

    if all_events:
        combined = pd.concat(all_events, ignore_index=True).sort_values("date")
        combined.to_csv(OUT_DIR / "events.csv", index=False)
        print(f"\nToplam {len(combined)} olay -> {OUT_DIR / 'events.csv'}")

    failed = [s for s, ok in results if not ok]
    print(f"\n--- Özet --- Toplam: {len(results)}, Başarılı: {len(results) - len(failed)}, Başarısız: {len(failed)}")
    if failed:
        print("Başarısız seriler:", ", ".join(failed))


if __name__ == "__main__":
    main()
