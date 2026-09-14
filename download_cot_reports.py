"""Download CFTC Commitment of Traders (COT) currency futures positioning,
2008-present. Free, official, weekly. Shows how large speculators
(non-commercial traders) are positioned in each currency -- a classic
sentiment signal that often shifts *before* a news-driven price move.

Source: CFTC legacy "Futures Only" report, one zip per year, unbroken
format since well before 2008: https://www.cftc.gov/files/dea/history/

Output:
  data/news/cot/cot_currency_positioning.csv
    date, currency, open_interest, noncommercial_long, noncommercial_short,
    noncommercial_net, commercial_net
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(__file__).resolve().parent / "data" / "news" / "cot"
START_YEAR = 2008
BASE_URL = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"

# CFTC "Market and Exchange Names" -> currency code. Cross-rate contracts
# (EUR/GBP, EUR/JPY) are skipped -- they mix two currencies' positioning.
CURRENCY_MARKETS = {
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "EUR",
    "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE": "GBP",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "GBP",  # renamed ~2023
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "JPY",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "CHF",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "CAD",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "AUD",
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE": "NZD",
    "U.S. DOLLAR INDEX - ICE FUTURES U.S.": "USD",
}


def fetch_year(year: int) -> list[dict]:
    url = BASE_URL.format(year=year)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {year}: {exc}")
        return []
    if resp.status_code != 200:
        print(f"[FAIL] {year}: HTTP {resp.status_code}")
        return []

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    text = zf.read(zf.namelist()[0]).decode("latin1")
    rows = []
    for row in csv.DictReader(text.splitlines()):
        currency = CURRENCY_MARKETS.get(row["Market and Exchange Names"].strip())
        if currency is None:
            continue
        try:
            nc_long = int(row["Noncommercial Positions-Long (All)"])
            nc_short = int(row["Noncommercial Positions-Short (All)"])
            c_long = int(row["Commercial Positions-Long (All)"])
            c_short = int(row["Commercial Positions-Short (All)"])
            open_interest = int(row["Open Interest (All)"])
        except (KeyError, ValueError):
            continue
        rows.append(
            {
                "date": row["As of Date in Form YYYY-MM-DD"],
                "currency": currency,
                "open_interest": open_interest,
                "noncommercial_long": nc_long,
                "noncommercial_short": nc_short,
                "noncommercial_net": nc_long - nc_short,
                "commercial_net": c_long - c_short,
            }
        )
    print(f"[OK] {year}: {len(rows)} satır")
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current_year = date.today().year
    all_rows = []
    for year in range(START_YEAR, current_year + 1):
        all_rows.extend(fetch_year(year))

    if not all_rows:
        print("HATA: hiç veri indirilemedi.")
        return

    df = pd.DataFrame(all_rows).sort_values(["currency", "date"])
    out_path = OUT_DIR / "cot_currency_positioning.csv"
    df.to_csv(out_path, index=False)
    print(f"\nToplam {len(df)} satır -> {out_path}")


if __name__ == "__main__":
    main()
