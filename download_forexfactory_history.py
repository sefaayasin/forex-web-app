"""Download ForexFactory weekly calendars 2010-01 .. 2024-09 for research/news_drift.

Stored apart from research/calendar_sources so earlier research inputs stay unchanged.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta

from forex_adaptive import ROOT
from forex_calendar import extract_days, week_param
from research_calendar_sources import _get

OUT = ROOT / "research/news_drift/raw_forexfactory"
START, END = date(2010, 1, 4), date(2024, 9, 23)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=START, help="a Monday")
    parser.add_argument("--end", type=date.fromisoformat, default=END)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    monday = args.start
    while monday <= args.end:
        path = OUT / f"{monday}.json"
        if not path.exists():
            try:
                page = _get(f"https://www.forexfactory.com/calendar?week={week_param(monday)}").text
                path.write_text(json.dumps(extract_days(page)), encoding="utf-8")
                print(f"[OK] {monday}", flush=True)
            except RuntimeError as exc:
                print(f"[ERR] {monday}: {exc}", flush=True)
            time.sleep(1.0)
        monday += timedelta(days=7)


if __name__ == "__main__":
    main()
