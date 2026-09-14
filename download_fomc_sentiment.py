"""Download FOMC post-meeting statements (2008-present) from the Fed's
own website and score each one's tone with a transparent hawkish/dovish
keyword count. The rate decision itself (hike/hold/cut) is already
captured in data/news/events.csv via FRED -- what that misses is *how*
the decision was phrased, which is often the bigger price driver (a
"hawkish hold" can move USD more than the hold itself).

Source: federalreserve.gov (official, public press releases).
  - years <= ~2020: linked from /monetarypolicy/fomchistorical{year}.htm
  - recent years:   linked from /monetarypolicy/fomccalendars.htm

Output:
  data/news/fomc_statements.csv
    date, url, hawkish_count, dovish_count, sentiment_score, word_count
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

OUT_DIR = Path(__file__).resolve().parent / "data" / "news"
START_YEAR = 2008
BASE = "https://www.federalreserve.gov"

HAWKISH_WORDS = [
    "raise", "raising", "raised", "increase", "increases", "increasing",
    "tighten", "tightening", "restrictive", "elevated inflation",
    "further increases", "firming", "hike",
]
DOVISH_WORDS = [
    "lower", "lowering", "lowered", "decrease", "decreases", "decreasing",
    "accommodative", "ease", "easing", "eased", "cut", "cutting",
    "patient", "support the economy", "downside risks",
]


def find_statement_urls() -> dict[str, str]:
    """Return {YYYYMMDD: absolute_url} for every FOMC statement found."""
    urls: dict[str, str] = {}
    current_year = date.today().year
    for year in range(START_YEAR, current_year + 1):
        try:
            resp = requests.get(
                f"{BASE}/monetarypolicy/fomchistorical{year}.htm",
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0"},
            )
        except Exception:  # noqa: BLE001
            continue
        if resp.status_code != 200:
            continue
        for match in re.finditer(r'href="(/[^"]*monetary[^"]*(\d{8})a\.htm)"', resp.text):
            path, ymd = match.groups()
            urls[ymd] = BASE + path

    # recent years live on the rolling calendar page instead.
    try:
        resp = requests.get(
            f"{BASE}/monetarypolicy/fomccalendars.htm",
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code == 200:
            for match in re.finditer(r'href="(/[^"]*pressreleases/monetary(\d{8})a\.htm)"', resp.text):
                path, ymd = match.groups()
                urls[ymd] = BASE + path
    except Exception:  # noqa: BLE001
        pass

    return urls


def score_statement(text: str) -> tuple[int, int, int]:
    lower = text.lower()
    hawkish = sum(lower.count(w) for w in HAWKISH_WORDS)
    dovish = sum(lower.count(w) for w in DOVISH_WORDS)
    word_count = len(lower.split())
    return hawkish, dovish, word_count


def fetch_statement_text(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    except Exception:  # noqa: BLE001
        return None
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    article = soup.find("div", id="article") or soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
    return (article or soup).get_text(" ", strip=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = find_statement_urls()
    print(f"{len(urls)} FOMC bildiri linki bulundu.")

    rows = []
    for ymd, url in sorted(urls.items()):
        text = fetch_statement_text(url)
        if not text:
            print(f"[FAIL] {ymd}: metin alınamadı")
            continue
        hawkish, dovish, word_count = score_statement(text)
        score = (hawkish - dovish) / (hawkish + dovish + 1)
        rows.append(
            {
                "date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}",
                "url": url,
                "hawkish_count": hawkish,
                "dovish_count": dovish,
                "sentiment_score": round(score, 4),
                "word_count": word_count,
            }
        )
        print(f"[OK] {ymd}: hawkish={hawkish} dovish={dovish} score={score:.3f}")

    if not rows:
        print("HATA: hiç bildiri işlenemedi.")
        return

    df = pd.DataFrame(rows).sort_values("date")
    out_path = OUT_DIR / "fomc_statements.csv"
    df.to_csv(out_path, index=False)
    print(f"\nToplam {len(df)} bildiri -> {out_path}")


if __name__ == "__main__":
    main()
