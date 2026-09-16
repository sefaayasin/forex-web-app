"""Aggregate data/news/events_with_impact.csv into a small per-(event, pair,
direction) historical tendency table, for a read-only "haber yön eğilimi"
panel in the live app.

Direction is defined by the sign of (actual - previous) for that release —
the only surprise proxy this dataset has (no forecast/consensus column).
This is exploratory, non-causal history, not a prediction; the app must
label it as such and must not use it to gate or place trades on its own.

Run after analyze_news_impact.py has produced events_with_impact.csv.
Output: data/news/event_direction_tendency.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forex_decision_core import stationary_bootstrap_mean_test

NEWS_DIR = Path(__file__).resolve().parent / "data" / "news"
MIN_SAMPLES = 8


def main() -> None:
    impact_path = NEWS_DIR / "events_with_impact.csv"
    if not impact_path.exists():
        print(f"HATA: {impact_path} yok. Önce analyze_news_impact.py çalıştırın.")
        return

    df = pd.read_csv(impact_path)
    df = df.dropna(subset=["change", "ret_1d_pct", "pair", "event"])
    df["direction"] = np.where(df["change"] > 0, "ARTTI", np.where(df["change"] < 0, "AZALDI", "DEĞİŞMEDİ"))
    df = df[df["direction"] != "DEĞİŞMEDİ"]

    rows = []
    for (event, pair, direction), group in df.groupby(["event", "pair", "direction"]):
        values = group["ret_1d_pct"].tolist()
        if len(values) < MIN_SAMPLES:
            continue
        test = stationary_bootstrap_mean_test(values, simulations=2000, mean_block_length=1.0, seed=42)
        rows.append({
            "event": event,
            "currency": group["currency"].iloc[0],
            "pair": pair,
            "direction": direction,
            "n": test["sample_count"],
            "mean_ret_1d_pct": test["observed_mean"],
            "ci_low_1d_pct": test["ci_low"],
            "ci_high_1d_pct": test["ci_high"],
            "p_value": test["p_value"],
        })

    out = pd.DataFrame(rows).sort_values(["event", "pair", "direction"])
    out_path = NEWS_DIR / "event_direction_tendency.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] {len(out)} satır (event x pair x yön, n>={MIN_SAMPLES}) -> {out_path}")


if __name__ == "__main__":
    main()
