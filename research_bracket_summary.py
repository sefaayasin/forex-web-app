"""Combine the five 9 Oct 2026 bracket searches into research/bracket_search/summary.csv."""
from __future__ import annotations

import pandas as pd

from forex_adaptive import ROOT

OUT = ROOT / "research/bracket_search"
FAMILIES = {
    "night_bracket": ("Gece seansı ortalamaya dönüş (15M)", ["symbol", "rule", "stop_atr"]),
    "daily_bracket": ("Günlük trend, dönüş ve carry", ["symbol", "rule", "stop_atr", "hold_days"]),
    "weekend_gap": ("Hafta sonu boşluğu kapanışı", ["symbol", "entry_ny_hour", "min_gap_atr", "stop_atr"]),
    "cot_bracket": ("CFTC spekülatör pozisyonu (COT)", ["symbol", "rule", "stop_atr", "hold_days"]),
    "news_drift": ("Haber sürprizi sonrası devam", ["symbol", "min_z", "delay_min", "stop_atr"]),
    "indicator_zoo": ("30 yeni indikatör: hacim, pivot, likidite, mum (15M/1H/4H)",
        ["symbol", "timeframe", "rule", "stop_atr"]),
    "gold_daily": ("Altın günlük (Yahoo GC=F): aynı 30 indikatör + trend", ["rule", "stop_atr", "hold_days"]),
}
PERIODS = ("development", "validation", "audit")


def family_row(folder, label, keys):
    s = pd.read_csv(ROOT / "research" / folder / "summary.csv")
    wide = s.pivot_table(index=keys, columns="period", values=["trades", "mean_usd", "stress_mean_usd"])
    dev = (wide[("trades", "development")] >= 100) & (wide[("stress_mean_usd", "development")] > 0)
    val = dev & (wide[("trades", "validation")] >= 50) & (wide[("mean_usd", "validation")] > 0)

    def weighted(period, column):
        part = s[(s.period == period) & s.trades.gt(0)]
        return float((part[column] * part.trades).sum() / part.trades.sum())

    row = {"family": folder, "label": label, "configs": len(wide), "development_picks": int(dev.sum()),
        "validation_survivors": int(val.sum()),
        "audit_positive_survivors": int((wide[val][("mean_usd", "audit")] > 0).sum()),
        "median_cost_share": weighted("development", "median_cost_share")}
    for period in PERIODS:
        row[f"trades_{period}"] = int(s[s.period == period].trades.sum())
        row[f"net_usd_{period}"] = weighted(period, "mean_usd")
        row[f"gross_r_{period}"] = weighted(period, "gross_mean_r")
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([family_row(f, label, keys) for f, (label, keys) in FAMILIES.items()])
    table.to_csv(OUT / "summary.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
