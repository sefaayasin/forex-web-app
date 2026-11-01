"""ML dönüş çıkışı: %50 çizgisi, ters yönde %52 güven ve iki saat teyidi. Bot portföy motoruyla, sabit risk görünümü.

Kurallar sonuçlardan önce research/exit_confirm_20261101/protocol.json dosyasına yazıldı. Çıkış değişince sonraki
girişler de değiştiğinden varyantlar E0 ile ortak girişlerde eşleştirilir. Yalnız saat başına hizalı ML ölçülür.

    python -X utf8 research_exit_confirm.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import research_bot_backtest as bt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research" / "exit_confirm_20261101"
BIG_LOSS_R = -0.75
REENTRY_MINUTES = 60
RESAMPLES = 2000

_BASE = dict(**bt._LIVE_BASKET, trailing=False, min_conf=0.55, max_cost_share=0.10)
VARIANTS = [bt.Variant("E0 mevcut (%50 çizgisi)", **_BASE),
            bt.Variant("E1 ters güven ≥ %52", **_BASE, exit_conf=0.52),
            bt.Variant("E2 iki saat teyit", **_BASE, exit_confirm=2),
            bt.Variant("E3 ML çıkışı yok", **_BASE, ml_exit=False)]


def trade_frame(trades: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(trades)
    frame["r"] = frame["net_usd"] / frame["risk_usd"]
    frame["hold_minutes"] = (frame["closed"] - frame["opened"]) / 60e9
    frame["recent"] = frame["opened"] >= bt.CHOOSE_END.value
    return frame


def reentries(frame: pd.DataFrame, minutes: int = REENTRY_MINUTES) -> int:
    """Aynı paritede önceki kapanıştan en çok `minutes` dakika sonra açılan işlem sayısı."""
    ordered = frame.sort_values(["pair", "opened"])
    wait = ordered["opened"] - ordered.groupby("pair")["closed"].shift()
    return int((wait <= minutes * 60e9).sum())


def describe(frame: pd.DataFrame) -> dict:
    losses, worst = frame.loc[frame["r"] < 0, "r"], frame["r"].nsmallest(max(len(frame) // 100, 1))
    return {"trades": len(frame), "net_usd": float(frame["net_usd"].sum()), "mean_r": float(frame["r"].mean()),
            "mean_r_first": float(frame.loc[~frame["recent"], "r"].mean()),
            "mean_r_second": float(frame.loc[frame["recent"], "r"].mean()),
            "win_rate": float((frame["r"] > 0).mean()), "mean_loss_r": float(losses.mean()),
            "big_losses": int((frame["r"] <= BIG_LOSS_R).sum()), "worst_1pct_r": float(worst.mean()),
            "median_hold_minutes": float(frame["hold_minutes"].median()),
            "reentries": reentries(frame), "ml_exits": int(frame["reason"].eq("ML YÖN DÖNDÜ").sum())}


def paired(base: pd.DataFrame, other: pd.DataFrame, seed: int = 0) -> dict:
    """Ortak girişlerde other − base R farkı; aralık haftalık blok bootstrap."""
    common = base.merge(other, on=["pair", "opened", "side"], suffixes=("_base", "_other"))
    common["diff"] = common["r_other"] - common["r_base"]
    common["week"] = pd.to_datetime(common["opened"], utc=True).dt.tz_localize(None).dt.to_period("W")
    weeks = common.groupby("week")["diff"].agg(["sum", "size"]).to_numpy()
    rng = np.random.default_rng(seed)
    picks = weeks[rng.integers(0, len(weeks), (RESAMPLES, len(weeks)))]
    means = picks[..., 0].sum(axis=1) / picks[..., 1].sum(axis=1)
    return {"common": len(common), "share_of_base": len(common) / len(base), "changed": int((common["diff"].abs() > 1e-9).sum()),
            "diff_r": float(common["diff"].mean()), "ci95": [float(x) for x in np.percentile(means, [2.5, 97.5])],
            "diff_r_first": float(common.loc[~common["recent_base"], "diff"].mean()),
            "diff_r_second": float(common.loc[common["recent_base"], "diff"].mean())}


def _run(variant: bt.Variant) -> pd.DataFrame:
    market, events, stats = bt.load_market(parallel=False), bt.high_impact_events(), bt.load_plan_stats()
    return trade_frame(bt.run(variant, market, events, stats, fixed_risk=True)["trades"])


def main() -> None:
    with Pool(len(VARIANTS)) as pool:
        frames = pool.map(_run, VARIANTS)
    base = frames[0]
    results = [{"variant": v.name, **describe(f), "paired_with_e0": None if f is base else paired(base, f)}
               for v, f in zip(VARIANTS, frames)]
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in results:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if k != "paired_with_e0"})
        print("   ", row["paired_with_e0"])


if __name__ == "__main__":
    main()
