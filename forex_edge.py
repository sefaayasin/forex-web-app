"""Statistical edge-validation orchestration and reporting."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from forex_config import TIMEFRAMES, get_pip_size
from forex_decision_core import (
    bonferroni_adjust,
    circular_shift_timing_test,
    classify_edge_evidence,
    stationary_bootstrap_mean_test,
)


def _utc_index_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out.sort_index()


def build_edge_validation_report(
    symbol: str,
    tf_name: str,
    period: str,
    bt,
    cost_pips: float,
    fetch_ohlc_fn: Callable[[str, str, str], pd.DataFrame],
    horizon_bars: int = 16,
    simulations: int = 1000,
    mean_block_length: float = 5.0,
    trial_count: int = 24,
    min_trades: int = 60,
) -> dict:
    """Test trade R and entry timing against dependency-preserving nulls."""
    unavailable = {
        "label": "HESAPLANAMADI",
        "trade_count": 0,
        "average_r": np.nan,
        "r_ci_low": np.nan,
        "r_ci_high": np.nan,
        "bootstrap_p": np.nan,
        "bootstrap_p_adjusted": np.nan,
        "timing_p": np.nan,
        "timing_p_adjusted": np.nan,
        "timing_mean_pips": np.nan,
        "timing_null_p95_pips": np.nan,
        "timing_percentile": np.nan,
        "blockers": ["Edge istatistik modülü kullanılamıyor"],
        "text": "Edge istatistik modülü kullanılamıyor.",
    }
    if bt is None or bt.trades is None or bt.trades.empty:
        return {
            **unavailable,
            "label": "YETERSİZ ÖRNEK",
            "blockers": ["Backtest işlemi oluşmadı"],
            "text": "Backtest işlemi oluşmadığı için edge sınanamadı.",
        }

    trades = bt.trades.copy()
    r_values = (
        pd.to_numeric(trades.get("PnL"), errors="coerce")
        / pd.to_numeric(trades.get("Risk Amount"), errors="coerce").replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    bootstrap = stationary_bootstrap_mean_test(
        r_values.to_numpy(dtype=float),
        simulations=int(simulations),
        mean_block_length=float(mean_block_length),
        seed=42,
    )

    parameters = TIMEFRAMES[tf_name]
    raw = fetch_ohlc_fn(symbol, parameters["interval"], period)
    raw = _utc_index_df(raw).dropna(subset=["Open", "Close"]) if raw is not None and not raw.empty else pd.DataFrame()
    entry_indices: list[int] = []
    side_signs: list[float] = []
    if not raw.empty and "Entry Time" in trades.columns and "Side" in trades.columns:
        entry_times = pd.to_datetime(trades["Entry Time"], utc=True, errors="coerce")
        for timestamp, side in zip(entry_times, trades["Side"].astype(str)):
            if pd.isna(timestamp):
                continue
            index = int(raw.index.searchsorted(timestamp, side="left"))
            if index < len(raw):
                entry_indices.append(index)
                side_signs.append(1.0 if side.upper() == "LONG" else -1.0)

    timing = circular_shift_timing_test(
        raw["Open"].to_numpy(dtype=float) if not raw.empty else [],
        raw["Close"].to_numpy(dtype=float) if not raw.empty else [],
        entry_indices,
        side_signs,
        horizon_bars=int(horizon_bars),
        pip_size=get_pip_size(symbol),
        cost_pips=float(cost_pips),
        simulations=int(simulations),
        seed=43,
    )

    raw_bootstrap_p = float(bootstrap.get("p_value", np.nan))
    raw_timing_p = float(timing.get("p_value", np.nan))
    adjusted_bootstrap_p = bonferroni_adjust(raw_bootstrap_p, int(trial_count))
    adjusted_timing_p = bonferroni_adjust(raw_timing_p, int(trial_count))
    average_r = float(bootstrap.get("observed_mean", np.nan))
    ci_low = float(bootstrap.get("ci_low", np.nan))
    ci_high = float(bootstrap.get("ci_high", np.nan))
    label, blockers = classify_edge_evidence(
        trade_count=len(r_values),
        average_r=average_r,
        r_ci_low=ci_low,
        bootstrap_p_adjusted=adjusted_bootstrap_p,
        timing_p_adjusted=adjusted_timing_p,
        min_trades=int(min_trades),
        alpha=0.05,
    )
    text = (
        f"{len(r_values)} işlem; ortalama {average_r:.3f}R, %95 GA [{ci_low:.3f}, {ci_high:.3f}]. "
        f"Düzeltilmiş bootstrap p={adjusted_bootstrap_p:.3f}, zamanlama p={adjusted_timing_p:.3f}."
    )
    if blockers:
        text += " Engeller: " + "; ".join(blockers) + "."
    return {
        "label": label,
        "trade_count": len(r_values),
        "average_r": average_r,
        "r_ci_low": ci_low,
        "r_ci_high": ci_high,
        "bootstrap_p": raw_bootstrap_p,
        "bootstrap_p_adjusted": adjusted_bootstrap_p,
        "timing_p": raw_timing_p,
        "timing_p_adjusted": adjusted_timing_p,
        "timing_mean_pips": float(timing.get("observed_mean_pips", np.nan)),
        "timing_null_p95_pips": float(timing.get("null_p95_pips", np.nan)),
        "timing_percentile": float(timing.get("percentile", np.nan)),
        "horizon_bars": int(horizon_bars),
        "trial_count": int(trial_count),
        "blockers": blockers,
        "text": text,
    }


def edge_validation_table(reports: dict[str, dict]) -> pd.DataFrame:
    def fmt(value: object, digits: int = 3) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "-"
        return "-" if not np.isfinite(number) else f"{number:.{digits}f}"

    names = {
        "TREND": "Trend: düzeltme + tepki",
        "RANGE": "Yatay: Bollinger orta banda dönüş",
    }
    rows = []
    for engine in ["TREND", "RANGE"]:
        report = reports.get(engine, {})
        rows.append({
            "Motor": names[engine],
            "Edge Kanıtı": report.get("label", "HESAPLANAMADI"),
            "İşlem": int(report.get("trade_count", 0) or 0),
            "Ortalama R": fmt(report.get("average_r")),
            "%95 R Aralığı": f"{fmt(report.get('r_ci_low'))} – {fmt(report.get('r_ci_high'))}",
            "Bootstrap p (düz.)": fmt(report.get("bootstrap_p_adjusted")),
            "Zamanlama p (düz.)": fmt(report.get("timing_p_adjusted")),
            "Gerçek ufuk pips": fmt(report.get("timing_mean_pips"), 2),
            "Null %95 pips": fmt(report.get("timing_null_p95_pips"), 2),
            "Sonuç": report.get("text", "-"),
        })
    return pd.DataFrame(rows)
