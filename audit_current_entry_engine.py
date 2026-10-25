"""Reproduce the existing app's 365-day 15M engine audit without market I/O."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from compare_entry_models import LocalArchive, load_backtest, summarize
from research_meta_label import resample_bars

ROOT = Path(__file__).resolve().parent


class RebuiltArchive(LocalArchive):
    def _frame(self, interval):
        if interval in ("60m", "1h", "4h"):
            if interval not in self.frames:
                self.frames[interval] = resample_bars(super()._frame("15m"), "4h" if interval == "4h" else "1h")
            return self.frames[interval]
        return super()._frame(interval)


def main():
    rows = []
    for symbol in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"):
        archive = RebuiltArchive(symbol)
        run = load_backtest(archive.fetch)
        result = run(symbol=symbol + "=X", tf_name="15 Dakika", period="365d",
            initial_balance=10000., risk_pct=.5, rr=1.5, atr_mult=1.5,
            signal_threshold=60, spread_pips=1.5, pip_value_per_lot=10.,
            cooldown_bars=16, session_filter="Tüm Gün", max_same_direction_trades=1,
            min_trades_required=40, stop_mode="Hibrit (uzak olan)", target_mode="Sabit R",
            swing_lookback=10, max_holding_bars=24, break_even_at_r=1., entry_model="Düzeltme + Tepki")
        row = {"symbol": symbol, **summarize(result, 10000.), "diagnostics": result.diagnostics}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "diagnostics"}, ensure_ascii=False), flush=True)
    output = ROOT / "research/adaptive_entry"
    output.mkdir(parents=True, exist_ok=True)
    (output / "current_engine_audit.json").write_text(json.dumps({
        "period_days": 365, "entry_timeframe": "15m", "cost_pips": 1.5,
        "higher_timeframes": "Rebuilt from 15m archive",
        "note": "Existing app engine with the stated settings; distinct entries/exits from adaptive candidate research.",
        "results": rows}, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k != "diagnostics"} for r in rows]).to_csv(
        output / "current_engine_summary.csv", index=False)


if __name__ == "__main__":
    main()
