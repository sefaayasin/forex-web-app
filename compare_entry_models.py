"""Offline comparison of two entry models and London/all-day sessions.

Uses the app's actual backtest functions without starting Streamlit or fetching
network data. Results are exploratory; this script never changes live settings.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from forex_analysis import calculate_stop_target_distances, market_structure_frame
from forex_config import TIMEFRAMES, TRADING_SESSIONS, get_pip_size
from forex_decision_core import decide_mtf_signal
from forex_indicators import add_indicators

ROOT = Path(__file__).resolve().parent


def load_backtest(fetch):
    """Compile only the backtest dependency tree; replace market I/O explicitly."""
    path = ROOT / "forex_web_app_streamlit_v14_alert_decision.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    wanted = {"run_backtest"}
    while True:
        dependencies = {n.id for name in wanted for n in ast.walk(definitions[name]) if isinstance(n, ast.Name)}
        dependencies = (dependencies & definitions.keys()) - {"fetch_ohlc"}
        if dependencies <= wanted:
            break
        wanted |= dependencies
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [n for n in tree.body if getattr(n, "name", None) in wanted], type_ignores=[])
    namespace = dict(globals(), fetch_ohlc=fetch)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["run_backtest"]


class LocalArchive:
    def __init__(self, symbol):
        self.symbol = symbol
        self.frames = {}
        self.end = self._frame("15m").index[-1]

    def _frame(self, interval):
        folders = {"15m": "historical_15m", "60m": "historical_1h", "1h": "historical_1h", "4h": "historical_4h"}
        if interval not in self.frames:
            df = pd.read_csv(ROOT / "data" / folders[interval] / f"{self.symbol}.csv")
            df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
            df.columns = [c.capitalize() for c in df.columns]
            self.frames[interval] = df.sort_index()
        return self.frames[interval]

    def fetch(self, symbol, interval, period):
        if symbol.replace("=X", "") != self.symbol:
            raise ValueError("Archive symbol mismatch")
        df = self._frame(interval)
        return df.loc[(df.index >= self.end - pd.Timedelta(days=int(period.removesuffix("d")))) & (df.index <= self.end)].copy()


def summarize(result, initial_balance):
    trades = result.trades
    if trades.empty:
        return {"trades": 0, "net_pnl_usd": 0.0, "avg_r": None, "profit_factor": None, "max_drawdown_pct": 0.0, "last_30pct_avg_r": None}
    pnl = trades["PnL"]
    r = pnl / trades["Risk Amount"].replace(0, np.nan)
    losses = -pnl[pnl < 0].sum()
    # Include starting equity, so an initial loss is counted as drawdown.
    equity = pd.concat([pd.Series([initial_balance]), result.equity["Balance"]], ignore_index=True)
    peak = equity.cummax()
    tail = r.iloc[max(1, int(len(r) * .7)):]
    return {
        "trades": len(trades), "net_pnl_usd": float(pnl.sum()),
        "avg_r": float(r.mean()),
        "profit_factor": float(pnl[pnl > 0].sum() / losses) if losses > 0 else None,
        "max_drawdown_pct": float(((peak - equity) / peak * 100).max()),
        "last_30pct_avg_r": float(tail.mean()) if len(tail) else None,
    }


def main():
    rows = []
    details = []
    for symbol in ("EURUSD", "EURCHF", "AUDCAD"):
        archive = LocalArchive(symbol)
        run = load_backtest(archive.fetch)
        for model in ("Düzeltme + Tepki", "Trend + Yapı"):
            for session in ("Londra", "Tüm Gün"):
                result = run(
                    symbol=symbol + "=X", tf_name="15 Dakika", period="60d",
                    initial_balance=10000., risk_pct=.5, rr=1.5, atr_mult=1.5,
                    signal_threshold=60, spread_pips=1.5, pip_value_per_lot=10.,
                    cooldown_bars=16, session_filter=session,
                    max_same_direction_trades=1, min_trades_required=40,
                    stop_mode="Hibrit (uzak olan)", target_mode="Sabit R",
                    swing_lookback=10, max_holding_bars=24, break_even_at_r=1.,
                    entry_model=model,
                )
                counts = list(result.diagnostics["counts"].values())
                if not all(a >= b for a, b in zip(counts, counts[1:])) or counts[-1] != len(result.trades):
                    raise AssertionError("Invalid sequential diagnostic counts")
                row = {"symbol": symbol, "model": model, "session": session, **summarize(result, 10000.)}
                rows.append(row)
                details.append({**row, "diagnostics": result.diagnostics})
                print(json.dumps(row, ensure_ascii=False), flush=True)
    output = ROOT / "research" / "entry_comparison"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
    (output / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
