"""Replay the app's 15M + 5M opportunity-feed scan at past moments, with the app's own code.

The scan functions (analyse_symbol, global_bias, build_market_model_status,
scanner_opportunity_from_row and their helpers) are compiled straight from
forex_web_app_streamlit_v14_alert_decision.py; only fetch_ohlc is replaced by a
feed that returns what the app would have downloaded at time t. Like Yahoo, the
feed includes the still-forming bar as the last row, which analyse_symbol drops,
so no bar that closes after t is ever used. The 4h bars are rebuilt from hourly
bars exactly as the app does for Yahoo (label="right", closed="right").
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from forex_analysis import evaluate_bias, evaluate_market_structure, label_from_score
from forex_config import TIMEFRAMES

ROOT = Path(__file__).resolve().parent
APP = ROOT / "forex_web_app_streamlit_v14_alert_decision.py"
ENTRY_TFS = ("15 Dakika", "5 Dakika")
WANTED = {"analyse_symbol", "global_bias", "build_market_model_status", "scanner_opportunity_from_row"}
PERIOD_DAYS = {"4h": 60, "60m": 30, "15m": 10, "5m": 5}  # forex_config.TIMEFRAMES periods


def load_scan_functions(fetch_ohlc: Callable) -> dict:
    """Compile the scan functions and everything they call from the app, with `fetch_ohlc` swapped in."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    constants = {t.id: n for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                 if isinstance(t, ast.Name) and t.id.isupper()}
    wanted = set(WANTED)
    while True:
        names = {n.id for name in wanted for n in ast.walk(definitions[name]) if isinstance(n, ast.Name)}
        extra = (names & definitions.keys()) - {"fetch_ohlc"} - wanted
        if not extra:
            break
        wanted |= extra
    used_constants = {n.id for name in wanted for n in ast.walk(definitions[name]) if isinstance(n, ast.Name)} & constants.keys()
    body = [constants[c] for c in sorted(used_constants) if c not in {"TIMEFRAMES"}]
    body += [n for n in tree.body if getattr(n, "name", None) in wanted]
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + body, type_ignores=[])
    namespace = {"np": np, "pd": pd, "Optional": Optional, "TIMEFRAMES": TIMEFRAMES, "evaluate_bias": evaluate_bias,
                 "evaluate_market_structure": evaluate_market_structure, "label_from_score": label_from_score,
                 "fetch_ohlc": fetch_ohlc}
    exec(compile(module, str(APP), "exec"), namespace)
    return namespace


def resample_4h_like_app(hourly: pd.DataFrame) -> pd.DataFrame:
    return (hourly.resample("4h", label="right", closed="right")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna(subset=["Open", "High", "Low", "Close"]))


class ReplayFeed:
    """Bars as the app would have downloaded them at `self.now` (the forming bar included last)."""

    def __init__(self, bars_5m: pd.DataFrame, bars_15m: pd.DataFrame, bars_1h: pd.DataFrame):
        self.frames = {"5m": bars_5m, "15m": bars_15m, "60m": bars_1h}
        self.now: Optional[pd.Timestamp] = None

    def window(self, interval: str) -> pd.DataFrame:
        source = self.frames["60m" if interval == "4h" else interval]
        start = self.now - pd.Timedelta(days=PERIOD_DAYS[interval])
        lo, hi = source.index.searchsorted(start, side="right"), source.index.searchsorted(self.now, side="right")
        part = source.iloc[lo:hi]
        return resample_4h_like_app(part) if interval == "4h" else part

    def fetch_ohlc(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        return self.window(interval)


def scan_at(functions: dict, feed: ReplayFeed, symbol: str, when: pd.Timestamp) -> dict:
    """The feed's verdict for both entry timeframes at `when`, as run_symbol_scanner would label it."""
    feed.now = when
    summary, _ = functions["analyse_symbol"](symbol)
    out = {}
    for tf in ENTRY_TFS:
        label, score, _note = functions["global_bias"](summary, tf)
        model = functions["build_market_model_status"](summary, tf, enabled=True, entry_model="Düzeltme + Tepki")
        row = {"Genel Bias": label, "Skor": round(float(score), 1), "Backtest Kalitesi": "-"}
        opportunity, _reason, signal_score = functions["scanner_opportunity_from_row"](row, "Dengeli Sinyal")
        if opportunity != "PAS" and not model["entry_allowed"]:
            opportunity = "PAS"
        side = 1 if opportunity.startswith("LONG") else (-1 if opportunity.startswith("SHORT") else 0)
        out[tf] = {"side": side, "opportunity": opportunity, "score": round(float(signal_score), 1), "bias": label,
                   "direction_allowed": label != "İşlem Yok", "gate_passed": bool(model["entry_allowed"])}
    return out
