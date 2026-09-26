"""Trading-cost helpers based on measured spreads (measure_spreads.py).

The profile holds Dukascopy's raw ECN spread (no commission) per pair and New
York local hour (sessions and the 5 pm ET rollover follow New York time across
DST). It is a realistic floor for a good account; retail "standard" accounts
usually quote wider spreads instead of charging a commission. No Streamlit
dependency so it can be tested without starting the app.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
SPREAD_PROFILE_PATH = ROOT / "data/ml/spread_profile.json"
# Dukascopy-style ECN commission: $35 per $1M per side, ~0.7 pip round trip on USD-quoted majors.
ECN_COMMISSION_PIPS = 0.7
HIGH_COST_SHARE = 0.15


def current_ny_hour(now: Optional[pd.Timestamp] = None) -> int:
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    return int(now.tz_convert("America/New_York").hour)


def load_spread_profile(path: Path = SPREAD_PROFILE_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def measured_spread_pips(profile: dict, symbol: str, ny_hour: int, quantile: str = "median") -> Optional[float]:
    """Measured spread for `symbol` at New York hour `ny_hour`, or None when the profile has no usable value."""
    pair = str(symbol).replace("=X", "").upper()
    hours = profile.get("pairs", {}).get(pair, {}).get(f"{quantile}_by_ny_hour")
    if not hours or not 0 <= int(ny_hour) < len(hours):
        return None
    value = hours[int(ny_hour)]
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def estimated_round_trip_cost_pips(profile: dict, symbol: str, ny_hour: int) -> Optional[float]:
    """Median ECN spread at this New York hour plus a typical ECN commission, in pips."""
    spread = measured_spread_pips(profile, symbol, ny_hour)
    return None if spread is None else round(spread + ECN_COMMISSION_PIPS, 1)


def cost_share_of_risk(cost_pips: float, stop_pips: float) -> Optional[float]:
    """Share of the planned loss that is pure trading cost: cost / (stop + cost)."""
    if stop_pips is None or cost_pips is None or stop_pips <= 0 or cost_pips < 0:
        return None
    return cost_pips / (stop_pips + cost_pips)
