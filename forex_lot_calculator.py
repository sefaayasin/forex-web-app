"""Lot / stop / take-profit calculator for a given account balance.

Pure arithmetic: it converts a balance, a risk percentage and stop/target
distances in pips into a lot size, dollar outcomes and price levels. It says
nothing about whether price will reach the target. The round-trip trading cost
(spread + commission) is paid once per trade, so it widens the loss at the stop
and shrinks the profit at the target. No Streamlit dependency so it can be
tested without starting the app.
"""
from __future__ import annotations

import math
from typing import Optional

LOT_STEP = 0.01
MIN_LOT = 0.01
COMMON_LOTS = (0.01, 0.05, 0.10, 0.30, 0.50, 1.00)

# Share of the balance lost if the stop is hit -> label shown to the user.
RISK_LEVELS = ((1.0, "Makul"), (2.0, "Dikkat"), (5.0, "Yüksek"), (math.inf, "Çok tehlikeli"))


def floor_lot(lot: float) -> float:
    """Round down to the broker's 0.01 lot step, so the real risk never exceeds the plan."""
    if not math.isfinite(lot) or lot <= 0:
        return 0.0
    return math.floor(lot / LOT_STEP + 1e-9) * LOT_STEP


def ceil_lot(lot: float) -> float:
    """Round up to the 0.01 lot step, so a profit target is actually reached."""
    if not math.isfinite(lot) or lot <= 0:
        return 0.0
    return math.ceil(lot / LOT_STEP - 1e-9) * LOT_STEP


def lot_for_risk(balance: float, risk_pct: float, stop_pips: float, cost_pips: float, pip_value_per_lot: float) -> float:
    """Raw (unrounded) lot that loses `risk_pct` of `balance` when the stop is hit, cost included."""
    risk_pips = stop_pips + max(cost_pips, 0.0)
    if balance <= 0 or risk_pct <= 0 or risk_pips <= 0 or pip_value_per_lot <= 0:
        return 0.0
    return balance * risk_pct / 100.0 / (risk_pips * pip_value_per_lot)


def lot_for_profit(target_usd: float, target_pips: float, cost_pips: float, pip_value_per_lot: float) -> Optional[float]:
    """Lot that earns `target_usd` net of cost when price moves `target_pips`; None if the move cannot cover the cost."""
    net_pips = target_pips - max(cost_pips, 0.0)
    if target_usd <= 0 or net_pips <= 0 or pip_value_per_lot <= 0:
        return None
    return target_usd / (net_pips * pip_value_per_lot)


def pips_for_profit(target_usd: float, lot: float, cost_pips: float, pip_value_per_lot: float) -> Optional[float]:
    """Price move in pips a position of `lot` needs to earn `target_usd` net of cost."""
    if target_usd <= 0 or lot <= 0 or pip_value_per_lot <= 0:
        return None
    return target_usd / (lot * pip_value_per_lot) + max(cost_pips, 0.0)


def margin_usd(lot: float, price: float, pip_size: float, pip_value_per_lot: float, leverage: float) -> Optional[float]:
    """Approximate margin in USD. price * pip_value / pip_size is the USD notional of one lot for any pair."""
    if lot <= 0 or price <= 0 or pip_size <= 0 or pip_value_per_lot <= 0 or leverage <= 0:
        return None
    return lot * price * pip_value_per_lot / pip_size / leverage


def risk_level(risk_pct_of_balance: float) -> str:
    for limit, label in RISK_LEVELS:
        if risk_pct_of_balance <= limit:
            return label
    return RISK_LEVELS[-1][1]


def trade_outcome(balance: float, lot: float, stop_pips: float, target_pips: float, cost_pips: float, pip_value_per_lot: float) -> dict:
    """Dollar result at the stop and at the target for a given lot."""
    cost = max(cost_pips, 0.0)
    per_pip = lot * pip_value_per_lot
    loss = per_pip * (stop_pips + cost)
    profit = per_pip * (target_pips - cost)
    loss_pct = loss / balance * 100.0 if balance > 0 else math.nan
    return {
        "lot": lot,
        "usd_per_pip": per_pip,
        "loss_usd": loss,
        "profit_usd": profit,
        "loss_pct": loss_pct,
        "profit_pct": profit / balance * 100.0 if balance > 0 else math.nan,
        "risk_level": risk_level(loss_pct) if math.isfinite(loss_pct) else "-",
    }


def price_levels(entry: float, side: str, stop_pips: float, target_pips: float, pip_size: float) -> tuple[float, float]:
    """(stop, take-profit) prices for a LONG or SHORT entry."""
    direction = 1 if side == "LONG" else -1
    return entry - direction * stop_pips * pip_size, entry + direction * target_pips * pip_size


def lot_comparison(
    balance: float,
    stop_pips: float,
    target_pips: float,
    cost_pips: float,
    pip_value_per_lot: float,
    target_usd: float,
    extra_lots: tuple[float, ...] = (),
) -> list[dict]:
    """One row per lot size: what the stop costs, what the target pays, and how far price must move for `target_usd`."""
    lots = sorted({round(lot, 2) for lot in (*COMMON_LOTS, *extra_lots) if lot >= MIN_LOT})
    rows = []
    for lot in lots:
        row = trade_outcome(balance, lot, stop_pips, target_pips, cost_pips, pip_value_per_lot)
        row["pips_for_target"] = pips_for_profit(target_usd, lot, cost_pips, pip_value_per_lot)
        rows.append(row)
    return rows
