"""Fixed trade-plan levels, quote observations and a risk/reward candle chart.

Reference prices only. No signal generation, broker execution or inferred fills.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from forex_config import get_pip_size, normalize_symbol, price_decimals


def utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Geçerli bir zaman gerekli.")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def positive(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (ValueError, TypeError):
        return False


def validate_plan(plan):
    if not plan.get("symbol") or not plan.get("provider") or plan.get("side") not in {"LONG", "SHORT"}:
        raise ValueError("Parite, veri kaynağı ve LONG/SHORT yönü gerekli.")
    if not all(positive(plan.get(key)) for key in ("entry", "stop", "target")):
        raise ValueError("Giriş, stop ve hedef sıfırdan büyük geçerli fiyatlar olmalı.")
    sign = 1 if plan["side"] == "LONG" else -1
    if sign * (float(plan["entry"]) - float(plan["stop"])) <= 0 or sign * (float(plan["target"]) - float(plan["entry"])) <= 0:
        raise ValueError("LONG için stop < giriş < hedef; SHORT için hedef < giriş < stop olmalı.")
    utc(plan["tracked_at"])


def make_plan(symbol, provider, side, entry, stop, target, *, now, lot=None, source="manual"):
    plan = {"symbol": normalize_symbol(symbol), "provider": str(provider), "side": side,
            "entry": float(entry), "stop": float(stop), "target": float(target),
            "tracked_at": utc(now).isoformat(), "source": source,
            "lot": float(lot) if positive(lot) else None,
            "direction_timeframes": "4H / 1H", "entry_timeframes": "15M / 5M"}
    validate_plan(plan)
    return plan


def plan_from_reading(reading, provider, *, now):
    return make_plan(reading["symbol"], provider, reading["side"], reading.get("entry"),
                     reading.get("stop"), reading.get("target"), now=now,
                     lot=reading.get("lot"), source="pivot")


def observe_plan(plan, quote, *, now):
    """Only observe the current reference price; never assert a TP/SL fill.

    A quote must be fresh, not from before tracking, and from the saved source.
    Targets touched between observations cannot be inferred from a last quote.
    """
    validate_plan(plan)
    result = {"status": "Güncel fiyat bekleniyor", "fresh": False, "price": None,
              "move_pips": None, "target_distance_pips": None, "quote_time": None,
              "elapsed_minutes": max(0., (utc(now) - utc(plan["tracked_at"])).total_seconds() / 60)}
    if not quote or (quote.get("provider") is not None and quote["provider"] != plan["provider"]):
        return result
    price = quote.get("price")
    if not positive(price):
        return result
    try:
        at = utc(quote.get("time")) if quote.get("time") is not None else None
    except (ValueError, TypeError, OverflowError):
        return result
    age = (utc(now) - at).total_seconds() if at is not None else math.inf
    if age < 0:
        result["status"] = "Fiyat zamanı geçersiz"
        return result
    result.update(price=float(price), quote_time=at)
    if not 0 <= age <= 120:
        result["status"] = "Fiyat eski · takip güncel değil"
        return result
    # The opening quote may precede the click by a few seconds. Do not infer any
    # post-tracking movement from that quote, while still displaying its price.
    if at < utc(plan["tracked_at"]):
        result["status"] = "Takip sonrası yeni fiyat bekleniyor"
        return result
    sign = 1 if plan["side"] == "LONG" else -1
    pip = get_pip_size(plan["symbol"])
    move = sign * (float(price) - plan["entry"])
    target = sign * (plan["target"] - float(price))
    result.update(fresh=True, move_pips=move / pip, target_distance_pips=target / pip)
    if sign * (float(price) - plan["stop"]) <= 0:
        result["status"] = "Fiyat stop bölgesinde"
    elif target <= 0:
        result["status"] = "Fiyat hedef bölgesinde"
    elif result["elapsed_minutes"] >= 240:
        result["status"] = "4 saat doldu · planı gözden geçir"
    else:
        result["status"] = "İzleniyor"
    return result


def position_figure(bars, plan, *, quote=None, now=None):
    """5M candles plus true-price risk/reward zones; no fabricated future path."""
    validate_plan(plan)
    now = utc(now if now is not None else pd.Timestamp.now(tz="UTC"))
    start = utc(plan["tracked_at"])
    frame = pd.DataFrame() if bars is None else bars.copy()
    columns = ["Open", "High", "Low", "Close"]
    if not frame.empty and set(columns).issubset(frame):
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame[~frame.index.isna()].sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1)
        valid &= (values.High >= values[["Open", "Close", "Low"]].max(axis=1)) & (values.Low <= values[["Open", "Close", "High"]].min(axis=1))
        frame = values[valid & (frame.index <= now)].tail(120)
        # Do not put months-old archive candles beside a new watch plan.
        frame = frame[frame.index >= start - pd.Timedelta(hours=12)]
    else:
        frame = pd.DataFrame()
    last = frame.index[-1] if not frame.empty else start
    chart_now = now if now - last <= pd.Timedelta(hours=12) else last
    end = max(chart_now, start, last) + pd.Timedelta(minutes=40)
    # Keep the first few candles of a long-running plan accessible by zooming;
    # default view follows the last 120 candles and doesn't compress days of data.
    left = frame.index[0] if not frame.empty else start - pd.Timedelta(hours=2)
    zone_start = max(start, left)
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(go.Candlestick(x=frame.index.tz_convert("Europe/Istanbul"), open=frame.Open,
            high=frame.High, low=frame.Low, close=frame.Close, name="5M fiyat",
            increasing=dict(line_color="#00d4a5", fillcolor="#00d4a5"),
            decreasing=dict(line_color="#f36b80", fillcolor="#f36b80")))
    else:
        fig.add_annotation(x=.02, y=.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
            text="5M mum verisi alınamadı · yalnız kayıtlı seviyeler", showarrow=False, font_color="#e9b95b")
    x0, x1 = zone_start.tz_convert("Europe/Istanbul"), end.tz_convert("Europe/Istanbul")
    for limit, color in ((plan["stop"], "rgba(243,107,128,0.22)"), (plan["target"], "rgba(0,212,165,0.20)")):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=min(plan["entry"], limit),
                      y1=max(plan["entry"], limit), fillcolor=color, line_width=0, layer="below")
    decimals = price_decimals(plan["symbol"])
    for label, value, color in (("STOP", plan["stop"], "#ff8a9e"),
            ("GİRİŞ", plan["entry"], "#e8f1f8"), ("HEDEF", plan["target"], "#25dfb7")):
        offset = 0 if label == "GİRİŞ" else (14 if value > plan["entry"] else -14)
        fig.add_shape(type="line", x0=x0, x1=x1, y0=value, y1=value,
                      line=dict(color=color, width=1.3, dash="solid" if label == "GİRİŞ" else "dot"))
        fig.add_annotation(x=x1, y=value, text=f"{label}  {value:.{decimals}f}",
            xanchor="right", yshift=offset, showarrow=False, font=dict(color=color, size=12),
            bgcolor="#081722", borderpad=3)
    observation = observe_plan(plan, quote or {}, now=now)
    if observation["price"] is not None:
        price = observation["price"]
        fig.add_hline(y=price, line_color="#e9b95b", line_dash="dash", line_width=1)
        stamp = observation["quote_time"]
        if stamp is not None:
            fig.add_trace(go.Scatter(x=[stamp.tz_convert("Europe/Istanbul")], y=[price], mode="markers",
                marker=dict(size=8, color="#e9b95b", line=dict(width=2, color="#081722")),
                name="Güncel fiyat" if observation["fresh"] else "Son alınan fiyat"))
    levels = [plan["entry"], plan["stop"], plan["target"]]
    if not frame.empty:
        levels += [float(frame.Low.min()), float(frame.High.max())]
    if observation["price"] is not None:
        levels.append(observation["price"])
    extent = max(max(levels) - min(levels), get_pip_size(plan["symbol"]) * 5)
    fig.update_layout(height=460, paper_bgcolor="#081722", plot_bgcolor="#081722",
        margin=dict(l=8, r=64, t=32, b=32), showlegend=False, dragmode="pan",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#b2c4d3"),
        hovermode="x unified", uirevision=f"{plan['symbol']}:{plan['tracked_at']}",
        xaxis=dict(range=[left.tz_convert("Europe/Istanbul"), x1], rangeslider_visible=False,
                   showgrid=False, tickformat="%H:%M", title="İstanbul saati · 5M"),
        yaxis=dict(side="right", range=[min(levels) - extent * .12, max(levels) + extent * .14],
                   gridcolor="#142b39", tickformat=f".{decimals}f", zeroline=False))
    return fig
