"""MetaTrader 5 zaman yardımcıları: uygulama ve demo botu aynı çeviriyi kullanır.

MT5 mum, tick, pozisyon ve işlem zamanları broker sunucu saatindedir, UTC değil (Pepperstone: UTC+2/+3).
Streamlit bağımlılığı yoktur.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd


def server_offset(mt5, mt5_symbol: str) -> pd.Timedelta:
    """Sunucu saati − UTC. Canlı tick'ten ölçülür.

    Piyasa kapalıyken tick eskidir; o zaman MT5 brokerlarının yaygın "New York 17:00 = sunucu 00:00"
    kuralına (New York + 7 saat) düşülür.
    """
    now = pd.Timestamp.now(tz="UTC")
    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is not None and int(tick.time) > 0:
        diff = int(tick.time) - now.timestamp()
        step = round(diff / 1800) * 1800
        if abs(diff - step) <= 120 and -12 * 3600 <= step <= 14 * 3600:
            return pd.Timedelta(seconds=step)
    return now.tz_convert("America/New_York").utcoffset() + pd.Timedelta(hours=7)


def server_time_to_utc(seconds: int, offset: pd.Timedelta) -> pd.Timestamp:
    return pd.to_datetime(int(seconds), unit="s", utc=True) - offset


def rates_utc(mt5, mt5_symbol: str, timeframe: int, start: datetime, end: datetime,
              daily: bool = False) -> pd.DataFrame:
    """UTC aralığındaki mumlar, UTC indeksli Open/High/Low/Close/Volume/Spreadpoints.

    Günlük mum etiketi zaten işlem günüdür (pivotlar bunu kullanır); yalnız gün içi mumlar kaydırılır.
    """
    offset = pd.Timedelta(0) if daily else server_offset(mt5, mt5_symbol)
    rates = mt5.copy_rates_range(mt5_symbol, timeframe, start + offset, end + offset)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(rates)
    out["time"] = pd.to_datetime(out["time"], unit="s", utc=True) - offset
    out = out[out["time"] >= pd.Timestamp(start)]
    return out.set_index("time").rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "tick_volume": "Volume", "spread": "Spreadpoints",
    })
