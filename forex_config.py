"""Static market universe and shared symbol conventions."""

from __future__ import annotations


SYMBOL_LIST = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    "EURGBP=X", "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURJPY=X", "EURNZD=X",
    "GBPJPY=X", "GBPAUD=X", "GBPCAD=X", "GBPCHF=X", "GBPNZD=X",
    "AUDCAD=X", "AUDCHF=X", "AUDJPY=X", "AUDNZD=X",
    "CADCHF=X", "CADJPY=X", "CHFJPY=X",
    "NZDCAD=X", "NZDCHF=X", "NZDJPY=X", "EURZAR=X",
]

MAJOR_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
]
MINOR_PAIRS = [symbol for symbol in SYMBOL_LIST if symbol not in MAJOR_PAIRS]
ALERT_PAIR_GROUPS = {"Major": MAJOR_PAIRS, "Minör": MINOR_PAIRS}

TIMEFRAMES = {
    "4 Saat": {"interval": "4h", "period": "60d", "weight": 4},
    "1 Saat": {"interval": "60m", "period": "30d", "weight": 3},
    "15 Dakika": {"interval": "15m", "period": "10d", "weight": 2},
    "5 Dakika": {"interval": "5m", "period": "5d", "weight": 1},
}

BACKTEST_PERIODS = {
    "5 Dakika": "5d",
    "15 Dakika": "60d",
    "1 Saat": "90d",
    "4 Saat": "120d",
}

PRICE_CHANGE_WINDOWS = {
    "Son 5 dakika": 5,
    "Son 10 dakika": 10,
    "Son 15 dakika": 15,
    "Son 30 dakika": 30,
    "Son 1 saat": 60,
    "Son 4 saat": 240,
    "Son 1 gün": 1440,
}

INTRADAY_CHART_WINDOWS = {
    "Son 1 saat": 60,
    "Son 2 saat": 120,
    "Son 4 saat": 240,
    "Son 8 saat": 480,
    "Son 12 saat": 720,
    "Son 24 saat": 1440,
}

TRADING_SESSIONS = {
    "Tüm Gün": None,
    "Asya": {"timezone": "Asia/Tokyo", "start": 9, "end": 18},
    "Londra": {"timezone": "Europe/London", "start": 8, "end": 17},
    "New York": {"timezone": "America/New_York", "start": 8, "end": 17},
    "Londra + New York Kesişimi": "OVERLAP",
}

# A single dual-engine laboratory run evaluates exactly two predeclared
# hypotheses for the selected symbol: TREND and RANGE.
EDGE_DUAL_ENGINE_TRIALS = 2


def normalize_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().upper().replace("/", "")
    if normalized and not normalized.endswith("=X") and len(normalized) == 6:
        normalized += "=X"
    return normalized


def symbol_pair(symbol: str) -> tuple[str, str]:
    normalized = str(symbol).upper().replace("=X", "").replace("/", "").replace("-", "")
    return normalized[:3], normalized[-3:]


def get_pip_size(symbol: str) -> float:
    base, quote = symbol_pair(symbol)
    if quote == "JPY":
        return 0.01
    if base in {"XAU", "XAG"}:
        return 0.1
    if base in {"BTC", "ETH"} or quote in {"BTC", "ETH"}:
        return 1.0
    return 0.0001


def price_decimals(symbol: str) -> int:
    _, quote = symbol_pair(symbol)
    return 3 if quote == "JPY" else 5
