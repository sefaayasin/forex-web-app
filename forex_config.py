"""Static market universe and shared symbol conventions."""

from __future__ import annotations


SYMBOL_LIST = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
    "EURGBP=X", "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURJPY=X", "EURNZD=X",
    "GBPJPY=X", "GBPAUD=X", "GBPCAD=X", "GBPCHF=X", "GBPNZD=X",
    "AUDCAD=X", "AUDCHF=X", "AUDJPY=X", "AUDNZD=X",
    "CADCHF=X", "CADJPY=X", "CHFJPY=X",
    "NZDCAD=X", "NZDCHF=X", "NZDJPY=X", "EURZAR=X",
    "XAUUSD=X",
]

MAJOR_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X",
]
METAL_SYMBOLS = ["XAUUSD=X"]
MINOR_PAIRS = [symbol for symbol in SYMBOL_LIST if symbol not in MAJOR_PAIRS and symbol not in METAL_SYMBOLS]
ALERT_PAIR_GROUPS = {"Major": MAJOR_PAIRS, "Minör": MINOR_PAIRS, "Altın": METAL_SYMBOLS}

# Broker names for the same instrument; "GOLD/USD" and "XAU/USD" are both spot gold.
SYMBOL_ALIASES = {"GOLD": "XAUUSD", "GOLDUSD": "XAUUSD", "XAU": "XAUUSD"}
# Yahoo no longer serves spot gold (XAUUSD=X returns 404), so the app reads the
# front-month COMEX future instead. It trades a few dollars to tens of dollars
# above spot and jumps when the contract rolls.
YAHOO_TICKERS = {"XAUUSD=X": "GC=F"}

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
    normalized = str(symbol).strip().upper().replace("/", "").replace(" ", "")
    normalized = SYMBOL_ALIASES.get(normalized.removesuffix("=X"), normalized)
    if normalized and not normalized.endswith("=X") and len(normalized) == 6:
        normalized += "=X"
    return normalized


def yahoo_ticker(symbol: str) -> str:
    """Ticker Yahoo actually serves for an app symbol (spot gold -> COMEX gold future)."""
    normalized = normalize_symbol(symbol)
    return YAHOO_TICKERS.get(normalized, normalized)


def is_metal(symbol: str) -> bool:
    return symbol_pair(symbol)[0] in {"XAU", "XAG"}


def contract_size(symbol: str) -> int:
    """Units in one standard lot: 100 oz of gold, 5,000 oz of silver, 100,000 of a currency."""
    base, _ = symbol_pair(symbol)
    return {"XAU": 100, "XAG": 5_000}.get(base, 100_000)


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
    base, quote = symbol_pair(symbol)
    if base == "XAU":
        return 2
    if base == "XAG":
        return 3
    return 3 if quote == "JPY" else 5
